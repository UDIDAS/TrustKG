"""End-to-end CORAL graph stage: materialize the ensemble triples into RDF and run
SPARQL cohort queries. PER COHORT (PDAC / BRCA separately, never pooled). CPU-only.

Completes the pipeline: extraction -> validation -> RDF materialization -> SPARQL.
Produces graph-quality/scale stats (paper Table XIII) and cohort-retrieval results
(Table XV) for each cohort.

    python scripts/run_coral_graph.py
"""
from __future__ import annotations
import json, logging, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
logging.basicConfig(level=logging.WARNING, format="%(message)s")
from src.graph.rdf_builder import build_patient_graph, build_cohort_graph, serialize_graph, run_sparql_queries
from src.graph.schema import build_schema_graph, write_schema

# ── trust admission: materialize ONLY triples the gate admits (trust >= delta AND source-grounded),
#    as the draft specifies — the KG is trust-admitted, not the raw union pass-through ──
import functools
import src.extraction.evaluate as _ev
import src.extraction.validation as _val
_ev._normalize = functools.lru_cache(maxsize=None)(_ev._normalize)   # memoize the hot path (contradiction is O(n^2))
_val._normalize = functools.lru_cache(maxsize=None)(_val._normalize)
from src.extraction.validation import validate_patient_triples

DELTA = 0.4
SRC = Path("results/extraction/coral_final/union")   # current Gemma-4 sub-5B ensemble (paper Table II)
OUT = Path("results/rdf"); OUT.mkdir(parents=True, exist_ok=True)
NOTE_DIR = {"pdac": "data/coral/pdac", "brca": "data/coral/breastca"}


def note_text(pid: str) -> str:
    coh, n = pid.split("_"); p = Path(NOTE_DIR[coh]) / f"{n}.txt"
    return p.read_text(errors="ignore") if p.exists() else ""


def admit(pid: str, triples: list[dict]) -> tuple[list[dict], int]:
    """Validate raw triples, keep only those the trust gate admits (trust >= DELTA AND grounded)."""
    cand = [t for t in triples if isinstance(t, dict)]
    vr = validate_patient_triples(cand, note_text(pid), trust_threshold=0.0)
    validated = vr["accepted"] + vr["rejected"]
    kept = [t for t in validated
            if t["_validation"]["trust_score"] >= DELTA and t["_validation"]["source_grounding"] > 0]
    return kept, len(validated)

# cohort-retrieval queries (Table XV) on top of the module defaults
COHORT_Q = {
    "cancer_cohort": """
        SELECT (COUNT(DISTINCT ?patient) AS ?n) WHERE {
            ?patient trustkg:hasEntity ?e . ?e a fhir:Condition . ?e rdfs:label ?l .
            FILTER(REGEX(?l, "cancer|carcinoma|adenocarc|malignan|neoplas|tumou?r", "i"))
        }""",
    "chemo_cohort": """
        SELECT (COUNT(DISTINCT ?patient) AS ?n) WHERE {
            ?patient trustkg:hasEntity ?e . ?e a fhir:MedicationStatement . ?e rdfs:label ?l .
            FILTER(REGEX(?l, "chemo|gemcitabine|abraxane|cisplatin|taxol|cyclophosphamide|doxorubicin|fluorouracil|carboplatin", "i"))
        }""",
    "patients_with_temporal": """
        SELECT (COUNT(DISTINCT ?patient) AS ?n) WHERE {
            ?patient trustkg:hasEntity ?e . ?e trustkg:temporalAnchor ?t .
        }""",
}


def count(q):  # COUNT(...) query -> int
    return int(q[0].get("n", q[0].get("count", 0))) if q else 0


write_schema(str(OUT / "schema.ttl"))   # standalone shared TBox
report = {}
for cohort in ["pdac", "brca"]:
    files = sorted(SRC.glob(f"{cohort}_*.json"))
    pgs = {}
    n_cand = n_adm = 0
    for f in files:
        d = json.load(open(f))
        pid = d.get("patient") or f.stem
        admitted, n = admit(pid, d.get("triples", []))
        n_cand += n; n_adm += len(admitted)
        # already gated; pass trust_threshold=0 so build_patient_graph does not double-filter
        pgs[pid] = build_patient_graph(pid, admitted, trust_threshold=0.0)
    cg = build_cohort_graph(pgs)
    cg += build_schema_graph()   # schema + instances: merge the TBox so each .ttl is self-contained
    serialize_graph(cg, OUT / f"coral_{cohort}.ttl")
    q = run_sparql_queries(cg)
    q.update(run_sparql_queries(cg, COHORT_Q))
    report[cohort] = {
        "patients": len(pgs),
        "candidate_triples": n_cand,
        "admitted_triples": n_adm,
        "admission_rate": round(n_adm / max(n_cand, 1), 4),
        "rdf_triples": len(cg),
        "kg_entities": count(q["entity_count"]),
        "ontology_linked": len(q["ontology_linked_entities"]),
        "conditions": len(q["all_conditions"]),
        "medications": len(q["all_medications"]),
        "procedures": len(q["all_procedures"]),
        "temporal_facts": len(q["temporal_facts"]),
        "high_trust(>=0.8)": len(q["high_trust_entities"]),
        "cancer_cohort_patients": f"{count(q['cancer_cohort'])}/{len(pgs)}",
        "chemo_cohort_patients": f"{count(q['chemo_cohort'])}/{len(pgs)}",
        "patients_with_temporal": f"{count(q['patients_with_temporal'])}/{len(pgs)}",
        "sparql_queries_ok": f"{sum(1 for v in q.values() if v is not None)}/{len(q)}",
    }

json.dump(report, open("results/coral_graph_report.json", "w"), indent=2)
print("=" * 70)
print("CORAL END-TO-END GRAPH STAGE  (RDF materialization + SPARQL, per cohort)")
print("=" * 70)
for cohort, r in report.items():
    print(f"\n### CORAL-{cohort.upper()} ({r['patients']} patients) -> results/rdf/coral_{cohort}.ttl")
    for k, v in r.items():
        if k != "patients":
            print(f"    {k:26s} {v}")
print("\nSaved results/coral_graph_report.json")
