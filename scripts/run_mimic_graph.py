"""MIMIC graph stage: materialize the ensemble union into RDF and run SPARQL cohort
queries, PER COHORT (MIMIC-III / MIMIC-IV separately). CPU-only. Mirror of
run_coral_graph.py; MIMIC has no gold so this reports scale / grounding / cohort-retrieval
(paper Tables VI, XV) — not F1. build_patient_graph normalizes by default (clean KG).

    python scripts/run_mimic_graph.py
"""
from __future__ import annotations
import json, logging, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
logging.basicConfig(level=logging.WARNING, format="%(message)s")
from src.graph.rdf_builder import build_patient_graph, build_cohort_graph, serialize_graph, run_sparql_queries
from src.graph.schema import build_schema_graph, write_schema

DELTA = 0.4
OUT = Path("results/rdf"); OUT.mkdir(parents=True, exist_ok=True)

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


def count(q):
    return int(q[0].get("n", q[0].get("count", 0))) if q else 0


MODELS = ["gemma4-e4b", "llama32-3b", "qwen3-4b", "medgemma-4b"]


def _dedup(ts):
    seen, out = set(), []
    for t in ts:
        if not isinstance(t, dict):
            continue
        k = (str(t.get("entity", "")).lower().strip(),
             str(t.get("attribute", "")).lower().strip(),
             str(t.get("value", "")).lower().strip())
        if k not in seen:
            seen.add(k); out.append(t)
    return out


write_schema(str(OUT / "schema.ttl"))   # standalone shared TBox
report = {}
for cohort in ["mimiciii", "mimiciv"]:
    # ensemble union per note, pooled+deduped from the complete bymodel caches
    # (same raw ensemble union CORAL's KG is built from; independent of the slow validated union)
    bym = Path(f"results/extraction/mimic_{cohort}/bymodel")
    ids = sorted({p.stem for m in MODELS for p in (bym / m).glob("*.json")})
    pgs = {}
    for nid in ids:
        pooled = []
        for m in MODELS:
            f = bym / m / f"{nid}.json"
            if f.exists():
                pooled += json.load(open(f)).get("triples", [])
        pgs[nid] = build_patient_graph(nid, _dedup(pooled), trust_threshold=DELTA)
    cg = build_cohort_graph(pgs)
    cg += build_schema_graph()   # schema + instances: merge the TBox so each .ttl is self-contained
    serialize_graph(cg, OUT / f"mimic_{cohort}.ttl")
    q = run_sparql_queries(cg)
    q.update(run_sparql_queries(cg, COHORT_Q))
    report[cohort] = {
        "patients": len(pgs),
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

json.dump(report, open("results/mimic_graph_report.json", "w"), indent=2)
print("=" * 70)
print("MIMIC END-TO-END GRAPH STAGE  (RDF materialization + SPARQL, per cohort)")
print("=" * 70)
for cohort, r in report.items():
    print(f"\n### MIMIC-{cohort.upper()} ({r['patients']} patients) -> results/rdf/mimic_{cohort}.ttl")
    for k, v in r.items():
        if k != "patients":
            print(f"    {k:26s} {v}")
print("\nSaved results/mimic_graph_report.json")
