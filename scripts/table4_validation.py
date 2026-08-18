"""Table IV — Construction-Time Validation on CORAL.

Aggregates the 5 validation-layer scores over the FULL current CORAL run
(results/extraction/coral_final/union, Gemma-4 sub-5B ensemble, 40 patients):
    source grounding | ontology compatibility | schema validity |
    temporal consistency | contradiction control
Recomputes with the live validator (union files predate score-storing), so the
numbers are reproducible from the current run — not the old 30-patient values.

    python scripts/table4_validation.py
"""
from __future__ import annotations
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.extraction.validation import validate_triple

NOTE_DIR = {"pdac": "data/coral/pdac", "brca": "data/coral/breastca"}
LAYERS = ["source_grounding", "ontology_check", "schema_check",
          "temporal_consistency", "contradiction_score"]
LABEL = {
    "source_grounding": "Source grounding",
    "ontology_check": "Ontology compatibility",
    "schema_check": "Schema validity",
    "temporal_consistency": "Temporal consistency",
    "contradiction_score": "Contradiction control",
}


def note_text(pid: str) -> str:
    coh, n = pid.split("_")
    p = Path(NOTE_DIR[coh]) / f"{n}.txt"
    return p.read_text(errors="ignore") if p.exists() else ""


def main():
    per_cohort = {"pdac": defaultdict(list), "brca": defaultdict(list)}
    overall = defaultdict(list)
    n_triples = {"pdac": 0, "brca": 0}
    for f in sorted(glob.glob("results/extraction/coral_final/union/*.json")):
        d = json.load(open(f))
        pid = str(d.get("patient") or Path(f).stem)
        coh = pid.split("_")[0]
        text = note_text(pid)
        triples = [t for t in d.get("triples", []) if isinstance(t, dict)]
        n_triples[coh] += len(triples)
        for t in triples:
            v = validate_triple(t, text, triples)["_validation"]
            for L in LAYERS:
                per_cohort[coh][L].append(v[L])
                overall[L].append(v[L])

    def agg(d):
        return {L: round(st.mean(d[L]), 4) if d[L] else None for L in LAYERS}

    report = {
        "extractor": "Gemma-4-E4B 2-pass anchor + Llama-3.2-3B / Qwen3-4B / MedGemma-4B",
        "source": "results/extraction/coral_final/union (full CORAL, 40 patients)",
        "n_triples": {**n_triples, "total": sum(n_triples.values())},
        "pdac": agg(per_cohort["pdac"]),
        "brca": agg(per_cohort["brca"]),
        "overall": agg(overall),
    }
    json.dump(report, open("results/table4_coral_validation.json", "w"), indent=2)

    print("=" * 68)
    print("TABLE IV — Construction-Time Validation on CORAL (full run, 40 patients)")
    print("=" * 68)
    print(f"{'Layer':26s} {'PDAC':>8s} {'BRCA':>8s} {'Overall':>8s}")
    o, p, b = report["overall"], report["pdac"], report["brca"]
    for L in LAYERS:
        print(f"{LABEL[L]:26s} {p[L]:>8.3f} {b[L]:>8.3f} {o[L]:>8.3f}")
    print(f"\nn_triples: PDAC {n_triples['pdac']}  BRCA {n_triples['brca']}  "
          f"total {sum(n_triples.values())}")
    # Pass-rate view — to contrast with the mean scores (these ARE NOT pass rates)
    print("\nMean score vs pass-rate (overall) — the mean is not any single pass rate:")
    print(f"{'Layer':26s} {'mean':>7s} {'%>=0.5':>8s} {'%=1.0':>8s}")
    pr = {}
    for L in LAYERS:
        v = overall[L]
        ge = sum(1 for x in v if x >= 0.5) / len(v)
        eq = sum(1 for x in v if x >= 0.999) / len(v)
        pr[L] = {"mean": round(st.mean(v), 4), "pass_ge_0.5": round(ge, 4), "pass_eq_1.0": round(eq, 4)}
        print(f"{LABEL[L]:26s} {st.mean(v):>7.3f} {ge*100:>7.1f}% {eq*100:>7.1f}%")
    report["overall_passrates"] = pr
    json.dump(report, open("results/table4_coral_validation.json", "w"), indent=2)
    print("Saved results/table4_coral_validation.json")


if __name__ == "__main__":
    main()
