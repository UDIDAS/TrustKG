"""Precompute entity-level P/R/F1 (+ grounding) for every model/patient/config we ran,
so the results notebook can load a cache instead of re-running the slow matcher.
Writes results/all_metrics.json.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
from src.data.reader import load_coral_documents
from src.extraction.evaluate import evaluate_single_model

docs = {d.patient_id: d for d in load_coral_documents()}
PATIENTS = ["pdac_0", "brca_20"]
RUNS = {
    "baseline": ("results/extraction/model_compare", ["llama32-3b", "gemma3-4b", "qwen3-8b"]),
    "hirecall": ("results/extraction/model_compare_hirecall", ["gemma3-4b", "qwen3-8b", "olmoe-1b7b"]),
}
rows = []
for run, (root, models) in RUNS.items():
    for model in models:
        for pid in PATIENTS:
            f = Path(root) / model / f"{pid}.json"
            if not f.exists():
                rows.append({"run": run, "model": model, "patient": pid, "status": "no_output"})
                continue
            t0 = time.time()
            m = evaluate_single_model(f, Path(docs[pid].metadata["file"].replace(".txt", ".ann.txt")),
                                      docs[pid].text, pid)
            rows.append({"run": run, "model": model, "patient": pid, "status": "ok",
                         "num_triples": m["num_extracted"], "num_gold_unique": m["num_gt_unique"],
                         "precision": m["entity_precision"], "recall": m["entity_recall"],
                         "f1": m["entity_f1"], "hallucination": m["hallucination_rate"],
                         "grounded": m["source_grounded"]})
            print(f"{run:9s} {model:12s} {pid:8s} P={m['entity_precision']:.3f} "
                  f"R={m['entity_recall']:.3f} F1={m['entity_f1']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(rows, open("results/all_metrics.json", "w"), indent=2)
print("SAVED results/all_metrics.json  (%d rows)" % len(rows))
