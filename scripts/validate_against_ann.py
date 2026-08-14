"""Validate each model's extraction QUALITY against the CORAL gold .ann.txt.

Independent of the smoke's printed numbers: re-parses gold annotations + each
model's extracted triples, computes entity-level P/R/F1 + source grounding, and
shows concrete matched / missed-gold / ungrounded examples so quality is inspectable.

Run from project root:  python scripts/validate_against_ann.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")

from src.data.reader import load_coral_documents, load_ground_truth
from src.extraction.evaluate import (
    evaluate_single_model, _get_triple_texts, _match_score, _check_source_grounding,
)

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--outdir", default="results/extraction/model_compare")
_ap.add_argument("--models", nargs="*", default=["llama32-3b", "gemma3-4b", "qwen3-8b", "phi-moe"])
_ap.add_argument("--patients", nargs="*", default=["pdac_0", "brca_20"])
_args = _ap.parse_args()
MODELS = _args.models
PATIENTS = _args.patients
OUT = Path(_args.outdir)

docs = {d.patient_id: d for d in load_coral_documents()}


def gold_unique(ann_path: Path):
    ents = load_ground_truth(ann_path)
    uniq = {}
    for e in ents:
        uniq.setdefault(e["text"].lower().strip(), e)
    return ents, uniq


def matched_gold(triples, gold_uniq):
    hit = set()
    for tr in triples:
        for t in _get_triple_texts(tr):
            for k, g in gold_uniq.items():
                if _match_score(t, g["text"]) >= 0.4:
                    hit.add(k)
    return hit


print("=" * 78)
print("QUALITY VALIDATION vs gold .ann.txt  (entity match threshold 0.4)")
print("=" * 78)

summary = {}
for model in MODELS:
    fscores = []
    print(f"\n########## MODEL: {model} ##########")
    for pid in PATIENTS:
        d = docs[pid]
        ann = Path(d.metadata["file"].replace(".txt", ".ann.txt"))
        ext = OUT / model / f"{pid}.json"
        if not ext.exists():
            print(f"  {pid}: NO OUTPUT ({ext})"); continue
        triples = json.load(open(ext)).get("triples", [])
        ents, guniq = gold_unique(ann)
        m = evaluate_single_model(ext, ann, d.text, pid)
        fscores.append(m["entity_f1"])
        hit = matched_gold(triples, guniq)
        missed = [g["text"] for k, g in guniq.items() if k not in hit]
        ungrounded = [", ".join(_get_triple_texts(tr))[:60] for tr in triples
                      if not any(_check_source_grounding(t, d.text) for t in _get_triple_texts(tr))]
        print(f"  {pid}: gold_uniq={m['num_gt_unique']}  extracted={m['num_extracted']}  "
              f"matched={m['gt_unique_matched']}  P={m['entity_precision']:.3f} "
              f"R={m['entity_recall']:.3f} F1={m['entity_f1']:.3f}  "
              f"halluc={m['hallucination_rate']:.3f} grounded={m['source_grounded']}/{m['num_extracted']}")
        print(f"      matched gold (sample): {sorted(hit)[:6]}")
        print(f"      MISSED gold (sample):  {missed[:6]}")
        if ungrounded:
            print(f"      ungrounded triples:    {ungrounded[:4]}")
    if fscores:
        summary[model] = sum(fscores) / len(fscores)

print("\n" + "=" * 78)
print("MEAN entity-F1 by model (2 patients):")
for m, f in sorted(summary.items(), key=lambda x: -x[1]):
    print(f"  {m:12s} {f:.3f}")
print("=" * 78)
