"""MIMIC-III Ext-Notes: Full pipeline with proper train/val/test splits.

Uses the MIMIC-III-Ext-Notes benchmark (150 clinician-annotated notes).
Split: 80 train / 20 val / 50 test
Train+Val KG seeds test extraction.
Evaluate against clinician annotations.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HF_TOKEN", "")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

STOPWORDS = {"unspecified", "other", "nos", "nec", "disease", "disorder"}


def normalize(text):
    t = re.sub(r"[^\w\s]", " ", text.lower().strip())
    return " ".join(w for w in t.split() if w not in STOPWORDS and len(w) > 1)


def concept_match(a, b):
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 4 and shorter in longer:
        return True
    ta, tb = set(na.split()), set(nb.split())
    return len(ta & tb) / max(len(ta | tb), 1) > 0.3


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--model", default="gemma4-4b")
    args = parser.parse_args()

    from src.data.reader import ClinicalDocument
    from src.extraction.fhir_normalizer import normalize_patient_triples
    from src.extraction.rag_extractor import RAGExtractor

    # Load benchmark
    zp = "data/mimic/mimic-iii-ext-notes-1.0.0.zip"
    with zipfile.ZipFile(zp) as z:
        with z.open("mimic-iii-ext-notes-1.0.0/labels.csv") as f:
            labels = list(csv.DictReader(io.TextIOWrapper(f)))
        with z.open("mimic-iii-ext-notes-1.0.0/notes.csv") as f:
            notes_data = list(csv.DictReader(io.TextIOWrapper(f)))

    # GT: detectable + encounter-relevant + not negated
    gt_by_note = {}
    for l in labels:
        if l["detection"] == "yes" and l["encounter"] == "yes" and l["negation"] == "no":
            gt_by_note.setdefault(l["row_id"], set()).add(l["concept"].lower())

    # Split: 80 train / 20 val / 50 test
    all_notes = notes_data[:150]
    train_notes = all_notes[:80]
    val_notes = all_notes[80:100]
    test_notes = all_notes[100:150]

    logger.info("Split: %d train / %d val / %d test", len(train_notes), len(val_notes), len(test_notes))

    def to_docs(note_list):
        return [
            ClinicalDocument(
                patient_id=f"extnote_{n['row_id']}",
                cohort="mimic3",
                source="ext_notes",
                text=n["text"][:4000],
                metadata={"row_id": n["row_id"]},
            )
            for n in note_list
            if len(n["text"]) >= 200
        ]

    train_docs = to_docs(train_notes)
    val_docs = to_docs(val_notes)
    test_docs = to_docs(test_notes)

    out_dir = Path("results/extraction/mimic3_ext")
    extractor = RAGExtractor(output_dir=out_dir)

    # Step 1: Extract train (no seed)
    logger.info("=== STEP 1: Extract %d TRAIN notes ===", len(train_docs))
    train_results = extractor.extract_batch(train_docs, model_name=args.model, gpu_id=args.gpu)
    train_kg = []
    for r in train_results:
        normalize_patient_triples(r["triples"])
        train_kg.extend(r["triples"])
    logger.info("Train KG: %d triples", len(train_kg))

    # Step 2: Extract val (with train KG seed)
    logger.info("=== STEP 2: Extract %d VAL notes with %d seed ===", len(val_docs), len(train_kg))
    val_results = extractor.extract_batch(val_docs, model_name=args.model, gpu_id=args.gpu, seed_triples=train_kg)
    for r in val_results:
        normalize_patient_triples(r["triples"])
        train_kg.extend(r["triples"])
    logger.info("Train+Val KG: %d triples", len(train_kg))

    # Step 3: Extract test (with train+val KG seed)
    logger.info("=== STEP 3: Extract %d TEST notes with %d seed ===", len(test_docs), len(train_kg))
    test_results = extractor.extract_batch(test_docs, model_name=args.model, gpu_id=args.gpu, seed_triples=train_kg)

    # Evaluate all splits
    logger.info("=== EVALUATING ===")

    def evaluate_split(results_list, doc_list, split_name):
        metrics = []
        for r, doc in zip(results_list, doc_list):
            rid = doc.metadata["row_id"]
            gt = gt_by_note.get(rid, set())
            if not gt:
                continue

            normalize_patient_triples(r["triples"])
            extracted = set()
            for t in r["triples"]:
                for field in ["entity", "value"]:
                    v = str(t.get(field, "")).lower().strip()
                    if v and len(v) > 3:
                        extracted.add(v)

            matched_gt = sum(1 for g in gt if any(concept_match(e, g) for e in extracted))
            matched_ext = sum(1 for e in extracted if any(concept_match(e, g) for g in gt))

            precision = matched_ext / max(len(extracted), 1)
            recall = matched_gt / max(len(gt), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            # Source grounding
            source_lower = doc.text.lower()
            grounded = sum(
                1
                for e in extracted
                if e in source_lower or any(w in source_lower for w in e.split()[:2] if len(w) > 3)
            )
            grounding = grounded / max(len(extracted), 1)

            metrics.append(
                {"p": precision, "r": recall, "f1": f1, "grounding": grounding, "n_ext": len(extracted), "n_gt": len(gt)}
            )

        if metrics:
            logger.info(
                "%s (n=%d): P=%.3f±%.3f R=%.3f±%.3f F1=%.3f±%.3f Grounding=%.3f",
                split_name,
                len(metrics),
                np.mean([m["p"] for m in metrics]),
                np.std([m["p"] for m in metrics]),
                np.mean([m["r"] for m in metrics]),
                np.std([m["r"] for m in metrics]),
                np.mean([m["f1"] for m in metrics]),
                np.std([m["f1"] for m in metrics]),
                np.mean([m["grounding"] for m in metrics]),
            )
        return metrics

    train_m = evaluate_split(train_results, train_docs, "TRAIN")
    val_m = evaluate_split(val_results, val_docs, "VAL")
    test_m = evaluate_split(test_results, test_docs, "TEST")

    # Save
    summary = {"train": train_m, "val": val_m, "test": test_m, "train_kg_size": len(train_kg)}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "full_eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== MIMIC-III EXT-NOTES SUMMARY ===")
    print(f"Train+Val KG: {len(train_kg)} triples")
    for name, m in [("TRAIN", train_m), ("VAL", val_m), ("TEST", test_m)]:
        if m:
            print(
                f"{name} (n={len(m)}): "
                f"P={np.mean([x['p'] for x in m]):.3f} "
                f"R={np.mean([x['r'] for x in m]):.3f} "
                f"F1={np.mean([x['f1'] for x in m]):.3f} "
                f"Grounding={np.mean([x['grounding'] for x in m]):.3f}"
            )


if __name__ == "__main__":
    main()
