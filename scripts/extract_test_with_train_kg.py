"""Extract TEST patients using TRAIN-KG as seed for graph neighborhood retrieval.

This implements the split-aware evaluation:
  - Load TRAIN patient triples as seed KG
  - Extract TEST patients with seed KG context in retrieval
  - The graph neighborhood retrieval (Draft §3.3) uses train triples
    to provide prior evidence during TEST extraction

Usage:
  python scripts/extract_test_with_train_kg.py --gpu 0
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HF_TOKEN", "")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default="gemma4-4b")
    parser.add_argument("--split", default="test", choices=["test", "val", "all_missing"])
    args = parser.parse_args()

    from src.config_splits import CORAL_TRAIN, CORAL_VAL, CORAL_TEST
    from src.data.reader import load_coral_documents
    from src.extraction.fhir_normalizer import normalize_patient_triples
    from src.extraction.rag_extractor import RAGExtractor

    # ── 1. Load TRAIN triples as seed KG ──
    logger.info("Loading TRAIN-KG seed triples...")
    seed_triples = []
    train_dir = Path("results/extraction/full_rag/gemma4-4b")
    for pid in CORAL_TRAIN:
        f = train_dir / f"{pid}.json"
        if f.exists():
            with open(f) as fh:
                triples = json.load(fh).get("triples", [])
            normalize_patient_triples(triples)
            seed_triples.extend(triples)
    logger.info("Loaded %d seed triples from %d TRAIN patients",
                len(seed_triples), len([p for p in CORAL_TRAIN if (train_dir / f"{p}.json").exists()]))

    # ── 2. Determine which patients to extract ──
    if args.split == "test":
        target_pids = list(CORAL_TEST)
    elif args.split == "val":
        target_pids = list(CORAL_VAL)
    else:
        # All patients missing from extraction dir
        target_pids = []
        for pid in CORAL_VAL + CORAL_TEST:
            if not (train_dir / f"{pid}.json").exists():
                target_pids.append(pid)

    # Check which are already extracted
    already = [p for p in target_pids if (train_dir / f"{p}.json").exists()]
    missing = [p for p in target_pids if not (train_dir / f"{p}.json").exists()]

    logger.info("Target: %d patients (%d already extracted, %d new)",
                len(target_pids), len(already), len(missing))

    if not missing and not already:
        logger.info("Nothing to extract.")
        return

    # For already-extracted patients, re-extract with seed KG
    to_extract = target_pids  # re-extract all with seed KG

    # ── 3. Load documents ──
    all_docs = load_coral_documents(annotated_only=False)
    doc_map = {d.patient_id: d for d in all_docs}

    # For annotated-only loading, also try all docs
    if not all(p in doc_map for p in to_extract):
        all_docs2 = load_coral_documents(annotated_only=True)
        for d in all_docs2:
            if d.patient_id not in doc_map:
                doc_map[d.patient_id] = d

    target_docs = [doc_map[p] for p in to_extract if p in doc_map]
    if not target_docs:
        logger.warning("No documents found for target patients: %s", to_extract)
        return

    logger.info("Extracting %d patients with %d seed triples...",
                len(target_docs), len(seed_triples))

    # ── 4. Extract with seed KG ──
    extractor = RAGExtractor(output_dir=Path("results/extraction/full_rag"))
    results = extractor.extract_batch(
        target_docs,
        model_name=args.model,
        gpu_id=args.gpu,
        seed_triples=seed_triples,
    )

    # ── 5. Summary ──
    for r in results:
        logger.info("  %s: %d triples (seed: %d)",
                     r["patient_id"], r["num_triples"], r.get("seed_triples_used", 0))

    logger.info("Done. Results in results/extraction/full_rag/%s/", args.model)


if __name__ == "__main__":
    main()
