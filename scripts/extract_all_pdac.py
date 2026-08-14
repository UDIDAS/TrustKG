"""Extract PDAC patients: train first (no seed), then test with train-KG seed.

Step 1: Extract PDAC train patients (0-11) with standard RAG pipeline
Step 2: Build PDAC train-KG from step 1 results
Step 3: Extract PDAC test patients (16-19) seeded with PDAC train-KG

Usage:
  python scripts/extract_all_pdac.py --gpu 0
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
    args = parser.parse_args()

    from src.config_splits import CORAL_TRAIN, CORAL_TEST
    from src.data.reader import load_coral_documents
    from src.extraction.fhir_normalizer import normalize_patient_triples
    from src.extraction.rag_extractor import RAGExtractor

    out_dir = Path("results/extraction/full_rag")
    model_dir = out_dir / args.model

    pdac_train = [p for p in CORAL_TRAIN if p.startswith("pdac")]
    pdac_test = [p for p in CORAL_TEST if p.startswith("pdac")]

    all_docs = load_coral_documents(cohorts=["pdac"])
    doc_map = {d.patient_id: d for d in all_docs}

    # ── Step 1: Extract PDAC train (no seed) ──
    train_docs = [doc_map[p] for p in pdac_train if p in doc_map]
    already = [p for p in pdac_train if (model_dir / f"{p}.json").exists()]
    need = [doc_map[p] for p in pdac_train if p in doc_map and not (model_dir / f"{p}.json").exists()]

    if need:
        logger.info("Step 1: Extracting %d PDAC train patients (no seed)...", len(need))
        extractor = RAGExtractor(output_dir=out_dir)
        extractor.extract_batch(need, model_name=args.model, gpu_id=args.gpu)
    else:
        logger.info("Step 1: All %d PDAC train patients already extracted", len(already))

    # ── Step 2: Build PDAC train-KG ──
    logger.info("Step 2: Building PDAC train-KG...")
    seed_triples = []
    for pid in pdac_train:
        f = model_dir / f"{pid}.json"
        if f.exists():
            with open(f) as fh:
                triples = json.load(fh).get("triples", [])
            normalize_patient_triples(triples)
            seed_triples.extend(triples)
    logger.info("PDAC train-KG: %d triples from %d patients",
                len(seed_triples),
                sum(1 for p in pdac_train if (model_dir / f"{p}.json").exists()))

    # ── Step 3: Extract PDAC test with train-KG seed ──
    test_docs = [doc_map[p] for p in pdac_test if p in doc_map]
    if test_docs:
        logger.info("Step 3: Extracting %d PDAC test patients with %d seed triples...",
                     len(test_docs), len(seed_triples))
        extractor = RAGExtractor(output_dir=out_dir)
        results = extractor.extract_batch(
            test_docs,
            model_name=args.model,
            gpu_id=args.gpu,
            seed_triples=seed_triples,
        )
        for r in results:
            logger.info("  %s: %d triples", r["patient_id"], r["num_triples"])
    else:
        logger.info("Step 3: No PDAC test documents found")

    logger.info("Done.")


if __name__ == "__main__":
    main()
