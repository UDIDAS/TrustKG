#!/usr/bin/env python3
"""Run TRUST-KG pipeline on MIMIC-IV-Note discharge summaries.

Processes in batches of 50 notes, pushes results to GDrive after each batch.
Compares LLM-extracted KG against structured MIMIC data (ground truth).

Usage:
    python scripts/run_mimic_notes_pipeline.py --subset oncology --max-notes 200
    python scripts/run_mimic_notes_pipeline.py --subset temporal --max-notes 500
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["HF_TOKEN"] = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="TRUST-KG on MIMIC Notes")
    parser.add_argument("--subset", choices=["oncology", "icu", "temporal"], default="oncology")
    parser.add_argument("--max-notes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--model", default="gemma4-4b")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    from src.data.mimic_notes_reader import load_mimic_notes_subset, check_notes_available

    if not check_notes_available():
        logger.error("MIMIC-IV-Note zip not found. Waiting for download...")
        sys.exit(1)

    output_dir = Path(f"results/mimic_notes/{args.subset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load notes
    logger.info("Loading MIMIC %s notes (max %d)...", args.subset, args.max_notes)
    docs = load_mimic_notes_subset(
        subset=args.subset,
        max_notes=args.max_notes,
    )
    logger.info("Loaded %d notes", len(docs))

    if not docs:
        logger.error("No notes loaded")
        sys.exit(1)

    # Process in batches
    from src.extraction.rag_extractor import RAGExtractor

    extractor = RAGExtractor(output_dir=output_dir / "extraction")
    total_triples = 0

    for batch_start in range(0, len(docs), args.batch_size):
        batch = docs[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(docs) + args.batch_size - 1) // args.batch_size

        logger.info("=== Batch %d/%d: %d notes ===", batch_num, total_batches, len(batch))

        t0 = time.time()
        results = extractor.extract_batch(batch, model_name=args.model, gpu_id=args.gpu)
        elapsed = time.time() - t0

        batch_triples = sum(r["num_triples"] for r in results)
        total_triples += batch_triples
        logger.info(
            "Batch %d: %d triples in %.1fs (%.1fs/note)",
            batch_num, batch_triples, elapsed, elapsed / max(len(batch), 1),
        )

        # Push batch to GDrive
        batch_dir = output_dir / "extraction" / args.model
        try:
            subprocess.run(
                ["rclone", "copy", str(batch_dir),
                 f"drive_UD:CIKM26_results/mimic_notes/{args.subset}/{args.model}/"],
                capture_output=True, timeout=120,
            )
            logger.info("Batch %d pushed to GDrive", batch_num)
        except Exception as e:
            logger.warning("GDrive push failed: %s", e)

        # Check quota
        try:
            result = subprocess.run(
                ["quota", "-s"], capture_output=True, text=True, timeout=5,
            )
            logger.info("Quota: %s", result.stdout.strip().split("\n")[-1].strip())
        except Exception:
            pass

    logger.info("=== MIMIC NOTES PIPELINE COMPLETE ===")
    logger.info("Total: %d notes, %d triples", len(docs), total_triples)

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "subset": args.subset,
            "total_notes": len(docs),
            "total_triples": total_triples,
            "model": args.model,
            "avg_triples_per_note": round(total_triples / max(len(docs), 1), 1),
        }, f, indent=2)


if __name__ == "__main__":
    main()
