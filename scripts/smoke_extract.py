"""Single-patient smoke test: validate the recovered pipeline end-to-end
(NER -> hybrid retrieval -> Gemma EAV extraction -> triples) and pre-download Gemma.
Run from project root:  python scripts/smoke_extract.py --gpu 0
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("smoke")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="gemma4-4b")
    ap.add_argument("--pid", default="pdac_0")
    args = ap.parse_args()

    from src.data.reader import load_coral_documents
    from src.extraction.rag_extractor import RAGExtractor

    cohort = "pdac" if args.pid.startswith("pdac") else "brca"
    docs = [d for d in load_coral_documents(cohorts=[cohort]) if d.patient_id == args.pid]
    assert docs, f"patient {args.pid} not found"
    log.info("Smoke: extracting %s (%d chars) on GPU %d with %s",
             args.pid, len(docs[0].text), args.gpu, args.model)

    t0 = time.time()
    ex = RAGExtractor(output_dir=Path("results/extraction/smoke"))
    res = ex.extract_batch(docs, model_name=args.model, gpu_id=args.gpu)
    r = res[0]
    log.info("SMOKE OK: %s num_triples=%s in %.1fs", r["patient_id"], r.get("num_triples"), time.time() - t0)
    print("=== SAMPLE TRIPLES ===")
    print(json.dumps(r.get("triples", [])[:5], indent=2, default=str))


if __name__ == "__main__":
    main()
