"""Multi-model extractor smoke: run the recovered pipeline (NER -> hybrid retrieval
-> LLM EAV extraction) across several HF LLMs / MoEs on 1-2 CORAL patients, then
score each against the gold .ann.txt. Produces a comparison table so we can pick
which model(s) to scale to the full run.

Run from project root:  python scripts/compare_models_smoke.py --gpu 0
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("compare")

MODELS = ["llama32-3b", "gemma3-4b", "qwen3-8b", "phi-moe"]   # small dense + small MoE
PATIENTS = ["pdac_0", "brca_20"]                               # 1 per cohort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--patients", nargs="*", default=PATIENTS)
    ap.add_argument("--outdir", default="results/extraction/model_compare")
    args = ap.parse_args()

    from src.data.reader import load_coral_documents
    from src.extraction.rag_extractor import RAGExtractor
    from src.extraction.local_llm import unload_all
    from src.extraction.evaluate import evaluate_single_model

    docs = {d.patient_id: d for d in load_coral_documents()}
    targets = [docs[p] for p in args.patients if p in docs]
    out_root = Path(args.outdir)
    tag = out_root.name

    rows = []
    for model in args.models:
        log.info("=" * 60)
        log.info("MODEL: %s", model)
        t0 = time.time()
        try:
            ex = RAGExtractor(output_dir=out_root)
            ex.extract_batch(targets, model_name=model, gpu_id=args.gpu)
            for d in targets:
                ann = Path(d.metadata["file"].replace(".txt", ".ann.txt"))
                ext = out_root / model / f"{d.patient_id}.json"
                m = evaluate_single_model(ext, ann, d.text, d.patient_id)
                rows.append({"model": model, **m})
                log.info("  %s/%s: triples=%s eF1=%.3f eP=%.3f eR=%.3f halluc=%.3f",
                         model, d.patient_id, m["num_extracted"], m["entity_f1"],
                         m["entity_precision"], m["entity_recall"], m["hallucination_rate"])
            unload_all()
            log.info("MODEL %s done in %.0fs", model, time.time() - t0)
        except Exception as e:
            log.exception("MODEL %s FAILED: %s", model, e)
            rows.append({"model": model, "error": str(e)[:200]})

    # ── summary ──
    Path("results").mkdir(exist_ok=True)
    json.dump(rows, open(f"results/model_compare_{tag}.json", "w"), indent=2, default=str)
    print("\n================ EXTRACTOR COMPARISON (smoke) ================")
    hdr = f"{'model':12s} {'patient':9s} {'#trip':>6s} {'entP':>6s} {'entR':>6s} {'entF1':>6s} {'halluc':>6s} {'grnd':>5s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['model']:12s} ERROR: {r['error']}")
        else:
            print(f"{r['model']:12s} {r['patient_id']:9s} {r['num_extracted']:6d} "
                  f"{r['entity_precision']:6.3f} {r['entity_recall']:6.3f} {r['entity_f1']:6.3f} "
                  f"{r['hallucination_rate']:6.3f} {r['source_grounded']:5d}")
    # per-model mean F1
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        if "error" not in r:
            agg[r["model"]].append(r["entity_f1"])
    print("\nMean entity-F1 by model:")
    for mdl, fs in agg.items():
        print(f"  {mdl:12s} {sum(fs)/len(fs):.3f}  (n={len(fs)})")


if __name__ == "__main__":
    main()
