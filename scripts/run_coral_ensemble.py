"""Full-CORAL ensemble run: for each patient, run each model's 2-pass extraction
(pass-2 seeded with pass-1 as graph-neighborhood evidence), union across models,
attach trust scores, apply the trust filter (delta=0.4), and score vs gold.

Resumable. NER is computed once per patient and reused across models.

    python scripts/run_coral_ensemble.py --gpu 0 --models gemma3-4b llama32-3b
    python scripts/run_coral_ensemble.py --gpu 0 --models gemma3-4b qwen3-8b llama32-3b
    python scripts/run_coral_ensemble.py --gpu 1 --patients brca_20 brca_21 ...   # split across GPUs
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("coral_ens")


def dedup(triples):
    seen, out = set(), []
    for t in triples:
        k = (str(t.get("entity","")).lower().strip(), str(t.get("attribute","")).lower().strip(),
             str(t.get("value","")).lower().strip())
        if k not in seen:
            seen.add(k); out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=["gemma3-4b", "llama32-3b"])
    ap.add_argument("--twopass", nargs="*", default=["gemma3-4b"],
                    help="models that get 2-pass (anchor); others are single-pass augmenters")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--tag", default=None, help="output subdir; default from models")
    args = ap.parse_args()
    tag = args.tag or "ens_" + "_".join(m.split("-")[0] for m in args.models)
    twopass_set = set(args.twopass)

    from src.data.reader import load_coral_documents
    from src.extraction.ner import extract_entities_batch
    from src.extraction.rag_extractor import _extract_document_rag
    from src.extraction.fhir_normalizer import normalize_patient_triples
    from src.extraction.validation import validate_patient_triples, validate_triple
    from src.extraction.evaluate import evaluate_single_model

    docs = {d.patient_id: d for d in load_coral_documents()}
    pids = args.patients or sorted(docs.keys())
    out = Path("results/extraction") / tag
    (out / "union").mkdir(parents=True, exist_ok=True)
    (out / "filtered").mkdir(parents=True, exist_ok=True)
    mpath = Path(f"results/{tag}_metrics_gpu{args.gpu}.json")  # per-GPU file avoids races when split
    metrics = json.load(open(mpath)) if mpath.exists() else []
    done = {m["patient"] for m in metrics}
    log.info("Ensemble models=%s | patients=%d | tag=%s", args.models, len(pids), tag)

    def two_pass(doc, model, mentions):
        p1 = _extract_document_rag(doc, model, args.gpu, mentions)["triples"]
        seed = [dict(t) for t in p1]; normalize_patient_triples(seed)
        p2 = _extract_document_rag(doc, model, args.gpu, mentions, seed_triples=seed)["triples"]
        return dedup(p1 + p2)

    for pid in pids:
        if pid in done:
            log.info("skip %s", pid); continue
        d = docs[pid]; t0 = time.time()
        log.info("=" * 60); log.info("PATIENT %s [gpu %d]", pid, args.gpu)
        mentions = extract_entities_batch([d.text])[0]
        per_model = {}
        pooled = []
        for model in args.models:
            if model in twopass_set:
                mt = two_pass(d, model, mentions)
            else:  # single-pass augmenter
                mt = dedup(_extract_document_rag(d, model, args.gpu, mentions)["triples"])
            per_model[model] = len(mt); pooled += mt
            log.info("  %s (%s): %d triples", model, "2pass" if model in twopass_set else "1pass", len(mt))
        ensemble = dedup(pooled)
        for t in ensemble:
            validate_triple(t, d.text, ensemble)
        filtered = validate_patient_triples(ensemble, d.text, trust_threshold=0.4)["accepted"]
        json.dump({"patient": pid, "triples": ensemble}, open(out / "union" / f"{pid}.json", "w"), indent=2, default=str)
        json.dump({"patient": pid, "triples": filtered}, open(out / "filtered" / f"{pid}.json", "w"), indent=2, default=str)

        ann = Path(d.metadata["file"].replace(".txt", ".ann.txt"))
        row = {"patient": pid, "cohort": d.cohort, "runtime_s": round(time.time()-t0,1),
               "per_model": per_model, "n_ensemble": len(ensemble), "n_filtered": len(filtered)}
        for stage in ["union", "filtered"]:
            m = evaluate_single_model(out / stage / f"{pid}.json", ann, d.text, pid)
            row[f"{stage}_P"], row[f"{stage}_R"], row[f"{stage}_F1"] = m["entity_precision"], m["entity_recall"], m["entity_f1"]
        metrics.append(row); json.dump(metrics, open(mpath, "w"), indent=2)
        log.info("DONE %s: ensemble F1=%.3f R=%.3f P=%.3f (%.0fs) [%d/%d]",
                 pid, row["union_F1"], row["union_R"], row["union_P"], row["runtime_s"], len(metrics), len(pids))

    import statistics as st
    for coh in ["brca", "pdac"]:
        f1 = [m["union_F1"] for m in metrics if m.get("cohort") == coh]
        if f1: log.info("%s: n=%d mean_F1=%.3f (sd %.3f)", coh.upper(), len(f1), st.mean(f1), st.pstdev(f1))
    log.info("Saved %s", mpath)


if __name__ == "__main__":
    main()
