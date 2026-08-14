"""Full-CORAL Gemma-3-4B 2-pass run (all 40 patients): pass-2 seeded with pass-1
triples as graph-neighborhood evidence -> union -> trust filter (delta=0.4).
Saves per-patient triples + trust scores and aggregate metrics. Resumable.

Feeds paper Tables II (entity extraction) and XII (patient-level robustness), and
produces the trust-scored, gold-labeled triples needed for the calibration/selective
analysis (Tables VIII/IX/XI).

    python scripts/run_coral_full.py --gpu 0                 # all 40 patients
    python scripts/run_coral_full.py --gpu 1 --patients brca_20 brca_21 ...
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("coral_full")
MODEL = "gemma3-4b"


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
    ap.add_argument("--patients", nargs="*", default=None, help="default: all 40")
    args = ap.parse_args()

    from src.data.reader import load_coral_documents
    from src.extraction.ner import extract_entities_batch
    from src.extraction.rag_extractor import _extract_document_rag
    from src.extraction.fhir_normalizer import normalize_patient_triples
    from src.extraction.validation import validate_patient_triples, validate_triple
    from src.extraction.evaluate import evaluate_single_model

    docs = {d.patient_id: d for d in load_coral_documents()}
    pids = args.patients or sorted(docs.keys())
    out = Path("results/extraction/coral_full")
    for s in ["union", "filtered"]:
        (out / s).mkdir(parents=True, exist_ok=True)
    metrics_path = Path("results/coral_full_metrics.json")
    metrics = json.load(open(metrics_path)) if metrics_path.exists() else []
    done = {m["patient"] for m in metrics}

    for pid in pids:
        if pid in done:
            log.info("skip %s (already done)", pid); continue
        d = docs[pid]
        log.info("=" * 60); log.info("PATIENT %s (%d chars) [gpu %d]", pid, len(d.text), args.gpu)
        t0 = time.time()
        mentions = extract_entities_batch([d.text])[0]
        p1 = _extract_document_rag(d, MODEL, args.gpu, mentions)["triples"]
        seed = [dict(t) for t in p1]; normalize_patient_triples(seed)
        p2 = _extract_document_rag(d, MODEL, args.gpu, mentions, seed_triples=seed)["triples"]
        union = dedup(p1 + p2)
        # attach trust score to every union triple (for calibration/selective tables)
        for t in union:
            validate_triple(t, d.text, union)
        vr = validate_patient_triples(union, d.text, trust_threshold=0.4)
        filtered = vr["accepted"]
        json.dump({"patient": pid, "triples": union, "num_triples": len(union)},
                  open(out / "union" / f"{pid}.json", "w"), indent=2, default=str)
        json.dump({"patient": pid, "triples": filtered, "num_triples": len(filtered)},
                  open(out / "filtered" / f"{pid}.json", "w"), indent=2, default=str)

        ann = Path(d.metadata["file"].replace(".txt", ".ann.txt"))
        row = {"patient": pid, "cohort": d.cohort, "runtime_s": round(time.time() - t0, 1),
               "n_pass1": len(p1), "n_union": len(union), "n_filtered": len(filtered),
               "mean_trust": vr["stats"]["mean_trust"]}
        for stage, root in [("union", out / "union"), ("filtered", out / "filtered")]:
            m = evaluate_single_model(root / f"{pid}.json", ann, d.text, pid)
            row[f"{stage}_P"] = m["entity_precision"]; row[f"{stage}_R"] = m["entity_recall"]
            row[f"{stage}_F1"] = m["entity_f1"]; row[f"{stage}_halluc"] = m["hallucination_rate"]
        metrics.append(row)
        json.dump(metrics, open(metrics_path, "w"), indent=2)
        log.info("DONE %s: union F1=%.3f R=%.3f (%.0fs)  [%d/%d]",
                 pid, row["union_F1"], row["union_R"], row["runtime_s"], len(metrics), len(pids))

    # aggregate
    import statistics as st
    for coh in ["brca", "pdac"]:
        f1 = [m["union_F1"] for m in metrics if m.get("cohort") == coh]
        if f1:
            log.info("%s: n=%d mean_union_F1=%.3f (sd %.3f)", coh.upper(), len(f1),
                     st.mean(f1), st.pstdev(f1))
    log.info("Saved %s (%d patients)", metrics_path, len(metrics))


if __name__ == "__main__":
    main()
