"""Gemma-3-4B recall push: 2-pass extraction (pass-2 seeded with pass-1 triples as
graph-neighborhood evidence) -> union -> trust-aware filter (delta=0.4).
Measures recall/precision/F1 at each stage vs gold. Run from project root:
    python scripts/run_gemma_2pass.py --gpu 0
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("2pass")

MODEL = "gemma3-4b"
PATIENTS = ["pdac_0", "brca_20"]


def dedup(triples):
    seen, out = set(), []
    for t in triples:
        k = (str(t.get("entity","")).lower().strip(),
             str(t.get("attribute","")).lower().strip(),
             str(t.get("value","")).lower().strip())
        if k in seen:
            continue
        seen.add(k); out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    from src.data.reader import load_coral_documents
    from src.extraction.ner import extract_entities_batch
    from src.extraction.rag_extractor import _extract_document_rag
    from src.extraction.fhir_normalizer import normalize_patient_triples
    from src.extraction.validation import validate_patient_triples
    from src.extraction.evaluate import evaluate_single_model

    docs = {d.patient_id: d for d in load_coral_documents()}
    outroot = Path("results/extraction/gemma_2pass")
    for stage in ["pass1", "union", "filtered"]:
        (outroot / stage).mkdir(parents=True, exist_ok=True)

    def save(stage, pid, triples):
        json.dump({"triples": triples, "num_triples": len(triples)},
                  open(outroot / stage / f"{pid}.json", "w"), indent=2, default=str)

    metrics = []
    for pid in PATIENTS:
        d = docs[pid]
        log.info("=" * 60); log.info("PATIENT %s (%d chars)", pid, len(d.text))
        mentions = extract_entities_batch([d.text])[0]
        log.info("NER: %d mentions", len(mentions))

        # ── Pass 1 ──
        t0 = time.time()
        r1 = _extract_document_rag(d, MODEL, args.gpu, mentions)
        p1 = r1["triples"]
        log.info("PASS1: %d triples (%.0fs)", len(p1), time.time() - t0)

        # ── Pass 2 (seed with pass-1 as graph-neighborhood evidence) ──
        seed = [dict(t) for t in p1]
        normalize_patient_triples(seed)
        t0 = time.time()
        r2 = _extract_document_rag(d, MODEL, args.gpu, mentions, seed_triples=seed)
        p2 = r2["triples"]
        log.info("PASS2: %d new triples (%.0fs)", len(p2), time.time() - t0)

        union = dedup(p1 + p2)
        log.info("UNION: %d triples (pass1=%d + pass2=%d, deduped)", len(union), len(p1), len(p2))

        # ── Trust-aware filter (rule-based layers, delta=0.4) ──
        vr = validate_patient_triples(union, d.text, trust_threshold=0.4)
        filtered = vr["accepted"]
        log.info("FILTER: %d/%d accepted (rate %.2f, mean_trust %.3f)",
                 len(filtered), len(union), vr["stats"]["acceptance_rate"], vr["stats"]["mean_trust"])

        save("pass1", pid, p1); save("union", pid, union); save("filtered", pid, filtered)

        # ── Evaluate each stage vs gold ──
        ann = Path(d.metadata["file"].replace(".txt", ".ann.txt"))
        for stage in ["pass1", "union", "filtered"]:
            m = evaluate_single_model(outroot / stage / f"{pid}.json", ann, d.text, pid)
            metrics.append({"patient": pid, "stage": stage, "n": m["num_extracted"],
                            "P": m["entity_precision"], "R": m["entity_recall"], "F1": m["entity_f1"],
                            "halluc": m["hallucination_rate"]})
            log.info("  [%s] %s: n=%d P=%.3f R=%.3f F1=%.3f", stage, pid,
                     m["num_extracted"], m["entity_precision"], m["entity_recall"], m["entity_f1"])

    json.dump(metrics, open("results/gemma_2pass_metrics.json", "w"), indent=2)
    print("\n================ GEMMA-3-4B: single-pass -> 2-pass -> filtered ================")
    print(f"{'patient':8s} {'stage':9s} {'n':>4s} {'P':>6s} {'R':>6s} {'F1':>6s} {'halluc':>6s}")
    for r in metrics:
        print(f"{r['patient']:8s} {r['stage']:9s} {r['n']:4d} {r['P']:6.3f} {r['R']:6.3f} {r['F1']:6.3f} {r['halluc']:6.3f}")


if __name__ == "__main__":
    main()
