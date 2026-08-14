"""MIMIC oncology extraction — LOCAL ONLY, never touches BigQuery.

Runs the TRUST-KG extraction pipeline on the pre-pulled notes
(data/mimic_oncology/<source>/notes_all.jsonl) and reports source-grounding +
scale statistics. MIMIC has no expert entity gold, so we report grounding/scale
(not P/R/F1) — this feeds the paper's MIMIC scale tables (I, III, VI, XIII, XIV).

Resumable (per-note JSON + running metrics). GPU-selectable.

    python scripts/run_mimic_extraction.py --source mimiciii --gpu 0
    python scripts/run_mimic_extraction.py --source mimiciv  --gpu 1 --twopass
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mimic_ex")


def dedup(ts):
    seen, out = set(), []
    for t in ts:
        k = (str(t.get("entity","")).lower().strip(), str(t.get("attribute","")).lower().strip(),
             str(t.get("value","")).lower().strip())
        if k not in seen:
            seen.add(k); out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mimiciii", "mimiciv"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="gemma3-4b")
    ap.add_argument("--twopass", action="store_true", help="2-pass (default 1-pass for scale feasibility)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from src.data.reader import ClinicalDocument
    from src.extraction.ner import extract_entities_batch
    from src.extraction.rag_extractor import _extract_document_rag
    from src.extraction.fhir_normalizer import normalize_patient_triples
    from src.extraction.validation import validate_patient_triples, validate_triple

    fp = Path(f"data/mimic_oncology/{args.source}/notes_all.jsonl")
    if not fp.exists():
        sys.exit(f"missing {fp} — run scripts/fetch_mimic_oncology.py first")
    notes = [json.loads(l) for l in open(fp)]
    if args.limit:
        notes = notes[:args.limit]

    out = Path(f"results/extraction/mimic_{args.source}/triples"); out.mkdir(parents=True, exist_ok=True)
    mpath = Path(f"results/mimic_{args.source}_metrics.json")
    metrics = json.load(open(mpath)) if mpath.exists() else []
    done = {m["note_id"] for m in metrics}
    log.info("MIMIC %s: %d notes | model=%s pass=%s gpu=%d | %d already done",
             args.source, len(notes), args.model, "2" if args.twopass else "1", args.gpu, len(done))

    for i, r in enumerate(notes):
        nid = str(r.get("note_id"))
        if nid in done:
            continue
        text = r.get("text") or ""
        if len(text) < 200:
            continue
        doc = ClinicalDocument(patient_id=f"{args.source}_{nid}", cohort=args.source,
                               source="mimic", text=text, metadata={"subject_id": r.get("subject_id")})
        t0 = time.time()
        mentions = extract_entities_batch([text])[0]
        p1 = _extract_document_rag(doc, args.model, args.gpu, mentions)["triples"]
        if args.twopass:
            seed = [dict(t) for t in p1]; normalize_patient_triples(seed)
            p2 = _extract_document_rag(doc, args.model, args.gpu, mentions, seed_triples=seed)["triples"]
            tri = dedup(p1 + p2)
        else:
            tri = p1
        for t in tri:
            validate_triple(t, text, tri)
        vr = validate_patient_triples(tri, text, trust_threshold=0.4)
        grounded = sum(1 for t in tri if t.get("_validation", {}).get("source_grounding", 0) >= 0.5)
        accepted = vr["accepted"]
        json.dump({"note_id": nid, "subject_id": r.get("subject_id"), "triples": tri},
                  open(out / f"{nid}.json", "w"), indent=2, default=str)
        metrics.append({
            "note_id": nid, "subject_id": r.get("subject_id"), "runtime_s": round(time.time()-t0, 1),
            "n_triples": len(tri), "n_grounded": grounded,
            "grounding_rate": round(grounded / max(len(tri), 1), 3),
            "n_accepted": len(accepted), "mean_trust": vr["stats"]["mean_trust"],
        })
        json.dump(metrics, open(mpath, "w"), indent=2)
        if len(metrics) % 10 == 0 or i < 3:
            log.info("  [%d/%d] %s: %d triples, grounded=%.2f (%.0fs)",
                     len(metrics), len(notes), nid, len(tri), grounded/max(len(tri),1), time.time()-t0)

    # ── aggregate scale + grounding ──
    ents = set()
    for f in out.glob("*.json"):
        for t in json.load(open(f)).get("triples", []):
            e = str(t.get("entity", "")).lower().strip()
            if e:
                ents.add(e)
    tot_tri = sum(m["n_triples"] for m in metrics)
    tot_gr = sum(m["n_grounded"] for m in metrics)
    log.info("=" * 60)
    log.info("MIMIC %s DONE: notes=%d  triples=%d  unique_entities=%d  grounding_rate=%.3f  mean_trust=%.3f",
             args.source, len(metrics), tot_tri, len(ents), tot_gr / max(tot_tri, 1),
             sum(m["mean_trust"] for m in metrics) / max(len(metrics), 1))
    log.info("Saved %s", mpath)


if __name__ == "__main__":
    main()
