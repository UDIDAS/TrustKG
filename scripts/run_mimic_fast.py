"""Throughput-optimized MIMIC oncology extraction — LOCAL only, never BigQuery.

Speedups vs run_mimic_extraction.py:
  - single RESIDENT model (loaded once, not re-initialized per note),
  - BATCHED chunk generation (many chunks per GPU forward pass, not one-at-a-time),
  - independent chunks (no cross-chunk graph-neighborhood) -> fully batchable.
MIMIC has no expert gold, so single-pass Gemma is sufficient for the scale/grounding
tables; this run also logs THROUGHPUT (notes/hour, chunks/s) -> Table XIV evidence.

Resumable in note-batches. GPU-selectable.

    python scripts/run_mimic_fast.py --source mimiciii --gpu 0 --note-batch 16 --gen-batch 6
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mimic_fast")


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
    ap.add_argument("--note-batch", type=int, default=16, help="notes per resumable batch")
    ap.add_argument("--gen-batch", type=int, default=6, help="chunk-prompts per GPU forward pass")
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from src.extraction.ner import extract_entities_batch
    from src.extraction.rag_extractor import _chunk_with_candidates, _retrieve_ontology_context
    from src.extraction.local_llm import generate_batch, _parse_json_response
    from src.extraction.prompts import RAG_SYSTEM_PROMPT, RAG_EXTRACTION_PROMPT, RAG_CHUNKED_PROMPT
    from src.extraction.validation import validate_patient_triples, validate_triple

    fp = Path(f"data/mimic_oncology/{args.source}/notes_all.jsonl")
    if not fp.exists():
        sys.exit(f"missing {fp} — run scripts/fetch_mimic_oncology.py first")
    notes = [json.loads(l) for l in open(fp)]
    if args.limit:
        notes = notes[:args.limit]

    out = Path(f"results/extraction/mimic_{args.source}_fast/triples"); out.mkdir(parents=True, exist_ok=True)
    mpath = Path(f"results/mimic_{args.source}_fast_metrics.json")
    metrics = json.load(open(mpath)) if mpath.exists() else []
    done = {m["note_id"] for m in metrics}
    todo = [r for r in notes if str(r.get("note_id")) not in done and len(r.get("text") or "") >= 200]
    log.info("MIMIC %s FAST: %d notes (%d todo) | model=%s note_batch=%d gen_batch=%d gpu=%d",
             args.source, len(notes), len(todo), args.model, args.note_batch, args.gen_batch, args.gpu)

    t0 = time.time(); n_chunks = 0
    for b in range(0, len(todo), args.note_batch):
        batch = todo[b:b + args.note_batch]
        texts = [r.get("text") or "" for r in batch]
        ners = extract_entities_batch(texts)                       # NER for the batch (CPU parallel)
        prompts, owner = [], []
        for j, mentions in enumerate(ners):
            for ci, (ctext, cand) in enumerate(_chunk_with_candidates(texts[j], mentions)):
                cm = [m for m in mentions if m.text.lower() in cand.lower()]
                octx = _retrieve_ontology_context(cm[:30], [])     # [] = no graph dep -> batchable
                full = cand + ("\n" + octx if octx else "")
                tmpl = RAG_EXTRACTION_PROMPT if ci == 0 else RAG_CHUNKED_PROMPT
                prompts.append((RAG_SYSTEM_PROMPT, tmpl.format(candidates=full, narrative=ctext)))
                owner.append(j)
        raws = generate_batch(args.model, prompts, gpu_id=args.gpu,
                              batch_size=args.gen_batch, max_new_tokens=args.max_new)
        n_chunks += len(prompts)
        note_tri = [[] for _ in batch]
        for oj, raw in zip(owner, raws):
            note_tri[oj].extend(_parse_json_response(raw))
        for j, r in enumerate(batch):
            nid = str(r.get("note_id")); text = texts[j]
            tri = dedup(note_tri[j])
            for t in tri:
                validate_triple(t, text, tri)
            vr = validate_patient_triples(tri, text, trust_threshold=0.4)
            grounded = sum(1 for t in tri if t.get("_validation", {}).get("source_grounding", 0) >= 0.5)
            json.dump({"note_id": nid, "subject_id": r.get("subject_id"), "triples": tri},
                      open(out / f"{nid}.json", "w"), indent=2, default=str)
            metrics.append({"note_id": nid, "subject_id": r.get("subject_id"),
                            "n_triples": len(tri), "n_grounded": grounded,
                            "grounding_rate": round(grounded / max(len(tri), 1), 3),
                            "n_accepted": len(vr["accepted"]), "mean_trust": vr["stats"]["mean_trust"]})
        json.dump(metrics, open(mpath, "w"), indent=2)
        el = time.time() - t0
        log.info("  %d/%d notes | %d chunks | throughput %.1f notes/hr, %.2f chunks/s",
                 len(metrics), len(notes), n_chunks,
                 len(metrics) / (el / 3600) if el > 0 else 0, n_chunks / el if el > 0 else 0)

    # ── aggregate scale + throughput (Table XIV / Volume evidence) ──
    ents = set()
    for f in out.glob("*.json"):
        for t in json.load(open(f)).get("triples", []):
            e = str(t.get("entity", "")).lower().strip()
            if e:
                ents.add(e)
    tot_tri = sum(m["n_triples"] for m in metrics); tot_gr = sum(m["n_grounded"] for m in metrics)
    el = time.time() - t0
    log.info("=" * 60)
    log.info("MIMIC %s FAST DONE: notes=%d triples=%d uniq_entities=%d grounding=%.3f | "
             "throughput=%.1f notes/hr, %.2f chunks/s (this session)",
             args.source, len(metrics), tot_tri, len(ents), tot_gr / max(tot_tri, 1),
             len(todo) / (el / 3600) if el > 0 else 0, n_chunks / el if el > 0 else 0)
    log.info("Saved %s", mpath)


if __name__ == "__main__":
    main()
