"""Throughput-optimized ENSEMBLE extractor — one consistent engine for CORAL and MIMIC.

Same framework everywhere (the point): candidate generation by an ensemble of models
(Gemma-3-4B 2-pass anchor + single-pass Qwen3-8B / Llama-3.2-3B augmenters), unioned and
deduped, then the identical construction-time validation. The ONLY speed change vs the
per-chunk CORAL runner is BATCHING: resident model + many chunk-prompts per GPU forward
pass. This is the Volume/RQ4 artifact and it logs throughput (notes/hr, chunks/s) -> Table VII.

Structure (efficient + resumable):
  Phase A  extraction, per MODEL (each loaded once, resident across all its batches),
           checkpointed per note-batch -> results/extraction/<tag>/bymodel/<model>/<id>.json
  Phase B  CPU union across models + dedup + validate (fixed grounding) -> union/ + filtered/,
           metrics (+ gold P/R/F1 when --dataset coral).

    python scripts/run_ensemble_fast.py --dataset mimiciii --gpu 0 \
        --models gemma3-4b qwen3-8b llama32-3b --twopass gemma3-4b
    python scripts/run_ensemble_fast.py --dataset coral --gpu 0 --limit 8   # benchmark
"""
from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)   # extraction never touches BigQuery
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ens_fast")


# per-model generation batch size — the 8B model OOMs at large batch on long clinical
# notes (KV cache for long input + max_new), so size it down for bigger models.
GEN_BATCH = {"gemma3-4b": 6, "qwen3-8b": 2, "qwen3-4b": 2, "medgemma-4b": 6,
             "gemma4-e4b": 4, "phi4-mini": 4, "llama32-3b": 4}   # gemma4 = 8B, large vocab -> batch 4


def dedup(ts):
    seen, out = set(), []
    for t in ts:
        k = (str(t.get("entity", "")).lower().strip(), str(t.get("attribute", "")).lower().strip(),
             str(t.get("value", "")).lower().strip())
        if k not in seen:
            seen.add(k); out.append(t)
    return out


def load_items(dataset, limit):
    """Return list of {id, text, ann} — unified over CORAL patients and MIMIC notes."""
    if dataset == "coral":
        from src.data.reader import load_coral_documents
        docs = load_coral_documents()
        items = [{"id": d.patient_id, "text": d.text, "cohort": d.cohort,
                  "ann": d.metadata["file"].replace(".txt", ".ann.txt")} for d in docs]
        items.sort(key=lambda x: x["id"])
    else:
        fp = Path(f"data/mimic_oncology/{dataset}/notes_all.jsonl")
        if not fp.exists():
            sys.exit(f"missing {fp} — run scripts/fetch_mimic_oncology.py first")
        items = []
        for l in open(fp):
            r = json.loads(l)
            if len(r.get("text") or "") >= 200:
                items.append({"id": str(r.get("note_id")), "text": r["text"],
                              "subject_id": r.get("subject_id"), "ann": None})
    return items[:limit] if limit else items


def batched_extract(batch, mentions_by_id, model, gpu, gen_batch, max_new, seeds=None):
    """Batched RAG extraction of a note-batch on one model. seeds: id -> pass-1 triples
    (graph-neighborhood evidence for pass 2). Returns id -> list[triple]."""
    from src.extraction.rag_extractor import _chunk_with_candidates, _retrieve_ontology_context
    from src.extraction.local_llm import generate_batch, _parse_json_response
    from src.extraction.prompts import RAG_SYSTEM_PROMPT, RAG_EXTRACTION_PROMPT, RAG_CHUNKED_PROMPT
    prompts, owner = [], []
    for it in batch:
        mentions = mentions_by_id[it["id"]]
        seed = (seeds or {}).get(it["id"], [])
        for ci, (ctext, cand) in enumerate(_chunk_with_candidates(it["text"], mentions)):
            cm = [m for m in mentions if m.text.lower() in cand.lower()]
            octx = _retrieve_ontology_context(cm[:30], seed)     # seed = prior graph evidence (pass 2)
            full = cand + ("\n" + octx if octx else "")
            tmpl = RAG_EXTRACTION_PROMPT if ci == 0 else RAG_CHUNKED_PROMPT
            prompts.append((RAG_SYSTEM_PROMPT, tmpl.format(candidates=full, narrative=ctext)))
            owner.append(it["id"])
    raws = generate_batch(model, prompts, gpu_id=gpu, batch_size=gen_batch, max_new_tokens=max_new)
    out = {it["id"]: [] for it in batch}
    for oid, raw in zip(owner, raws):
        out[oid].extend(_parse_json_response(raw))
    return out, len(prompts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coral", "mimiciii", "mimiciv"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=["gemma3-4b", "qwen3-8b", "llama32-3b"])
    ap.add_argument("--twopass", nargs="*", default=["gemma3-4b"])
    ap.add_argument("--note-batch", type=int, default=16)
    ap.add_argument("--gen-batch", type=int, default=0, help="0 = per-model default (GEN_BATCH); else uniform override")
    ap.add_argument("--max-new", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patients", nargs="*", default=None, help="restrict to these patient ids (split work across GPUs)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed-from", default=None,
                    help="reuse cached 1-pass triples from this tag as pass-1: 1-pass models are copied "
                         "(no GPU), 2-pass models load cached pass-1 and only compute pass-2")
    ap.add_argument("--extract-only", action="store_true",
                    help="cache per-model triples only; skip union+validate (for parallel combo sweeps)")
    args = ap.parse_args()
    tag = args.tag or f"ensfast_{args.dataset}"
    twopass = set(args.twopass)

    from src.extraction.ner import extract_entities_batch
    from src.extraction.fhir_normalizer import normalize_patient_triples
    from src.extraction.validation import validate_patient_triples

    items = load_items(args.dataset, args.limit)
    if args.patients:
        keep = set(args.patients); items = [it for it in items if it["id"] in keep]
    root = Path("results/extraction") / tag
    (root / "union").mkdir(parents=True, exist_ok=True)
    (root / "filtered").mkdir(parents=True, exist_ok=True)
    bym = root / "bymodel"
    log.info("ENSEMBLE-FAST %s | items=%d | models=%s twopass=%s | tag=%s gpu=%d",
             args.dataset, len(items), args.models, sorted(twopass), tag, args.gpu)

    # cache NER once per item (reused by every model + both passes)
    ner_cache = root / "ner_mentions.json"        # only ids -> mention spans (cheap CPU, resumable)
    from src.extraction.ner import EntityMention
    mentions_by_id = {}
    if ner_cache.exists():
        raw = json.load(open(ner_cache))
        for i, ms in raw.items():
            mentions_by_id[i] = [EntityMention(**m) for m in ms]
    todo_ner = [it for it in items if it["id"] not in mentions_by_id]
    for b in range(0, len(todo_ner), 32):
        chunk = todo_ner[b:b + 32]
        for it, ms in zip(chunk, extract_entities_batch([x["text"] for x in chunk])):
            mentions_by_id[it["id"]] = ms
        json.dump({i: [vars(m) for m in ms] for i, ms in mentions_by_id.items()}, open(ner_cache, "w"))
    log.info("NER ready for %d items", len(mentions_by_id))

    # ── Phase A: extraction per model (each model loaded once, resident) ──
    t0 = time.time(); tot_chunks = 0
    for model in args.models:
        mdir = bym / model; mdir.mkdir(parents=True, exist_ok=True)
        gb = args.gen_batch or GEN_BATCH.get(model, 4)
        todo = [it for it in items if not (mdir / f"{it['id']}.json").exists()]
        log.info("--- model %s (%s, gen_batch=%d): %d/%d to do ---",
                 model, "2pass" if model in twopass else "1pass", gb, len(todo), len(items))
        for b in range(0, len(todo), args.note_batch):
            batch = todo[b:b + args.note_batch]
            if args.seed_from:                        # reuse cached pass-1 (no GPU for pass-1)
                sd = Path("results/extraction") / args.seed_from / "bymodel" / model
                p1 = {it["id"]: json.load(open(sd / f"{it['id']}.json")).get("triples", []) for it in batch}
                nc = 0
            else:
                p1, nc = batched_extract(batch, mentions_by_id, model, args.gpu, gb, args.max_new)
            tot_chunks += nc
            if model in twopass:
                seeds = {}
                for it in batch:
                    s = [dict(t) for t in p1[it["id"]]]
                    normalize_patient_triples(s); seeds[it["id"]] = s
                p2, nc2 = batched_extract(batch, mentions_by_id, model, args.gpu,
                                          gb, args.max_new, seeds=seeds)
                tot_chunks += nc2
                res = {it["id"]: dedup(p1[it["id"]] + p2[it["id"]]) for it in batch}
            else:
                res = {it["id"]: dedup(p1[it["id"]]) for it in batch}
            for it in batch:
                json.dump({"id": it["id"], "triples": res[it["id"]]},
                          open(mdir / f"{it['id']}.json", "w"), indent=2, default=str)
            el = time.time() - t0
            log.info("  [%s] %d/%d | %d chunks | %.1f notes/hr, %.2f chunks/s",
                     model, min(b + args.note_batch, len(todo)), len(todo), tot_chunks,
                     ((b + len(batch)) / (el / 3600)) if el > 0 else 0, tot_chunks / el if el > 0 else 0)
    phaseA = time.time() - t0
    if args.extract_only:
        log.info("EXTRACT-ONLY done: %d models cached under %s/bymodel | Phase A %.1f notes/hr",
                 len(args.models), root, len(items) / (phaseA / 3600) if phaseA > 0 else 0)
        return

    # ── Phase B: union across models + validate (CPU) ──
    from src.extraction.evaluate import evaluate_single_model
    metrics = []
    for it in items:
        pooled = []
        for model in args.models:
            f = bym / model / f"{it['id']}.json"
            if f.exists():
                pooled += json.load(open(f)).get("triples", [])
        ens = dedup(pooled)
        vr = validate_patient_triples(ens, it["text"], trust_threshold=0.4)
        validated = vr["accepted"] + vr["rejected"]        # these carry _validation (grounding fix)
        grounded = sum(1 for t in validated if t["_validation"]["source_grounding"] >= 0.5)
        json.dump({"id": it["id"], "triples": validated},
                  open(root / "union" / f"{it['id']}.json", "w"), indent=2, default=str)
        json.dump({"id": it["id"], "triples": vr["accepted"]},
                  open(root / "filtered" / f"{it['id']}.json", "w"), indent=2, default=str)
        row = {"id": it["id"], "n_ensemble": len(ens), "n_filtered": len(vr["accepted"]),
               "n_grounded": grounded, "grounding_rate": round(grounded / max(len(ens), 1), 3),
               "mean_trust": vr["stats"]["mean_trust"]}
        if args.dataset == "coral":
            row["cohort"] = it.get("cohort")
            for stage in ["union", "filtered"]:
                m = evaluate_single_model(root / stage / f"{it['id']}.json", Path(it["ann"]), it["text"], it["id"])
                row[f"{stage}_P"], row[f"{stage}_R"], row[f"{stage}_F1"] = \
                    m["entity_precision"], m["entity_recall"], m["entity_f1"]
        metrics.append(row)
    mpath = Path(f"results/{tag}_metrics.json"); json.dump(metrics, open(mpath, "w"), indent=2)

    tot_tri = sum(m["n_ensemble"] for m in metrics)
    ents = set()
    for f in (root / "union").glob("*.json"):
        for t in json.load(open(f)).get("triples", []):
            e = str(t.get("entity", "")).lower().strip()
            if e: ents.add(e)
    log.info("=" * 64)
    log.info("DONE %s: notes=%d triples=%d uniq_entities=%d grounding=%.3f",
             args.dataset, len(metrics), tot_tri, len(ents),
             sum(m["n_grounded"] for m in metrics) / max(tot_tri, 1))
    log.info("THROUGHPUT (ensemble, Phase A): %.1f notes/hr, %.2f chunks/s over %d notes",
             len(items) / (phaseA / 3600) if phaseA > 0 else 0, tot_chunks / phaseA if phaseA > 0 else 0, len(items))
    if args.dataset == "coral":
        import statistics as st
        for coh in ["brca", "pdac"]:
            f1 = [m["union_F1"] for m in metrics if m.get("cohort") == coh]
            if f1: log.info("  CORAL-%s: n=%d mean_F1=%.3f (sd %.3f)", coh.upper(), len(f1), st.mean(f1), st.pstdev(f1))
    log.info("Saved %s", mpath)


if __name__ == "__main__":
    main()
