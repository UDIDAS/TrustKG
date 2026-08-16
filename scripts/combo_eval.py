"""Extractor-comparison sweep: evaluate EVERY model combination on CORAL vs gold.

Reads per-model cached triples (results/extraction/<tag>/bymodel/<model>/<pid>.json,
produced by run_ensemble_fast.py --extract-only) and, for each non-empty subset of models,
unions+dedups their triples per patient and scores entity-level P/R/F1 vs the expert gold —
same matcher/metric as evaluate_single_model (precision=tp_triples/|triples|,
recall=matched_gt_unique/|gt_unique|, match via _match_score>=0.4).

FAST + no model loading: maps patient ids straight to data/coral/<cohort>/<n>.ann.txt (skips
load_coral_documents, which would load SciSpaCy/MedCPT); memoizes _normalize; precomputes each
unique triple's matched-gold set ONCE per patient, then every 2^n-1 combination is a pure
set-union. Runs in seconds on CPU. Ranks combos by mean F1 (recall tiebreaker).

    python scripts/combo_eval.py --tag combo_coral
"""
from __future__ import annotations
import argparse, functools, json, os, sys, time
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")


def tkey(t):
    return (str(t.get("entity", "")).lower().strip(),
            str(t.get("attribute", "")).lower().strip(),
            str(t.get("value", "")).lower().strip())


def ann_path(pid):                       # pdac_0 -> data/coral/pdac/0.ann.txt ; brca_20 -> data/coral/breastca/20.ann.txt
    coh, num = pid.split("_", 1)
    d = "breastca" if coh == "brca" else coh
    return Path(f"data/coral/{d}/{num}.ann.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="combo_coral")
    ap.add_argument("--min-size", type=int, default=1)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--require-full", type=int, default=40)
    args = ap.parse_args()

    from src.data.reader import load_ground_truth       # pure .ann.txt parser, no models
    import src.extraction.evaluate as ev
    ev._normalize = functools.lru_cache(maxsize=None)(ev._normalize)   # memoize normalization
    from src.extraction.evaluate import _get_triple_texts, _match_score

    t0 = time.time()
    bym = Path("results/extraction") / args.tag / "bymodel"
    cand = sorted(p.name for p in bym.iterdir() if p.is_dir())
    if args.models:
        cand = [m for m in cand if m in args.models]
    models = [m for m in cand if len(list((bym / m).glob("*.json"))) >= args.require_full]
    skipped = [m for m in cand if m not in models]
    if not models:
        sys.exit(f"no complete per-model caches (>= {args.require_full}) under {bym}")
    pids = sorted({p.stem for m in models for p in (bym / m).glob("*.json")})
    print(f"models swept: {models}" + (f" | skipped: {skipped}" if skipped else "") + f" | {len(pids)} patients", flush=True)

    # per-model per-patient triple-key sets + a representative triple per key
    mkeys = {m: {} for m in models}
    rep = {p: {} for p in pids}
    for m in models:
        for pid in pids:
            f = bym / m / f"{pid}.json"
            ks = set()
            for t in (json.load(open(f)).get("triples", []) if f.exists() else []):
                k = tkey(t); ks.add(k); rep[pid].setdefault(k, t)
            mkeys[m][pid] = ks

    # PRECOMPUTE once: per patient, gt_unique + each unique triple-key's matched-gold set.
    # BLOCKING: only _match_score a triple against gold entities sharing >=1 normalized token
    # (exact/contains/jaccard all require token overlap; skips only rare no-overlap abbrev matches).
    from collections import defaultdict
    matched, n_gold = {}, {}
    for i, pid in enumerate(pids):
        gts = load_ground_truth(ann_path(pid))
        gu = {}
        for e in gts:
            gu.setdefault(e["text"].lower().strip(), e)
        n_gold[pid] = len(gu)
        tok2gold = defaultdict(set)
        for gk, ge in gu.items():
            for tk in ev._normalize(ge["text"]).split():
                tok2gold[tk].add(gk)
        mm = {}
        for k, t in rep[pid].items():
            texts = _get_triple_texts(t)
            cand = set()
            for tt in texts:
                for tk in ev._normalize(tt).split():
                    cand |= tok2gold.get(tk, set())
            s = {gk for gk in cand if any(_match_score(tt, gu[gk]["text"]) >= 0.4 for tt in texts)}
            mm[k] = s
        matched[pid] = mm
        print(f"  precompute {i+1}/{len(pids)} ({pid}: {len(rep[pid])} triples, {len(gu)} gold)  [{time.time()-t0:.0f}s]", flush=True)

    def score(subset):
        rows = []
        for pid in pids:
            uk = set().union(*(mkeys[m][pid] for m in subset)) if subset else set()
            n = len(uk); tp = sum(1 for k in uk if matched[pid][k])
            mg = set().union(*(matched[pid][k] for k in uk)) if uk else set()
            P = tp / n if n else 0.0
            R = len(mg) / n_gold[pid] if n_gold[pid] else 0.0
            F1 = 2 * P * R / (P + R) if (P + R) else 0.0
            rows.append((pid.split("_")[0], n, P, R, F1))
        def agg(rs):
            k = len(rs)
            return None if not k else {"P": round(sum(r[2] for r in rs)/k, 4), "R": round(sum(r[3] for r in rs)/k, 4),
                    "F1": round(sum(r[4] for r in rs)/k, 4), "mean_triples": round(sum(r[1] for r in rs)/k, 1)}
        return {"overall": agg(rows), "pdac": agg([r for r in rows if r[0] == "pdac"]),
                "brca": agg([r for r in rows if r[0] == "brca"])}

    results = []
    for kk in range(args.min_size, len(models) + 1):
        for subset in combinations(models, kk):
            results.append({"models": list(subset), "k": kk, **score(list(subset))})
    results.sort(key=lambda r: (r["overall"]["F1"], r["overall"]["R"]), reverse=True)

    sh = {"gemma3-4b": "gem3", "gemma4-e4b": "gem4", "medgemma-4b": "medg", "llama32-3b": "llama", "qwen3-4b": "qwen"}
    print("\n" + "=" * 90)
    print("EXTRACTOR-COMPARISON SWEEP ON CORAL (entity-level, 1-pass union vs gold; ranked by overall F1)")
    print("=" * 90)
    print(f"{'combination':32s} {'oP':>6s} {'oR':>6s} {'oF1':>6s} {'pdacF1':>7s} {'brcaF1':>7s} {'µtrip':>6s}")
    print("-" * 90)
    for r in results:
        o = r["overall"]
        print(f"{'+'.join(sh.get(m, m) for m in r['models']):32s} {o['P']:>6.3f} {o['R']:>6.3f} {o['F1']:>6.3f} "
              f"{r['pdac']['F1']:>7.3f} {r['brca']['F1']:>7.3f} {o['mean_triples']:>6.0f}")
    best = results[0]
    print("-" * 90)
    print(f"BEST by F1: {'+'.join(best['models'])} -> F1={best['overall']['F1']} R={best['overall']['R']} P={best['overall']['P']}")
    json.dump(results, open(f"results/combo_eval_{args.tag}.json", "w"), indent=2)
    print(f"[{time.time()-t0:.0f}s] Saved results/combo_eval_{args.tag}.json")


if __name__ == "__main__":
    main()
