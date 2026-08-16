"""Extractor-comparison sweep: evaluate EVERY model combination on CORAL vs gold.

Reads per-model cached triples (results/extraction/<tag>/bymodel/<model>/<pid>.json,
produced by run_ensemble_fast.py --extract-only) and, for each non-empty subset of
models, unions+dedups their triples per patient and scores entity-level P/R/F1 vs the
expert gold — using the SAME matcher as the headline eval (evaluate_single_model).

Extraction is done once per model; all 2^n-1 combinations are then free set-unions.
Ranks combos by mean F1 (recall as tiebreaker). This is the paper's extractor-comparison
experiment and fills Table II (individual model rows + the chosen ensemble).

    python scripts/combo_eval.py --tag combo_coral
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")


def dedup(ts):
    seen, out = set(), []
    for t in ts:
        k = (str(t.get("entity", "")).lower().strip(), str(t.get("attribute", "")).lower().strip(),
             str(t.get("value", "")).lower().strip())
        if k not in seen:
            seen.add(k); out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="combo_coral")
    ap.add_argument("--min-size", type=int, default=1, help="smallest combo size to report")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict sweep to these models (default: all complete caches under bymodel/)")
    ap.add_argument("--require-full", type=int, default=40,
                    help="only include models with >= this many cached patients (skip in-progress)")
    args = ap.parse_args()

    from src.data.reader import load_coral_documents
    from src.extraction.evaluate import evaluate_single_model

    docs = {d.patient_id: d for d in load_coral_documents()}
    bym = Path("results/extraction") / args.tag / "bymodel"
    cand = sorted(p.name for p in bym.iterdir() if p.is_dir())
    if args.models:
        cand = [m for m in cand if m in args.models]
    # only sweep models whose cache is complete (>= require-full patients), so in-progress models don't skew unions
    models = [m for m in cand if len(list((bym / m).glob("*.json"))) >= args.require_full]
    skipped = [m for m in cand if m not in models]
    if not models:
        sys.exit(f"no complete per-model caches (>= {args.require_full}) under {bym}")
    print(f"models swept: {models}" + (f"  | skipped (incomplete): {skipped}" if skipped else ""))

    # load per-model triples: model -> pid -> [triples]
    tri = {m: {} for m in models}
    pids = sorted(docs.keys())
    for m in models:
        for pid in pids:
            f = bym / m / f"{pid}.json"
            tri[m][pid] = json.load(open(f)).get("triples", []) if f.exists() else []

    def score_combo(subset):
        """union subset per patient -> mean entity P/R/F1 per cohort + overall."""
        rows = []
        tmp = Path(tempfile.mkdtemp())
        for pid in pids:
            d = docs[pid]
            pooled = []
            for m in subset:
                pooled += tri[m].get(pid, [])
            ens = dedup(pooled)
            jp = tmp / f"{pid}.json"; json.dump({"triples": ens}, open(jp, "w"), default=str)
            ann = Path(d.metadata["file"].replace(".txt", ".ann.txt"))
            r = evaluate_single_model(jp, ann, d.text, pid)
            rows.append({"pid": pid, "cohort": d.cohort, "n": len(ens),
                         "P": r["entity_precision"], "R": r["entity_recall"], "F1": r["entity_f1"]})
        def agg(rs):
            n = len(rs)
            return None if not n else {
                "n_pat": n, "P": round(sum(r["P"] for r in rs) / n, 4),
                "R": round(sum(r["R"] for r in rs) / n, 4), "F1": round(sum(r["F1"] for r in rs) / n, 4),
                "mean_triples": round(sum(r["n"] for r in rs) / n, 1)}
        return {"overall": agg(rows), "pdac": agg([r for r in rows if r["cohort"] == "pdac"]),
                "brca": agg([r for r in rows if r["cohort"] == "brca"])}

    results = []
    for k in range(args.min_size, len(models) + 1):
        for subset in combinations(models, k):
            s = score_combo(list(subset))
            results.append({"models": list(subset), "k": k, **s})
    results.sort(key=lambda r: (r["overall"]["F1"], r["overall"]["R"]), reverse=True)

    print("\n" + "=" * 96)
    print("EXTRACTOR-COMPARISON SWEEP ON CORAL (entity-level, union vs gold; ranked by overall F1)")
    print("=" * 96)
    print(f"{'combination':52s} {'overallP':>8s} {'overallR':>8s} {'overallF1':>9s} "
          f"{'pdacF1':>7s} {'brcaF1':>7s} {'µtrip':>6s}")
    print("-" * 96)
    for r in results:
        o = r["overall"]
        name = "+".join(m.replace("-instruct", "").replace("32", "").replace("3-", "") for m in r["models"])
        print(f"{name:52s} {o['P']:>8.3f} {o['R']:>8.3f} {o['F1']:>9.3f} "
              f"{r['pdac']['F1']:>7.3f} {r['brca']['F1']:>7.3f} {o['mean_triples']:>6.0f}")
    best = results[0]
    print("-" * 96)
    print(f"BEST by F1: {'+'.join(best['models'])}  ->  "
          f"overall F1={best['overall']['F1']}  R={best['overall']['R']}  P={best['overall']['P']}")
    out = Path(f"results/combo_eval_{args.tag}.json"); json.dump(results, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
