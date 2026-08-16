"""Fast EXACT scorer for the final ensemble on CORAL — memoized + process-parallel.

Scores the recall-first ensemble (Gemma-4-E4B 2-pass  ∪  cached 1-pass llama/qwen/medgemma)
against CORAL gold, entity-level, using the SAME metric as evaluate_single_model
(precision=tp/|triples|, recall=matched_gt/|gt_unique|, match via _match_score>=0.4) — NO
blocking approximation, so numbers are exact.

Speed: memoize _normalize (each unique string normalized once) + ProcessPoolExecutor over
patients (CPU-bound => processes, not threads, to beat the GIL). ~seconds for 40 patients.

    python scripts/fast_score.py
"""
from __future__ import annotations
import functools, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")

FINAL = "results/extraction/coral_final/bymodel"     # gemma4-e4b = 2-pass
COMBO = "results/extraction/combo_coral/bymodel"     # augmenters = cached 1-pass
ANCHOR = "gemma4-e4b"
AUG = ["llama32-3b", "qwen3-4b", "medgemma-4b"]


def ann_path(pid):
    coh, num = pid.split("_", 1)
    d = "breastca" if coh == "brca" else coh
    return f"data/coral/{d}/{num}.ann.txt"


def _load(path):
    return json.load(open(path)).get("triples", []) if Path(path).exists() else []


def score_pid(pid):
    from src.data.reader import load_ground_truth
    import src.extraction.evaluate as ev
    ev._normalize = functools.lru_cache(maxsize=None)(ev._normalize)     # memoize per worker
    from src.extraction.evaluate import _get_triple_texts, _match_score

    tri = _load(f"{FINAL}/{ANCHOR}/{pid}.json") + [t for m in AUG for t in _load(f"{COMBO}/{m}/{pid}.json")]
    seen, uniq = set(), []
    for t in tri:
        k = (str(t.get("entity", "")).lower().strip(), str(t.get("attribute", "")).lower().strip(),
             str(t.get("value", "")).lower().strip())
        if k not in seen:
            seen.add(k); uniq.append(t)

    gu = {}
    for e in load_ground_truth(Path(ann_path(pid))):
        gu.setdefault(e["text"].lower().strip(), e)

    tp, matched = 0, set()
    for t in uniq:
        texts = _get_triple_texts(t); hit = False
        for gk, ge in gu.items():
            if any(_match_score(tt, ge["text"]) >= 0.4 for tt in texts):
                matched.add(gk); hit = True
        tp += hit
    n, ng = len(uniq), len(gu)
    P = tp / n if n else 0.0
    R = len(matched) / ng if ng else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"pid": pid, "cohort": pid.split("_")[0], "n": n, "P": P, "R": R, "F1": F1}


def main():
    pids = [f"pdac_{i}" for i in range(20)] + [f"brca_{i}" for i in range(20, 40)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(16, (os.cpu_count() or 4))) as ex:
        rows = list(ex.map(score_pid, pids))
    import statistics as st, math
    def agg(rs):
        k = len(rs)
        if not k:
            return None
        f1s = [r["F1"] for r in rs]; m = sum(f1s)/k; sd = st.pstdev(f1s); ci = 1.96*sd/math.sqrt(k)
        return dict(n_pat=k, P=round(sum(r["P"] for r in rs)/k, 4), R=round(sum(r["R"] for r in rs)/k, 4),
            F1=round(m, 4), F1_sd=round(sd, 4), CI=[round(m-ci, 4), round(m+ci, 4)],
            mean_triples=round(sum(r["n"] for r in rs)/k, 1))
    out = {"config": "gemma4-e4b(2pass) + llama+qwen+medgemma(1pass)",
           "overall": agg(rows), "pdac": agg([r for r in rows if r["cohort"] == "pdac"]),
           "brca": agg([r for r in rows if r["cohort"] == "brca"])}
    for coh in ("pdac", "brca", "overall"):
        a = out[coh]
        print(f"  CORAL-{coh.upper():7s} P={a['P']:.3f}  R={a['R']:.3f}  F1={a['F1']:.3f} ± {a['F1_sd']:.3f}  "
              f"CI={a['CI']}  (mean {a['mean_triples']:.0f} triples)")
    json.dump(out, open("results/coral_final_score.json", "w"), indent=2)
    print(f"  [{time.time()-t0:.1f}s]  Saved results/coral_final_score.json")


if __name__ == "__main__":
    main()
