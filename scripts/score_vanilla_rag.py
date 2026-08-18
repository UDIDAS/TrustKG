"""Vanilla-RAG baseline for Table III — single-model (Gemma-4-E4B) 1-pass RAG extraction on CORAL,
scored with the SAME entity-level metric as the ensemble (evaluate_single_model / fast_score), so it is
a fair floor in the progression Vanilla RAG -> Gemma-4 anchor (2-pass) -> Ensemble. No GPU: scores the
cached `combo_coral/gemma4-e4b` 1-pass extraction.

    python scripts/score_vanilla_rag.py
"""
from __future__ import annotations
import functools
import json
import math
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
SRC = "results/extraction/combo_coral/bymodel/gemma4-e4b"   # single model, 1-pass


def ann_path(pid):
    coh, num = pid.split("_", 1)
    d = "breastca" if coh == "brca" else coh
    return f"data/coral/{d}/{num}.ann.txt"


def _load(path):
    return json.load(open(path)).get("triples", []) if Path(path).exists() else []


def score_pid(pid):
    from src.data.reader import load_ground_truth
    import src.extraction.evaluate as ev
    ev._normalize = functools.lru_cache(maxsize=None)(ev._normalize)
    from src.extraction.evaluate import _get_triple_texts, _match_score

    seen, uniq = set(), []
    for t in _load(f"{SRC}/{pid}.json"):
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
    with ProcessPoolExecutor(max_workers=min(16, (os.cpu_count() or 4))) as ex:
        rows = list(ex.map(score_pid, pids))

    def agg(rs):
        k = len(rs); f1s = [r["F1"] for r in rs]; m = sum(f1s) / k
        sd = st.pstdev(f1s); ci = 1.96 * sd / math.sqrt(k)
        return dict(P=round(sum(r["P"] for r in rs) / k, 4), R=round(sum(r["R"] for r in rs) / k, 4),
                    F1=round(m, 4), F1_sd=round(sd, 4), CI=[round(m - ci, 4), round(m + ci, 4)],
                    mean_triples=round(sum(r["n"] for r in rs) / k, 1))

    out = {"config": "Vanilla RAG = Gemma-4-E4B single-model, 1-pass (no 2-pass, no ensemble)",
           "pdac": agg([r for r in rows if r["cohort"] == "pdac"]),
           "brca": agg([r for r in rows if r["cohort"] == "brca"])}
    print("Vanilla-RAG baseline (Table III floor):")
    for coh in ("pdac", "brca"):
        a = out[coh]
        print(f"  CORAL-{coh.upper():5s} P={a['P']:.3f}  R={a['R']:.3f}  F1={a['F1']:.3f} ± {a['F1_sd']:.3f}  "
              f"CI={a['CI']}  (mean {a['mean_triples']:.0f} triples)")
    json.dump(out, open("results/vanilla_rag_score.json", "w"), indent=2)
    print("  Saved results/vanilla_rag_score.json")


if __name__ == "__main__":
    main()
