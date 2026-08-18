"""Show the trust gate working ON the materialized KG (CORAL, gold-scored), using the
paper's actual mechanism: the LEARNED calibrated reliability model at the 99% operating point
(identical fit/split to Table II, so the numbers match — Union 0.952 -> Admitted 0.981).

Surfaces the triples the gate HOLDS BACK (review/reject) that are genuinely wrong, with the
per-layer scores that caught them — including the ungrounded/low-ontology ones a threshold-free
or rule-based method would keep.  Fast: reuses results/e1e2_labeled_coralfinal.json (features +
gold) and joins triple text by index (no re-validation).

    python scripts/trust_admission_demo.py
"""
from __future__ import annotations
import glob
import json
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from src.config_splits import CORAL_TRAIN, CORAL_VAL, CORAL_TEST

CACHE = "results/e1e2_labeled_coralfinal.json"
UNION = "results/extraction/coral_final/union"
LAYER = ["grounding", "ontology", "schema", "temporal", "contradiction"]   # feat[0:5]

data = json.load(open(CACHE))
# union triples in the SAME order build_labeled used (sorted files, triples in file order)
texts = []
for f in sorted(glob.glob(f"{UNION}/*.json")):
    for t in json.load(open(f)).get("triples", []):
        texts.append(t if isinstance(t, dict) else {})
assert len(texts) == len(data), f"align mismatch: {len(texts)} texts vs {len(data)} labeled"

X = np.array([r["feat"] for r in data], float)
y = np.array([r["correct"] for r in data], float)
pid = np.array([r["pid"] for r in data])
tr = np.isin(pid, list(CORAL_TRAIN)); va = np.isin(pid, list(CORAL_VAL)); te = np.isin(pid, list(CORAL_TEST))

# identical fit to Table II (seeded)
gb = GradientBoostingClassifier(max_depth=3, n_estimators=200, random_state=0).fit(X[tr | va], y[tr | va])
platt = LogisticRegression().fit(gb.predict_proba(X[va])[:, 1].reshape(-1, 1), y[va])
learn_cal_te = platt.predict_proba(gb.predict_proba(X[te])[:, 1].reshape(-1, 1))[:, 1]
dev_c_cal = platt.predict_proba(gb.predict_proba(X[va])[:, 1].reshape(-1, 1))[:, 1]
yte = y[te]


def bands(dev_c, dev_y, te_c, hi=0.99, lo=0.5):
    o = np.argsort(-dev_c); ys, ps = dev_y[o], dev_c[o]; pr = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    idx = np.where(pr >= hi)[0]; thi = ps[idx[-1]] if len(idx) else 1.01
    oa = np.argsort(dev_c); ys2, ps2 = dev_y[oa], dev_c[oa]; pl = np.cumsum(ys2) / np.arange(1, len(ys2) + 1)
    idl = np.where(pl <= lo)[0]; tlo = ps2[idl[-1]] if len(idl) else -0.01
    ins = te_c >= thi; rej = (te_c < tlo) & ~ins; rev = ~ins & ~rej
    return ins, rev, rej


ins, rev, rej = bands(dev_c_cal, y[va], learn_cal_te, hi=0.99)
te_idx = np.where(te)[0]
tex = [texts[i] for i in te_idx]
feat_te = X[te]

print("=" * 74)
print("TRUST GATE ON THE MATERIALIZED KG — CORAL held-out test (learned model @ 99% target)")
print("=" * 74)
print(f"Union KG (pass-through):  {te.sum():5d} triples   precision {yte.mean():.3f}")
print(f"Trust-admitted KG:        {int(ins.sum()):5d} triples   precision {yte[ins].mean():.3f}"
      f"   (+{yte[ins].mean()-yte.mean():.3f})")
print(f"Held back (review+reject):{int((rev|rej).sum()):5d} triples "
      f"(review {int(rev.sum())} / reject {int(rej.sum())})")
held = ~ins
hw = held & (yte == 0)
print(f"  → of the {int(held.sum())} held back, {int(hw.sum())} are genuinely WRONG "
      f"({100*hw.sum()/max(held.sum(),1):.0f}%); precision of the held-back set is {yte[held].mean():.3f}"
      f" vs {yte[ins].mean():.3f} admitted")

print("\nWRONG triples the gate held back — with the layer scores that caught them")
print("(low grounding = text not in note; low ontology = not a recognized concept):")
order = [i for i in range(len(tex)) if held[i] and yte[i] == 0]
order.sort(key=lambda i: learn_cal_te[i])
saved = []
for i in order[:14]:
    t = tex[i]; fv = feat_te[i]
    sig = " ".join(f"{LAYER[j]}={fv[j]:.2f}" for j in range(5))
    print(f"  {str(t.get('entity'))[:24]!r:26} --{str(t.get('attribute'))[:16]}--> "
          f"{str(t.get('value'))[:20]!r:22} | score={learn_cal_te[i]:.2f} | {sig}")
    saved.append({"entity": t.get("entity"), "attribute": t.get("attribute"), "value": t.get("value"),
                  "learned_score": round(float(learn_cal_te[i]), 3),
                  **{LAYER[j]: round(float(fv[j]), 3) for j in range(5)},
                  "evidence": str(t.get("evidence_span", ""))[:80]})

json.dump({"target": 0.99, "union_n": int(te.sum()), "union_prec": round(float(yte.mean()), 4),
           "admitted_n": int(ins.sum()), "admitted_prec": round(float(yte[ins].mean()), 4),
           "held_back_n": int(held.sum()), "held_back_wrong": int(hw.sum()),
           "held_back_prec": round(float(yte[held].mean()), 4),
           "discarded_examples": saved}, open("results/trust_admission_demo.json", "w"), indent=2, default=str)
print("\nSaved results/trust_admission_demo.json")
