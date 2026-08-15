"""E1-E2 (Veracity): calibration + selective admission of extracted triples.

For every CORAL ensemble triple we form (trust_score, is_correct):
  trust_score = rule-based validation trust (recomputed);
  is_correct  = the triple's entity/value text falls within an expert gold span.
Then:
  E1 (Table VIII): ECE / Brier / NLL of trust as P(correct), uncalibrated vs Platt-calibrated.
  E2 (Table IX):   risk-coverage -> AURC (lower better) and Coverage@95%
                   (fraction auto-insertable at >=95% retained-precision).
Leakage-safe: Platt fit on TRAIN+VAL patients, evaluated on held-out TEST patients.

    python scripts/exp_calibration_selective.py
"""
from __future__ import annotations
import glob, json, os, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
from src.data.reader import load_coral_documents, load_ground_truth
from src.extraction.validation import validate_patient_triples
from src.config_splits import CORAL_TRAIN, CORAL_VAL, CORAL_TEST

_norm = lambda s: re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
_ws = lambda s: re.sub(r"\s+", " ", s).strip()

docs = {d.patient_id: d for d in load_coral_documents()}
rows = []  # (pid, trust, correct)
for f in glob.glob("results/extraction/ens3/union/*.json"):
    d = json.load(open(f)); pid = d["patient"]; doc = docs[pid]
    gt = load_ground_truth(Path(doc.metadata["file"].replace(".txt", ".ann.txt")))
    gold_blob = " ||| ".join(_ws(_norm(e["text"])) for e in gt)
    triples = d.get("triples", [])
    vr = validate_patient_triples(triples, doc.text, trust_threshold=0.0)  # recompute trust
    for t in vr["accepted"] + vr["rejected"]:
        trust = t.get("_validation", {}).get("trust_score", 0.5)
        ent = _ws(_norm(t.get("entity", ""))); val = _ws(_norm(t.get("value", "")))
        correct = int((len(ent) >= 3 and ent in gold_blob) or (len(val) >= 3 and val in gold_blob))
        rows.append((pid, float(trust), correct))

arr = np.array([(t, c) for _, t, c in rows], float)
pid = np.array([p for p, _, _ in rows])
tr_mask = np.isin(pid, list(CORAL_TRAIN) + list(CORAL_VAL))
te_mask = np.isin(pid, list(CORAL_TEST))
print(f"triples: {len(rows)} | overall correct rate {arr[:,1].mean():.3f} | "
      f"train+val {tr_mask.sum()} / test {te_mask.sum()}")


def ece(p, y, bins=10):
    e, n = 0.0, len(p)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


def brier(p, y): return float(np.mean((p - y) ** 2))
def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def aurc(conf, y):
    """Area under risk-coverage curve (lower = better)."""
    order = np.argsort(-conf); ys = y[order]
    cov = np.arange(1, len(ys) + 1) / len(ys)
    risk = np.cumsum(1 - ys) / np.arange(1, len(ys) + 1)   # cumulative error rate
    return float(np.trapz(risk, cov))


def coverage_at(conf, y, target=0.95):
    """Max coverage whose retained precision >= target (sweeping the threshold)."""
    order = np.argsort(-conf); ys = y[order]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.where(prec >= target)[0]
    return float((ok[-1] + 1) / len(ys)) if len(ok) else 0.0


tr_p, tr_y = arr[tr_mask, 0], arr[tr_mask, 1]
te_p, te_y = arr[te_mask, 0], arr[te_mask, 1]

# Platt scaling: logistic regression trust -> P(correct), fit on train, apply to test
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression().fit(tr_p.reshape(-1, 1), tr_y)
te_cal = lr.predict_proba(te_p.reshape(-1, 1))[:, 1]

print("\nE1 — Calibration (held-out TEST)   [lower ECE/Brier/NLL better]")
print(f"  {'method':22s} {'ECE':>7s} {'Brier':>7s} {'NLL':>7s}")
print(f"  {'Heuristic trust (raw)':22s} {ece(te_p,te_y):>7.3f} {brier(te_p,te_y):>7.3f} {nll(te_p,te_y):>7.3f}")
print(f"  {'Heuristic + Platt':22s} {ece(te_cal,te_y):>7.3f} {brier(te_cal,te_y):>7.3f} {nll(te_cal,te_y):>7.3f}")

print("\nE2 — Selective admission (held-out TEST)   [lower AURC / higher Cov@95% better]")
print(f"  {'policy':22s} {'AURC':>7s} {'Cov@95%':>8s}")
print(f"  {'Admit-all (no select)':22s} {'  -   ':>7s} {'  -   ':>8s}   (precision = {te_y.mean():.3f})")
print(f"  {'Heuristic trust':22s} {aurc(te_p,te_y):>7.3f} {coverage_at(te_p,te_y):>8.3f}")
print(f"  {'Calibrated selective':22s} {aurc(te_cal,te_y):>7.3f} {coverage_at(te_cal,te_y):>8.3f}")

json.dump({"n": len(rows), "test_precision": float(te_y.mean()),
           "ece_raw": ece(te_p, te_y), "ece_platt": ece(te_cal, te_y),
           "brier_raw": brier(te_p, te_y), "brier_platt": brier(te_cal, te_y),
           "aurc_heuristic": aurc(te_p, te_y), "aurc_calibrated": aurc(te_cal, te_y),
           "cov95_heuristic": coverage_at(te_p, te_y), "cov95_calibrated": coverage_at(te_cal, te_y)},
          open("results/e1e2_calibration_selective.json", "w"), indent=2)
print("\nSaved results/e1e2_calibration_selective.json")
