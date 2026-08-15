"""E1-E3 (Veracity): reliability estimation, calibration, and selective admission.

Builds a labeled set over the CORAL ensemble triples:
  features   = 5 validation-layer scores + heuristic trust + structural features
  is_correct = triple matches an expert gold entity (fuzzy, same matcher as the F1 eval)
Then, leakage-safe (fit on TRAIN, Platt on VAL, evaluate on held-out TEST):

  Table VIII (calibration):  ECE / Brier / NLL for  Heuristic | Learned | Learned+Platt
  Table IX  (selective):     AURC / Cov@95% / Insert-Review-Reject for
                             Heuristic | Learned | Calibrated-Selective

Labeling is cached to results/e1e2_labeled.json (slow fuzzy match runs once).

    python scripts/exp_calibration_selective.py
"""
from __future__ import annotations
import glob, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")

VAL_KEYS = ["source_grounding", "ontology_check", "schema_check", "temporal_consistency", "contradiction_score"]
FHIR_CATS = ["Condition", "Observation", "Procedure", "MedicationStatement"]
CACHE = Path("results/e1e2_labeled.json")


def build_labeled():
    from src.data.reader import load_coral_documents, load_ground_truth
    from src.extraction.validation import validate_patient_triples
    from src.extraction.evaluate import _get_triple_texts, _match_score
    from src.graph.rdf_builder import _normalize_fhir_type
    docs = {d.patient_id: d for d in load_coral_documents()}
    out = []
    for f in sorted(glob.glob("results/extraction/ens3/union/*.json")):
        d = json.load(open(f)); pid = d["patient"]; doc = docs[pid]
        gt = load_ground_truth(Path(doc.metadata["file"].replace(".txt", ".ann.txt")))
        gold = list({e["text"] for e in gt})
        triples = d.get("triples", [])
        vr = validate_patient_triples(triples, doc.text, trust_threshold=0.0)
        for t in vr["accepted"] + vr["rejected"]:
            v = t.get("_validation", {})
            trust = float(v.get("trust_score", 0.5))
            ent = str(t.get("entity", "")); val = str(t.get("value", ""))
            fh = _normalize_fhir_type(t.get("fhir_type", ""))
            feat = [v.get(k, 0.5) for k in VAL_KEYS] + [trust,
                    len(ent.split()), len(val.split()),
                    1 if t.get("evidence_span") else 0,
                    1 if str(t.get("temporal_anchor", "")).strip() not in ("", "null", "none") else 0,
                    1 if val.replace(".", "").isdigit() else 0] + [1 if fh == c else 0 for c in FHIR_CATS]
            texts = _get_triple_texts(t)
            correct = int(any(_match_score(tx, g) >= 0.4 for tx in texts for g in gold))
            out.append({"pid": pid, "trust": trust, "correct": correct, "feat": feat})
    json.dump(out, open(CACHE, "w"))
    return out


data = json.load(open(CACHE)) if CACHE.exists() else build_labeled()
from src.config_splits import CORAL_TRAIN, CORAL_VAL, CORAL_TEST
X = np.array([r["feat"] for r in data], float)
y = np.array([r["correct"] for r in data], float)
trust = np.array([r["trust"] for r in data], float)
pid = np.array([r["pid"] for r in data])
tr = np.isin(pid, list(CORAL_TRAIN)); va = np.isin(pid, list(CORAL_VAL)); te = np.isin(pid, list(CORAL_TEST))
print(f"triples {len(data)} | correct-rate {y.mean():.3f} | train {tr.sum()} val {va.sum()} test {te.sum()}")

# ── learned reliability: gradient boosting on features, Platt on val ──
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
gb = GradientBoostingClassifier(max_depth=3, n_estimators=200).fit(X[tr | va], y[tr | va])
learn_te = gb.predict_proba(X[te])[:, 1]
platt = LogisticRegression().fit(gb.predict_proba(X[va])[:, 1].reshape(-1, 1), y[va])
learn_cal_te = platt.predict_proba(learn_te.reshape(-1, 1))[:, 1]
# also Platt-calibrate the heuristic trust (for the VIII heuristic->calibrated view)
platt_h = LogisticRegression().fit(trust[va].reshape(-1, 1), y[va])
trust_cal_te = platt_h.predict_proba(trust[te].reshape(-1, 1))[:, 1]
yte = y[te]


def ece(p, y, b=10):
    e = 0.0
    for i in range(b):
        lo, hi = i / b, (i + 1) / b
        m = (p >= lo) & (p <= hi) if i == b - 1 else (p >= lo) & (p < hi)
        if m.sum(): e += m.sum() / len(p) * abs(y[m].mean() - p[m].mean())
    return e
brier = lambda p, y: float(np.mean((p - y) ** 2))
def nll(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def aurc(c, y):
    o = np.argsort(-c); ys = y[o]
    return float(np.trapz(np.cumsum(1-ys)/np.arange(1, len(ys)+1), np.arange(1, len(ys)+1)/len(ys)))
def cov95(c, y, t=0.95):
    o = np.argsort(-c); ys = y[o]; pr = np.cumsum(ys)/np.arange(1, len(ys)+1)
    ok = np.where(pr >= t)[0]; return float((ok[-1]+1)/len(ys)) if len(ok) else 0.0
def bands(dev_c, dev_y, te_c, te_y, hi=0.95, lo=0.5):
    o = np.argsort(-dev_c); ys, ps = dev_y[o], dev_c[o]; pr = np.cumsum(ys)/np.arange(1, len(ys)+1)
    idx = np.where(pr >= hi)[0]; thi = ps[idx[-1]] if len(idx) else 1.01
    oa = np.argsort(dev_c); ys2, ps2 = dev_y[oa], dev_c[oa]; pl = np.cumsum(ys2)/np.arange(1, len(ys2)+1)
    idl = np.where(pl <= lo)[0]; tlo = ps2[idl[-1]] if len(idl) else -0.01
    ins = te_c >= thi; rej = te_c < tlo; rev = ~ins & ~rej
    ip = te_y[ins].mean() if ins.sum() else float("nan")
    return ins.mean(), rev.mean(), rej.mean(), ip


print("\nTABLE VIII — Reliability estimation & calibration (held-out TEST; lower better)")
print(f"  {'method':24s} {'ECE':>6s} {'Brier':>6s} {'NLL':>6s}")
for name, p in [("Heuristic Trust", trust[te]), ("Learned (uncalibrated)", learn_te), ("Learned + Platt", learn_cal_te)]:
    print(f"  {name:24s} {ece(p,yte):>6.3f} {brier(p,yte):>6.3f} {nll(p,yte):>6.3f}")

print("\nTABLE IX — Selective admission (held-out TEST). Insert/Review/Reject at 95%-precision op point")
print(f"  {'policy':24s} {'AURC':>6s} {'Cov@95':>7s} | {'Insert':>7s} {'Review':>7s} {'Reject':>7s} {'InsPrec':>7s}")
# dev = val (calibrated where applicable) to pick operating point
for name, dev_c, dev_y, te_c in [
        ("Heuristic Trust", trust_cal_te if False else platt_h.predict_proba(trust[va].reshape(-1,1))[:,1], y[va], trust_cal_te),
        ("Learned Reliability", gb.predict_proba(X[va])[:,1], y[va], learn_te),
        ("Calibrated Selective", platt.predict_proba(gb.predict_proba(X[va])[:,1].reshape(-1,1))[:,1], y[va], learn_cal_te)]:
    i, r, j, ip = bands(dev_c, dev_y, te_c, yte)
    print(f"  {name:24s} {aurc(te_c,yte):>6.3f} {cov95(te_c,yte):>7.3f} | {i*100:>6.1f}% {r*100:>6.1f}% {j*100:>6.1f}% {ip:>7.3f}")

json.dump({"n": len(data), "correct_rate": float(y.mean()), "test_n": int(te.sum()),
           "viii": {"heuristic_ece": ece(trust[te], yte), "learned_ece": ece(learn_te, yte),
                    "learned_platt_ece": ece(learn_cal_te, yte)},
           "ix": {"aurc_heuristic": aurc(trust_cal_te, yte), "aurc_learned": aurc(learn_te, yte),
                  "cov95_learned": cov95(learn_te, yte)}},
          open("results/e1e3_results.json", "w"), indent=2)
print("\nSaved results/e1e3_results.json")
