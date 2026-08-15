"""E10 (Value V): where keyword search FAILS and KG semantics wins.

Naive keyword cohort retrieval matches any note containing the term — including
NEGATED ("no evidence of metastasis") and FAMILY/EXPERIENCER ("mother had breast
cancer") mentions, which are FALSE POSITIVES for "the patient has X". The KG admits
only asserted, extracted entities, so it should exclude these.

We target negation/experiencer-prone oncology concepts, find patients whose ONLY
mentions are negated/family (= keyword false positives), and measure how many the
KG correctly excludes.

    python scripts/exp_value_baseline.py
"""
from __future__ import annotations
import glob, json, os, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
from src.data.reader import load_coral_documents

CONCEPTS = ["metasta", "recurren", "effusion", "lymphadenopathy", "adenopathy",
            "nodule", "lesion", "edema", "ascites", "thrombosis"]
NEG = re.compile(r"\b(no|not|without|denies|denied|negative for|r/o|rule[sd]? out|"
                 r"no evidence of|free of|absence of|resolved|unremarkable|"
                 r"family history|mother|father|sister|brother|maternal|paternal|aunt|uncle)\b", re.I)

docs = {d.patient_id: d for d in load_coral_documents()}
raw = {p: d.text.lower() for p, d in docs.items()}
kg = {}
for f in glob.glob("results/extraction/ens3/union/*.json"):
    d = json.load(open(f))
    kg[d["patient"]] = " ||| ".join(f"{t.get('entity','')} {t.get('value','')}".lower()
                                    for t in d.get("triples", []))


def mentions(text, concept):
    """(n_total, n_positive) — positive = occurrence NOT preceded by negation/family in a 55-char window."""
    tot = pos = 0
    for m in re.finditer(re.escape(concept), text):
        tot += 1
        if not NEG.search(text[max(0, m.start() - 55):m.start()]):
            pos += 1
    return tot, pos


rows, KWFP, KGEXCL = [], 0, 0
for c in CONCEPTS:
    kw_pts, neg_only = set(), set()
    for p in docs:
        t, po = mentions(raw[p], c)
        if t:
            kw_pts.add(p)
            if po == 0:                       # term present, but ONLY in negated/family context
                neg_only.add(p)               #   -> keyword FALSE POSITIVE ("patient has X" is false)
    kg_pts = {p for p in docs if c in kg.get(p, "")}
    kg_excl = {p for p in neg_only if p not in kg_pts}   # KG correctly did NOT assert it
    KWFP += len(neg_only); KGEXCL += len(kg_excl)
    if kw_pts:
        rows.append((c, len(kw_pts), len(neg_only), len(kg_excl)))

print("E10 — keyword semantic failures (negation/family) that the KG avoids   [CORAL, 40 patients]")
print("=" * 84)
print(f"{'concept':16s} {'keyword-cohort':>14s} {'keyword FALSE-POS':>18s} {'KG excluded them':>18s}")
print(f"{'':16s} {'(note has term)':>14s} {'(negated/family only)':>18s} {'(correct)':>18s}")
print("-" * 84)
for c, kwn, fp, ex in rows:
    print(f"{c:16s} {kwn:>14d} {fp:>18d} {ex:>18d}")
print("-" * 84)
rate = KGEXCL / KWFP if KWFP else 0.0
print(f"{'TOTAL':16s} {'':>14s} {KWFP:>18d} {KGEXCL:>18d}   -> KG avoids {rate*100:.0f}% of keyword false positives")
print(f"\nKeyword search retrieves {KWFP} patients whose ONLY mention of the concept is negated/family")
print(f"(false positives for 'patient has X'); the KG's semantic extraction excludes {KGEXCL} of them ({rate*100:.0f}%).")
json.dump({"rows": rows, "kw_false_pos": KWFP, "kg_excluded": KGEXCL, "avoidance_rate": rate},
          open("results/e10_value_baseline.json", "w"), indent=2)
