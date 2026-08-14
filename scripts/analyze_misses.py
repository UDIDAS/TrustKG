"""Per-patient miss tracking: which gold entities the extractor fails to capture.

For every patient it records the missed gold entities, categorized two ways:
  1. by GOLD LABEL   (PROBLEM / TEST / TREATMENT / ClinicalCondition / ...)
  2. by STRUCTURAL BUCKET:
       - anaphora        "the mass", "that procedure"        (eval ceiling)
       - lab_fragment    "g/dL", "x10E9", "U/L", ranges       (eval ceiling)
       - section_header  "PRIOR SURGERIES", "Vitals"          (eval ceiling)
       - abbreviation    "IHC", "LVEF", "HRT"                 (borderline)
       - substantive     real clinical entities we should catch (RECOVERABLE)

Aggregates recall-by-label (which entity types are worst captured) and the
recurring substantive misses across patients -> concrete fine-tuning targets
for the next extractor version.

    python scripts/analyze_misses.py --extraction results/extraction/ens3/union
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
from src.data.reader import load_coral_documents, load_ground_truth
from src.extraction.evaluate import _get_triple_texts, _match_score

_GENERIC = re.compile(r'^(the|a|an|this|that|these|those|his|her|their|its|our|your|my|some|any|no|it|there)\b', re.I)
_UNIT = re.compile(r'(g/dl|mg/dl|x10e|u/l|u/ml|mmol|ng/ml|mcg|mm\s*hg|sig/nuc|/nuc|ref\s*range|^\W*\d|\d\s*(mg|ml|cm|mm|%|u/l))', re.I)
_HEADERS = {"vitals", "medications", "imaging", "findings", "impression", "assessment",
            "allergies", "allergen reactions", "prior surgeries", "laboratory results",
            "social history", "family history", "range"}


def bucketize(text: str) -> str:
    t = text.strip(); tl = t.lower()
    if tl in _HEADERS or (t.isupper() and len(t) > 5): return "section_header"
    if _UNIT.search(t): return "lab_fragment"
    if _GENERIC.match(t): return "anaphora"
    if t.isupper() and len(t) <= 5: return "abbreviation"
    return "substantive"


def analyze(extraction_dir: str, out_json: str, out_txt: str):
    docs = {d.patient_id: d for d in load_coral_documents()}
    files = sorted(Path(extraction_dir).glob("*.json"))
    per_patient, agg_label, agg_bucket, recurring = [], defaultdict(lambda: [0, 0]), Counter(), Counter()

    for f in files:
        pid = f.stem
        if pid not in docs: continue
        d = docs[pid]
        triples = json.load(open(f)).get("triples", [])
        gt = load_ground_truth(Path(d.metadata["file"].replace(".txt", ".ann.txt")))
        guniq = {}
        for e in gt: guniq.setdefault(e["text"].lower().strip(), e)
        ttexts = [tt for t in triples for tt in _get_triple_texts(t)]

        matched = set()
        for k, g in guniq.items():
            if any(_match_score(tt, g["text"]) >= 0.4 for tt in ttexts):
                matched.add(k)

        miss_label, miss_bucket, subst = Counter(), Counter(), []
        for k, g in guniq.items():
            agg_label[g["label"]][0] += 1
            if k in matched:
                agg_label[g["label"]][1] += 1
            else:
                b = bucketize(g["text"]); miss_bucket[b] += 1; agg_bucket[b] += 1
                miss_label[g["label"]] += 1
                if b == "substantive":
                    subst.append(g["text"]); recurring[re.sub(r'\s+', ' ', g["text"].lower().strip())] += 1
        per_patient.append({
            "patient": pid, "cohort": d.cohort, "gold_unique": len(guniq),
            "matched": len(matched), "recall": round(len(matched) / max(len(guniq), 1), 3),
            "missed": len(guniq) - len(matched), "missed_by_label": dict(miss_label),
            "missed_by_bucket": dict(miss_bucket), "substantive_missed": sorted(subst),
        })

    recall_by_label = {lab: {"gold": g, "matched": m, "recall": round(m / max(g, 1), 3)}
                       for lab, (g, m) in sorted(agg_label.items(), key=lambda x: -x[1][0])}
    recurring_top = [[t, c] for t, c in recurring.most_common(50) if c >= 2]
    report = {"extraction_dir": extraction_dir, "n_patients": len(per_patient),
              "recall_by_label": recall_by_label, "missed_by_bucket_total": dict(agg_bucket),
              "recurring_substantive_misses": recurring_top, "per_patient": per_patient}
    json.dump(report, open(out_json, "w"), indent=2)

    tot = sum(agg_bucket.values()) or 1
    with open(out_txt, "w") as w:
        w.write(f"MISS ANALYSIS — {extraction_dir}  ({len(per_patient)} patients)\n" + "=" * 64 + "\n\n")
        w.write("RECALL BY GOLD LABEL (lowest = worst-captured entity type):\n")
        for lab, v in sorted(recall_by_label.items(), key=lambda x: x[1]["recall"]):
            w.write(f"  {lab:18s} recall={v['recall']:.3f}  ({v['matched']}/{v['gold']})\n")
        w.write("\nMISSED BY BUCKET (ceiling: anaphora/lab_fragment/section_header vs RECOVERABLE: substantive):\n")
        for b, c in agg_bucket.most_common():
            w.write(f"  {b:16s} {c:4d}  ({100*c/tot:.0f}%)\n")
        w.write("\nRECURRING SUBSTANTIVE MISSES (missed in >=2 patients -> FINE-TUNING TARGETS):\n")
        for t, c in recurring_top:
            w.write(f"  x{c:<2d} {t}\n")
        w.write("\nPER-PATIENT:\n")
        for p in per_patient:
            w.write(f"  {p['patient']:8s} recall={p['recall']:.3f} missed={p['missed']:3d}  by_label={p['missed_by_label']}\n")
    print(f"wrote {out_json} and {out_txt}  ({len(per_patient)} patients)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraction", default="results/extraction/ens3/union")
    ap.add_argument("--out", default="results/miss_report")
    a = ap.parse_args()
    analyze(a.extraction, a.out + ".json", a.out + ".txt")
