#!/usr/bin/env python
"""Survey + categorize extraction failure modes in the MIMIC triples.

Produces the failure taxonomy behind the veracity/normalization discussion:
which error types the zero-shot ensemble makes, at what frequency, with examples.
Categories overlap (one triple can hit several), so shares sum to >100%.

Two families:
  * STRUCTURAL (invalid fhir_type / degenerate / vacuous entity) -> a deterministic
    normalization stage fixes these; the underlying fact is usually fine.
  * SEMANTIC / PHI -> the trust gate demotes wrong-relation triples; PHI markers
    should be dropped outright (compliance).

Usage: python scripts/mimic_failure_survey.py [notes_per_model]
"""
import glob
import json
import re
import sys
from collections import Counter, defaultdict

N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
VALID_FHIR = {"patient", "condition", "observation", "procedure", "medicationstatement",
              "medicationrequest", "medication", "medicationorder", "allergyintolerance",
              "familymemberhistory", "diagnosticreport", "careplan", "encounter",
              "immunization", "specimen", "bodystructure", "deviceusestatement",
              "device", "servicerequest"}
GENERIC = {"multiple", "year", "years", "courses", "course", "myself", "colleagues",
           "colleague", "none", "unknown", "patient", "several", "various", "other",
           "yes", "no", "normal", "stable", "day", "days"}


def norm(s):
    if isinstance(s, (list, tuple)):
        s = " ".join(map(str, s))
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def load_notes(coh):
    d = {}
    for line in open(f"data/mimic_oncology/{coh}/notes_all.jsonl"):
        r = json.loads(line)
        d[str(r.get("note_id"))] = norm(r.get("text", ""))
    return d


def main():
    cats, ex, total = Counter(), defaultdict(list), 0
    for coh in ("mimiciii", "mimiciv"):
        notes = load_notes(coh)
        for model in ("gemma4-e4b", "llama32-3b", "qwen3-4b", "medgemma-4b"):
            for f in sorted(glob.glob(f"results/extraction/mimic_{coh}/bymodel/{model}/*.json"))[:N]:
                o = json.load(open(f))
                nt = notes.get(o["id"], "")
                for t in o.get("triples", []):
                    if not isinstance(t, dict):
                        continue
                    total += 1
                    e, v = str(t.get("entity", "")), str(t.get("value", ""))
                    ft, ev = norm(t.get("fhir_type", "")), norm(t.get("evidence_span") or t.get("value"))

                    def rec(c, s):
                        cats[c] += 1
                        if len(ex[c]) < 3:
                            ex[c].append(s)
                    if norm(e) == norm(v) and e:
                        rec("degenerate (entity==value)", f"{e!r} =={t.get('attribute')}=> {v!r}")
                    if re.search(r"\[|\*\*\*\*\*|known (lastname|firstname)", (e + v).lower()):
                        rec("PHI placeholder leak", repr(e))
                    if ft and ft not in VALID_FHIR:
                        rec("invalid FHIR type", f"fhir_type={t.get('fhir_type')!r} ({e!r})")
                    if norm(e) in GENERIC or (len(norm(e)) <= 3 and not norm(e).isupper()):
                        rec("vacuous/generic entity", f"{e!r} --{t.get('attribute')}--> {v!r}")
                    if ev and len(ev) >= 5 and ev not in nt:
                        rec("ungrounded (evidence not in note)", f"{e!r} span={ev[:40]!r}")
    print(f"scanned {total} triples ({N} notes/model/cohort). categories overlap.\n")
    for c, n in cats.most_common():
        print(f"{n:7d}  ({100 * n / total:4.1f}%)  {c}")
        for s in ex[c]:
            print(f"              e.g. {s}")


if __name__ == "__main__":
    main()
