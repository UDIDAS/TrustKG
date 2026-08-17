#!/usr/bin/env python
"""Per-model extraction-quality assessment for the MIMIC ensemble run.

For each (cohort, model) reports: notes cached, empty files, triples/note,
evidence-grounding (% of triples whose evidence_span appears verbatim in the
source note — a conservative anti-hallucination check), and unique-entity
diversity. This is the "division of labour" evidence in the README
(ensemble -> recall, trust -> precision).

Usage: python scripts/mimic_model_quality.py [sample_per_model]
"""
import glob
import json
import re
import statistics as st
import sys
from collections import Counter

COHORTS = ("mimiciii", "mimiciv")
MODELS = ("gemma4-e4b", "llama32-3b", "qwen3-4b", "medgemma-4b")
SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 150


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


def assess(coh, model, notes, sample):
    files = sorted(glob.glob(f"results/extraction/mimic_{coh}/bymodel/{model}/*.json"))
    if not files:
        return None
    per, gtot, gok, ents = [], 0, 0, Counter()
    for f in files[:sample]:
        o = json.load(open(f))
        tr = o.get("triples", [])
        per.append(len(tr))
        nt = notes.get(o["id"], "")
        for t in tr:
            if not isinstance(t, dict):
                continue
            ents[str(t.get("entity", ""))[:30]] += 1
            ev = norm(t.get("evidence_span") or t.get("value") or "")
            if ev and len(ev) >= 4:
                gtot += 1
                gok += (ev in nt)
    empty = sum(1 for f in files if not json.load(open(f)).get("triples"))
    return dict(files=len(files), empty=empty, mean=st.mean(per) if per else 0,
                grounded=100 * gok / gtot if gtot else 0, uniq=len(ents))


def main():
    print(f"{'cohort/model':28s} {'files':>5} {'empty':>5} {'t/note':>6} {'grnd%':>6} {'uniqEnt':>8}")
    for coh in COHORTS:
        notes = load_notes(coh)
        for m in MODELS:
            r = assess(coh, m, notes, SAMPLE)
            if r:
                print(f"{coh + '/' + m:28s} {r['files']:>5} {r['empty']:>5} "
                      f"{r['mean']:>6.1f} {r['grounded']:>6.1f} {r['uniq']:>8}")


if __name__ == "__main__":
    main()
