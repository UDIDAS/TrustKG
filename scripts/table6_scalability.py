"""Table VI — Corpus-Fraction Scalability on MIMIC (25 / 50 / 75 / 100%).

Reports, at each corpus fraction of the 800-note MIMIC oncology corpus:
  * Number of triples        — REAL, from the ensemble union (KG growth)
  * Throughput (notes/hr)     — measured vLLM ensemble extraction rate (see THROUGHPUT below)
  * Verification latency       — MEASURED fresh here: mean wall-time of the 5-layer
                                 validation per note (extrapolated linearly over the fraction)
  * Computational cost         — derived GPU-hours from the measured throughput

Honesty note: triple counts and verification latency are measured directly; the extraction
throughput is the rate measured during the run (Phase A per-cohort logs: gemma-4 2-pass
173.7 / 160.2 notes/hr; 1-pass augmenters ~=320 notes/hr), from which the ensemble rate and
cost are derived. For a single-source instrumented run, re-run extraction with per-fraction
timing (a few GPU-hours) — this reconstructs the same quantities from the completed run.

    python scripts/table6_scalability.py
"""
from __future__ import annotations
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.extraction.validation import validate_patient_triples

# measured extraction rates (notes/hr, single A6000, from the run's Phase A logs)
ANCHOR_2PASS = (173.7 + 160.2) / 2      # gemma-4-E4B 2-pass anchor
AUGMENTER_1PASS = 320.0                  # llama/qwen/medgemma 1-pass (≈2x anchor)
# per-note ensemble time = anchor + 3 augmenters (sequential per GPU)
_sec_per_note = 3600 / ANCHOR_2PASS + 3 * (3600 / AUGMENTER_1PASS)
ENSEMBLE_NPH_1GPU = 3600 / _sec_per_note
N_GPUS = 2                               # cohort-split (mimiciii/GPU0, mimiciv/GPU1)
ENSEMBLE_NPH = ENSEMBLE_NPH_1GPU * N_GPUS
FRACTIONS = [0.25, 0.50, 0.75, 1.00]


def load_corpus():
    notes = {}
    for coh in ("mimiciii", "mimiciv"):
        for line in open(f"data/mimic_oncology/{coh}/notes_all.jsonl"):
            r = json.loads(line)
            notes[str(r.get("note_id"))] = r.get("text", "")
    items = []
    for coh in ("mimiciii", "mimiciv"):
        for f in sorted(glob.glob(f"results/extraction/mimic_{coh}/union/*.json")):
            d = json.load(open(f))
            nid = str(d.get("id") or Path(f).stem)
            items.append((nid, [t for t in d.get("triples", []) if isinstance(t, dict)],
                          notes.get(nid, "")))
    return items


def measure_verify_latency(items, sample=40):
    step = max(1, len(items) // sample)
    picked = items[::step][:sample]
    t0 = time.time()
    ntr = 0
    for nid, triples, text in picked:
        validate_patient_triples(triples, text, trust_threshold=0.4)
        ntr += len(triples)
    dt = time.time() - t0
    return dt / len(picked), dt / max(ntr, 1)     # sec/note, sec/triple


def main():
    items = load_corpus()
    N = len(items)
    print(f"MIMIC corpus: {N} notes (mimiciii + mimiciv unions)")
    if N < 800:
        print(f"  WARNING: only {N}/800 union notes present — iv union may still be finishing.")
    sec_per_note, sec_per_triple = measure_verify_latency(items)

    cum_tri = []
    s = 0
    for _, triples, _ in items:
        s += len(triples)
        cum_tri.append(s)

    rows = []
    for fr in FRACTIONS:
        n = int(round(fr * N))
        triples = cum_tri[n - 1] if n else 0
        extract_h = n / ENSEMBLE_NPH
        gpu_h = n / ENSEMBLE_NPH_1GPU        # single-GPU-equivalent GPU-hours
        verify_s = sec_per_note * n
        rows.append({
            "fraction": f"{int(fr*100)}%", "notes": n, "triples": triples,
            "throughput_notes_hr": round(ENSEMBLE_NPH, 1),
            "verification_latency_ms_per_note": round(sec_per_note * 1000, 1),
            "verification_total_min": round(verify_s / 60, 1),
            "extraction_wallclock_h": round(extract_h, 2),
            "compute_cost_gpu_h": round(gpu_h, 2),
        })

    report = {
        "corpus_notes": N,
        "throughput_notes_hr_ensemble_2gpu": round(ENSEMBLE_NPH, 1),
        "throughput_notes_hr_per_gpu": round(ENSEMBLE_NPH_1GPU, 1),
        "verification_latency_ms_per_note": round(sec_per_note * 1000, 1),
        "verification_latency_ms_per_triple": round(sec_per_triple * 1000, 2),
        "fractions": rows,
    }
    json.dump(report, open("results/table6_scalability.json", "w"), indent=2)

    print("=" * 82)
    print("TABLE VI — Corpus-Fraction Scalability on MIMIC")
    print("=" * 82)
    print(f"{'Fraction':>8} {'Notes':>6} {'Triples':>9} {'Throughput':>11} "
          f"{'Verif.lat':>10} {'Extract':>9} {'Cost':>9}")
    print(f"{'':>8} {'':>6} {'':>9} {'(notes/hr)':>11} {'(ms/note)':>10} {'(h)':>9} {'(GPU-h)':>9}")
    for r in rows:
        print(f"{r['fraction']:>8} {r['notes']:>6} {r['triples']:>9} "
              f"{r['throughput_notes_hr']:>11} {r['verification_latency_ms_per_note']:>10} "
              f"{r['extraction_wallclock_h']:>9} {r['compute_cost_gpu_h']:>9}")
    print(f"\nverification: {report['verification_latency_ms_per_note']} ms/note "
          f"({report['verification_latency_ms_per_triple']} ms/triple)")
    print("Saved results/table6_scalability.json")


if __name__ == "__main__":
    main()
