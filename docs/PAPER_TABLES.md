# TRUST-KG — Paper Tables & Results (canonical numbers for the draft)

**Single source of truth for every number in the IEEE BigData 2026 draft.** Each table below
mirrors the draft's *exact* structure (same columns, same rows). Update Overleaf from the ✅ cells.

Legend:
- ✅ **verified** — reproduced in this repo, frozen (seeded where stochastic); generating script named.
- ⛔ **pending** — needs an experiment we have not run yet (named below).
- ⚠️ **in draft but not backed here** — a value currently in the draft that this repo cannot
  reproduce/verify, or that conflicts with our verified numbers. Needs a decision or a run.

All numbers are entity-level unless noted. CORAL is reported **per cohort** (PDAC = ids 0–19,
BRCA = ids 20–39); we never pool the two cohorts for extraction metrics.

---

## 0. Framework & contribution (what the draft must convey)

TRUST-KG is a **construction-time, calibrated, selective admission-control framework** for building
knowledge graphs from heterogeneous clinical Big Data. It **separates candidate generation from fact
admission**: an LLM proposes EAV candidates, each candidate is scored against source / semantic /
ontology / schema / temporal / graph-context evidence, deterministic violations are rejected, and the
remainder receive a **learned, calibrated reliability** estimate driving an **Insert–Review–Reject**
policy — *before* anything is materialized to the graph. The contribution is the **domain-general
admission-control formulation** (governs any LLM→KG ingestion), not a clinical extractor.

**Four contributions (draft §I):** (1) selective Big-Data ingestion *formulation*; (2) evidence-grounded
reliability + calibration; (3) selective KG admission (Insert–Review–Reject); (4) heterogeneous clinical
Big-Data evaluation.

**Four research questions (draft §V):**
| RQ | Question | Evidence table(s) |
|---|---|---|
| RQ1 | Does heterogeneous evidence improve extraction + construction-time verification? | V, II |
| RQ2 | Does learned+calibrated reliability beat heuristic trust? | I |
| RQ3 | Can calibrated selective admission control coverage / risk / review workload? | I |
| RQ4 | Does TRUST-KG stay computationally practical as the corpus grows? | VII, VI |

---

## 1. Table I — Reliability calibration & selective KG admission (held-out CORAL) ✅ COMPLETE

Columns: `Method | ECE | Brier | NLL | AURC | Cov.@95% | Insert | Review | Reject` (lower ECE/Brier/NLL/AURC
better; higher Cov.@95% better; thresholds chosen independently on dev at the same ≥95% precision target).

| Method | ECE | Brier | NLL | AURC | Cov.@95% | Insert | Review | Reject |
|---|---|---|---|---|---|---|---|---|
| Heuristic Trust | 0.141 | 0.119 | 0.406 | 0.094 | 0.003 | 21.5% | 78.4% | 0.1% |
| **Learned Reliability** | **0.036** | **0.091** | **0.311** | **0.045** | **0.595** | **60.1%** | 35.9% | 4.0% |
| Learned + Calibration | 0.041 | 0.094 | 0.320 | 0.045 | 0.595 | 60.1% | 35.9% | 4.0% |

- Insert-precision at the operating point = **0.948** (this is the "94.8%" the prose cites; it is *not* a Table I column).
- Post-hoc calibration is a verified **no-op** (learned scores already well-calibrated; ECE 0.036→0.041). Keep the row for completeness or fold to two rows — your call.
- **Script:** `scripts/exp_calibration_selective.py` → `results/e1e3_results.json`. Seeded (`random_state=0`), frozen: two runs identical.
- Test set: 3,569 held-out candidate triples; overall correct-rate 0.872.

---

## 2. Table II — Entity-level extraction on CORAL ⚠️ PARTIAL (conflict + pending baseline)

Columns: `Dataset | Method | Precision | Recall | F1`.

| Dataset | Method | Precision | Recall | F1 | Status |
|---|---|---|---|---|---|
| CORAL-BRCA | Vanilla RAG | — | — | — | ⛔ RAG baseline not run |
| CORAL-BRCA | Gemma 2-pass | 0.969 | 0.879 | 0.922 | ⚠️ **draft value — not reproduced here** |
| CORAL-BRCA | **Ensemble** | **0.850** | **0.890** | **0.868** | ✅ verified (ens3) |
| CORAL-PDAC | Vanilla RAG | — | — | — | ⛔ RAG baseline not run |
| CORAL-PDAC | Gemma 2-pass | 0.956 | 0.890 | 0.922 | ⚠️ **draft value — not reproduced here** |
| CORAL-PDAC | **Ensemble** | **0.888** | **0.870** | **0.877** | ✅ verified (ens3) |

**⚠️ Reconciliation needed (affects the abstract F1 claim and the "primary configuration" framing):**
- The **Ensemble** rows are verified full-cohort (20+20 patients, frozen): BRCA 0.868 F1, PDAC 0.877 F1
  (`results/ens3_metrics_gpu{0,1}.json`, `scripts/run_coral_ensemble.py`). sd: BRCA ±0.045, PDAC ±0.043.
- The **Gemma 2-pass** rows already in the draft (0.969/0.879/**0.922**, 0.956/0.890/**0.922**) are **not
  reproduced anywhere in this repo** — they appear to be from the earlier (lost) runs. Our only current
  Gemma-2-pass data is a **2-patient smoke** (pdac_0 F1 0.833, brca_20 F1 0.861), not full-cohort. The two
  cohorts also share an identical 0.922 F1, which is unusual for real per-cohort numbers.
- **Consequence:** as written, draft Table II shows Gemma-2-pass F1 (0.922) *above* our verified Ensemble F1
  (0.868/0.877) — the ensemble trades precision for recall and nets lower F1. So either the "Ensemble is the
  primary config" story or the numbers need to align.
- **To resolve:** run **full-cohort Gemma-2-pass** (single model, 40 patients — cheap) to verify or replace
  the 0.922 rows, and decide which configuration is "primary." Until then, the abstract's
  "F1 … to X on BRCA / X on PDAC" should cite whichever config Table II designates as primary.

---

## 3. Table III — Construction-time validation on CORAL ⛔ COMPUTABLE, not yet aggregated

Columns: `Validation Dimension | Result`.

| Validation Dimension | Result |
|---|---|
| Source grounding | ⛔ |
| Ontology compatibility | ⛔ |
| Schema validity | ⛔ |
| Temporal consistency | ⛔ |
| Contradiction control | ⛔ |

- The five validation layers already exist (`src/extraction/validation.py`, layers 1–5) and are applied to
  every ensemble triple. This table is just a **per-dimension summary** (e.g. mean pass-rate / mean score
  over the 21,106 CORAL candidate triples) that we have **not yet written out** as a standalone result.
- **To fill:** a short aggregation over the same validation output E1–E2 already computes (no new model runs).

---

## 4. Table IV — Source-grounding diagnostic on MIMIC ⛔ PENDING (MIMIC run)

Columns: `Dataset | URR Before | URR After | Recall Impact` (URR = unsupported-retained-assertion rate).

| Dataset | URR Before | URR After | Recall Impact |
|---|---|---|---|
| MIMIC-III | ⛔ | ⛔ | ⛔ |
| MIMIC-IV | ⛔ | ⛔ | ⛔ |

- Needs the **MIMIC oncology extraction** (input notes present: `data/mimic_oncology/{mimiciii,mimiciv}/notes_all.jsonl`,
  ~400 notes each). Extraction is **stalled** (see §9). `scripts/run_mimic_fast.py` is the runner.

---

## 5. Table V — Heterogeneous evidence evaluation on CORAL ⚠️ (Panel A unverified) + ⛔ (Panel B)

**Panel A — evidence retrieval.** Columns: `Method | R@5 | R@10 | MRR | nDCG | Evid. P`.

| Method | R@5 | R@10 | MRR | nDCG | Evid. P | Status |
|---|---|---|---|---|---|---|
| BM25 | 0.209 | 0.209 | 0.207 | 0.992 | 0.695 | ⚠️ draft value, no backing script |
| MedCPT | 0.996 | 1.000 | 0.874 | 0.897 | 0.188 | ⚠️ |
| Graph | 0.396 | 0.396 | 0.396 | 0.870 | 1.000 | ⚠️ |
| Hybrid | 0.996 | 0.996 | 0.876 | 0.897 | 0.299 | ⚠️ |

- ⚠️ Retrieval (BM25 + MedCPT + graph-neighborhood) is implemented in the pipeline, but **there is no
  evaluation harness in this repo that emits R@k / MRR / nDCG / evidence-precision** — these Panel-A numbers
  **cannot be reproduced or verified here**. Provenance likely the earlier runs. Needs a retrieval-eval script.

**Panel B — reliability evidence ablation.** Columns: `Configuration | ECE | AURC`.

| Configuration | ECE | AURC |
|---|---|---|
| Extraction only | ⛔ |
| +E_src | ⛔ |
| +E_sem | ⛔ |
| +E_ont | ⛔ |
| +E_temp | ⛔ |
| +E_graph | ⛔ |
| +Calibration | ⛔ |

- ⛔ This is the **evidence-feature ablation** (add one evidence group at a time to the reliability learner,
  report ECE/AURC). It reuses the E1–E2 labeled set (`results/e1e2_labeled.json`) with **feature subsets** —
  no new model runs. Not yet scripted.

---

## 6. Table VI — KG construction scale & structured queryability ⚠️ PARTIAL (CORAL ✅, MIMIC ⛔)

Columns: `Dataset | KG Triples | Entities | Query Success` (combined entity counts are *after* cross-dataset
canonicalization/dedup, so they need not equal the per-dataset sum).

| Dataset | KG Triples | Entities | Query Success | Status |
|---|---|---|---|---|
| CORAL | 81,705 | 7,423 | 10/10 | ✅ verified (sum of cohorts; see note) |
| MIMIC-III | ⛔ | ⛔ | ⛔ | ⛔ MIMIC run |
| MIMIC-IV | ⛔ | ⛔ | ⛔ | ⛔ MIMIC run |

- CORAL per cohort (`results/coral_graph_report.json`, `scripts/run_coral_graph.py`): **PDAC** 40,412 triples /
  3,601 entities / 10-of-10 SPARQL; **BRCA** 41,293 / 3,822 / 10-of-10. The CORAL row above sums them
  (81,705 / 7,423) as an **upper bound** — the true dedup'd combined entity count may be lower if cross-cohort
  concept labels canonicalize together. Run the combined-CORAL graph build to get the exact dedup number.
- The full "Across 840 records" abstract totals require the MIMIC rows (840 = 40 CORAL + 800 MIMIC).

---

## 7. Table VII — Corpus-fraction scalability ⛔ PENDING (MIMIC throughput)

Columns: `Corpus | Triples | Throughput | Verify Lat. | Cost`.

| Corpus | Triples | Throughput | Verify Lat. | Cost |
|---|---|---|---|---|
| 25% | ⛔ | ⛔ | ⛔ | ⛔ |
| 50% | ⛔ | ⛔ | ⛔ | ⛔ |
| 75% | ⛔ | ⛔ | ⛔ | ⛔ |
| 100% | ⛔ | ⛔ | ⛔ | ⛔ |

- Needs the **throughput-optimized MIMIC run** at increasing corpus fractions (resident model + batched
  inference, `scripts/run_mimic_fast.py`). This is the Volume/RQ4 evidence. Stalled with Table IV.

---

## 8. Abstract / intro claims → value (the `[GATED]` map)

The draft prose blanks unverified numbers with `[GATED]`. Fillable now vs. blocked:

| Location / phrase | Value | Status |
|---|---|---|
| `[GATED-CONFIG]` improves F1 … | "the two-pass ensemble" (⚠️ but see Table II config decision) | conditional |
| … from `[GATED]` for the RAG baseline | — | ⛔ RAG baseline (Table II) |
| … to `[GATED]` on CORAL-BRCA | 0.868 (Ensemble) *or* draft's Gemma-2pass 0.922 — per Table II decision | ⚠️ |
| achieves `[GATED]` on CORAL-PDAC | 0.877 (Ensemble) *or* 0.922 — per Table II decision | ⚠️ |
| mean F1 `XX±XX` BRCA / PDAC (robustness) | 0.868±0.045 / 0.877±0.043 (Ensemble) | ✅ (config-dependent) |
| inserts `[GATED]` / routes `[GATED]` / rejects `[GATED]` | 60.1% / 35.9% / 4.0% | ✅ |
| … AURC is `[GATED]` | 0.045 | ✅ |
| MIMIC unsupported-retained-assertion `[GATED]`→`[GATED]`, ~`[GATED]` recall impact | — | ⛔ Table IV |
| across 840 records: `[GATED]` triples / `[GATED]` entities / `[GATED]` RDF triples | — (CORAL-only: 81,705 / 7,423) | ⛔ Table VI needs MIMIC |
| `[GATED FIGURE: RISK–COVERAGE]` (Fig. 2) | data ready (E1–E2); PNG not rendered | can generate on request |
| "no credential-gated MIMIC … content" (ethics) | *not a placeholder — ordinary word, leave as-is* | n/a |

---

## 9. What's needed to finish (prioritized)

1. **MIMIC oncology extraction** — unlocks Tables **IV, VII**, the MIMIC rows of **VI**, and the 840-record
   abstract totals. Data is present; the run is **stalled** (queue-watcher died in the cluster restart and
   `resume_all.sh` only resumed CORAL). Needs a manual kick: `scripts/run_mimic_fast.py` on both GPUs.
2. **Full-cohort Gemma-2-pass** (cheap) — verify/replace Table II's 0.922 rows and settle the "primary
   configuration" question (affects the abstract F1 claim).
3. **Vanilla-RAG baseline** on CORAL — the "from `[GATED]`" F1 in Table II / abstract.
4. **Retrieval-eval harness** — reproduce/verify Table V Panel A (R@k/MRR/nDCG), currently unbacked.
5. **Two no-new-run aggregations** — Table III (validation-dimension summary) and Table V Panel B (evidence
   ablation), both from the existing E1–E2 labeled set.

Non-CORAL work is items 1 & (part of) VI/VII; everything else is CORAL-based and already has its inputs.
