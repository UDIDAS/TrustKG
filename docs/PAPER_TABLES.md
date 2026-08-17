# TRUST-KG — Paper Tables & Results (canonical numbers for the draft)

**Single source of truth for every number in the IEEE BigData 2026 draft.** Each table below
mirrors the draft's *exact* structure (same columns, same rows). Update Overleaf from the ✅ cells.

Legend:
- ✅ **verified** — reproduced in this repo, frozen (seeded where stochastic); generating script named.
- ⛔ **pending** — needs an experiment we have not run yet (named below).
- ⚠️ **draft has a stale pre-fill — ignore it** — the current PDF shows a number in this cell that did
  **not** come from our current experiments (earlier/lost runs, different protocol). Treat as pending and
  regenerate.

> **Every number in the paper comes from our own experiments.** The current PDF has some cells pre-filled
> (e.g. Table II "Gemma 2-pass" = 0.922, all of Table V's retrieval numbers) from earlier/lost runs — these
> are **not authoritative; ignore them** and regenerate every value from a run in this repo.

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
| Heuristic Trust | 0.172 | 0.082 | 0.315 | 0.031 | 0.981 | 100% | 0% | 0% |
| **Learned Reliability** | **0.008** | **0.046** | **0.181** | **0.019** | **0.992** | 67% | 31% | 2% |
| Learned + Calibration | 0.014 | 0.048 | 0.190 | 0.019 | 0.992 | 67% | 31% | 2% |

Insert/Review/Reject shown at the **99%-precision operating point** (achieving 98.1% insert-precision) — the point
where the gate does visible work. Because the Gemma-4 ensemble is already ≈95% precise, at a 95% target the policy
inserts ≈100% (nothing to reject); the **tunable operating-point curve** is the real story:

| Target precision | Insert | Review | Reject | Achieved precision |
|---|---|---|---|---|
| 95% | 100% | 0% | 2% | 0.949 |
| 98% | 86% | 12% | 2% | 0.967 |
| 99% | 67% | 31% | 2% | 0.981 |

- Verified exact on the new Gemma-4 ensemble (`results/e1e3_results.json`, seeded/reproducible). Headline calibration
  win: Learned ECE **0.008** vs Heuristic 0.172. Post-hoc calibration is a small no-op (0.008→0.014 ECE).
- **Script:** `scripts/exp_calibration_selective.py` → `results/e1e3_results.json`. Seeded (`random_state=0`), frozen: two runs identical.
- Test set: 3,569 held-out candidate triples; overall correct-rate 0.872.

---

## 2. Table II — Entity-level extraction on CORAL ✅ Ensemble verified · ⛔ baseline rows to run

Columns: `Dataset | Method | Precision | Recall | F1`. Configuration-comparison table: **Vanilla RAG → Gemma-4
anchor (2-pass) → Ensemble**. The extractor was chosen by a **full extractor-comparison sweep** over all model
unions on CORAL (`scripts/combo_eval.py`); all candidates ≤5B. Winner = **Gemma-4-E4B 2-pass anchor ∪
single-pass Llama-3.2-3B / Qwen3-4B / MedGemma-4B**. Supports RQ1.

| Dataset | Method | Precision | Recall | F1 | Status |
|---|---|---|---|---|---|
| CORAL-BRCA | Vanilla RAG | — | — | — | ⛔ run baseline |
| CORAL-BRCA | Gemma-4 anchor (2-pass, solo) | 0.972 | 0.770 | 0.854 | ✅ (exact) |
| CORAL-BRCA | **Ensemble** (primary) | **0.940** | **0.848** | **0.890 ± 0.044** | ✅ verified (coral_final) |
| CORAL-PDAC | Vanilla RAG | — | — | — | ⛔ run baseline |
| CORAL-PDAC | Gemma-4 anchor (2-pass, solo) | 0.983 | 0.721 | 0.829 | ✅ (exact) |
| CORAL-PDAC | **Ensemble** (primary) | **0.958** | **0.815** | **0.879 ± 0.048** | ✅ verified (coral_final) |

- **Ensemble** rows verified full-cohort, exact scorer — `results/coral_final_score.json`, config = Gemma-4-E4B
  2-pass (`scripts/run_ensemble_fast.py --twopass gemma4-e4b --seed-from combo_coral`) ∪ cached 1-pass augmenters,
  scored by `scripts/fast_score.py`. CI: BRCA [0.871, 0.909], PDAC [0.858, 0.900].
- **This all-≤5B ensemble matches/beats the old Gemma-3 + Qwen-8B ensemble** (0.868 / 0.877) at precision ~0.95 —
  the ≤5B constraint cost nothing. Gemma-4-E4B is the best anchor (solo F1 0.71 vs Gemma-3-4B 0.58); Gemma-3
  dropped as redundant; Qwen3-8B dropped (throughput); Phi-4-mini excluded (transformers 5.8 incompat).
- The **2-pass anchor** lifts recall: Gemma-4 solo 2-pass R≈0.72–0.77 → full ensemble R 0.815–0.848 (augmenters add
  recall); and 1-pass ensemble ~0.73 → 2-pass-anchor ensemble 0.83. All rows above are exact-scored (`fast_score.py`).
- **Vanilla RAG** baseline still needs a full-cohort run (feeds the abstract's "F1 from X for the RAG baseline").

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
| BM25 | — | — | — | — | — | ⛔ our retrieval-eval |
| MedCPT | — | — | — | — | — | ⛔ |
| Graph | — | — | — | — | — | ⛔ |
| Hybrid | — | — | — | — | — | ⛔ |

- ⛔ Retrieval (BM25 + MedCPT + graph-neighborhood) is implemented in the pipeline, but **there is no
  evaluation harness that emits R@k / MRR / nDCG / evidence-precision.** **Ignore the draft's pre-filled
  numbers** (0.209 / 0.996 / …, earlier runs) — build a retrieval-eval script (query = gold entity, corpus =
  note chunks; score each retriever + the hybrid fusion) to generate this panel ourselves. Supports RQ1.

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
| `[GATED-CONFIG]` improves F1 … | "the ensemble" (our primary config) | ✅ |
| … from `[GATED]` for the RAG baseline | — | ⛔ RAG baseline (Table II) |
| … to `[GATED]` on CORAL-BRCA | 0.890 (Ensemble, Gemma-4 sub-5B) | ✅ |
| achieves `[GATED]` on CORAL-PDAC | 0.879 (Ensemble, Gemma-4 sub-5B) | ✅ |
| mean F1 `XX±XX` BRCA / PDAC (robustness) | 0.890±0.044 / 0.879±0.048 (Ensemble) | ✅ |
| inserts `[GATED]` / routes `[GATED]` / rejects `[GATED]` | 67% / 31% / 2% @99% target (or 100/0/2 @95%) | ✅ |
| … AURC is `[GATED]` | 0.019 (learned) | ✅ |
| MIMIC unsupported-retained-assertion `[GATED]`→`[GATED]`, ~`[GATED]` recall impact | — | ⛔ Table IV |
| across 840 records: `[GATED]` triples / `[GATED]` entities / `[GATED]` RDF triples | — (CORAL-only: 81,705 / 7,423) | ⛔ Table VI needs MIMIC |
| `[GATED FIGURE: RISK–COVERAGE]` (Fig. 2) | data ready (E1–E2); PNG not rendered | can generate on request |
| "no credential-gated MIMIC … content" (ethics) | *not a placeholder — ordinary word, leave as-is* | n/a |

---

## 9. What's needed to finish (prioritized)

1. **MIMIC oncology extraction** — unlocks Tables **IV, VII**, the MIMIC rows of **VI**, and the 840-record
   abstract totals. Data is present; the run is **stalled** (queue-watcher died in the cluster restart and
   `resume_all.sh` only resumed CORAL). Needs a manual kick: `scripts/run_mimic_fast.py` on both GPUs.
2. **Full-cohort Gemma-2-pass** (cheap) — fills Table II's Gemma-2-pass comparison row (in *addition* to the
   ensemble, which stays primary). `run_coral_ensemble.py --models gemma3-4b --twopass gemma3-4b`.
3. **Vanilla-RAG baseline** on CORAL — Table II's baseline row + the abstract's "F1 from `[GATED]`".
4. **Retrieval-eval harness** — generate Table V Panel A (R@k/MRR/nDCG) ourselves; ignore the draft pre-fills.
5. **Two no-new-run aggregations** — Table III (validation-dimension summary) and Table V Panel B (evidence
   ablation), both from the existing E1–E2 labeled set.

Non-CORAL work is items 1 & (part of) VI/VII; everything else is CORAL-based and already has its inputs.
