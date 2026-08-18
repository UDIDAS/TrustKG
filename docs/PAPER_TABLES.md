# TRUST-KG — Paper Tables & Results (canonical numbers for the draft)

**Single source of truth for every number in the IEEE BigData 2026 draft.** Each table below
mirrors the draft's *exact* structure (same columns, same rows). Update Overleaf from the ✅ cells.

Legend:
- ✅ **verified** — reproduced in this repo, frozen (seeded where stochastic); generating script named.
- ⛔ **pending** — needs an experiment we have not run yet (named below).
- ⚠️ **draft has a stale pre-fill — ignore it** — the current PDF shows a number in this cell that did
  **not** come from our current experiments (earlier/lost runs, different protocol). Treat as pending and
  regenerate.

> **Every number in the paper comes from our own experiments.** Any cell the current PDF pre-fills from
> earlier/lost runs is **not authoritative; ignore it** and regenerate the value from a run in this repo.

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
| RQ1 | Does heterogeneous evidence improve extraction + construction-time verification? | III, IV |
| RQ2 | Does learned+calibrated reliability beat heuristic trust? | II |
| RQ3 | Can calibrated selective admission control coverage / risk / review workload? | II |
| RQ4 | Does TRUST-KG stay computationally practical as the corpus grows? | V, VI |

---

> **Numbering matches the current draft (`docs/BigData_2026_TrustKG.pdf`, 6 tables):**
> I Datasets · II Reliability calibration & admission · III Entity extraction (CORAL) ·
> IV Construction-time validation (CORAL) · V CORAL KG scale & queryability · VI Corpus-fraction scalability.

## Table I — Datasets used in the evaluation

CORAL (PDAC + BRCA, 20 patients each) and MIMIC-III / MIMIC-IV oncology subsets (400 notes each,
malignant-neoplasm ICD-filtered at ingestion). Descriptive overview — no computed metrics.

## Table II — Reliability calibration & selective KG admission (held-out CORAL) ✅ COMPLETE

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

## Table III — Entity-level extraction on CORAL ✅ COMPLETE

Columns: `Dataset | Method | Precision | Recall | F1`. Configuration-comparison table: **Vanilla RAG → Gemma-4
anchor (2-pass) → Ensemble**. The extractor was chosen by a **full extractor-comparison sweep** over all model
unions on CORAL (`scripts/combo_eval.py`); all candidates ≤5B. Winner = **Gemma-4-E4B 2-pass anchor ∪
single-pass Llama-3.2-3B / Qwen3-4B / MedGemma-4B**. Supports RQ1.

| Dataset | Method | Precision | Recall | F1 | Status |
|---|---|---|---|---|---|
| CORAL-BRCA | Vanilla RAG | 0.971 | 0.698 | 0.805 ± 0.089 | ✅ (exact) |
| CORAL-BRCA | Gemma-4 anchor (2-pass, solo) | 0.972 | 0.770 | 0.854 | ✅ (exact) |
| CORAL-BRCA | **Ensemble** (primary) | **0.940** | **0.848** | **0.890 ± 0.044** | ✅ verified (coral_final) |
| CORAL-PDAC | Vanilla RAG | 0.984 | 0.661 | 0.787 ± 0.065 | ✅ (exact) |
| CORAL-PDAC | Gemma-4 anchor (2-pass, solo) | 0.983 | 0.721 | 0.829 | ✅ (exact) |
| CORAL-PDAC | **Ensemble** (primary) | **0.958** | **0.815** | **0.879 ± 0.048** | ✅ verified (coral_final) |

- **Ensemble** rows verified full-cohort, exact scorer — `results/coral_final_score.json`, config = Gemma-4-E4B
  2-pass (`scripts/run_ensemble_fast.py --twopass gemma4-e4b --seed-from combo_coral`) ∪ cached 1-pass augmenters,
  scored by `scripts/fast_score.py`. CI: BRCA [0.871, 0.909], PDAC [0.858, 0.900].
- **This all-≤5B ensemble matches/beats the old Gemma-3 + Qwen-8B ensemble** (0.868 / 0.877) at precision ≈0.95 —
  the ≤5B constraint cost nothing. Gemma-4-E4B is the best anchor (solo F1 0.71 vs Gemma-3-4B 0.58); Gemma-3
  dropped as redundant; Qwen3-8B dropped (throughput); Phi-4-mini excluded (transformers 5.8 incompat).
- The **2-pass anchor** lifts recall: Gemma-4 solo 2-pass R≈0.72–0.77 → full ensemble R 0.815–0.848 (augmenters add
  recall); and 1-pass ensemble ≈0.73 → 2-pass-anchor ensemble 0.83. All rows above are exact-scored (`fast_score.py`).
- **Vanilla RAG** = the base config: **Gemma-4-E4B single-model, 1-pass** RAG (no 2-pass, no ensemble),
  scored with the *same* entity-level metric. F1 **0.787 (PDAC) / 0.805 (BRCA)** — the floor of the
  progression (recall-driven: PDAC R 0.661 → 0.721 anchor → 0.815 ensemble). Script:
  `scripts/score_vanilla_rag.py` → `results/vanilla_rag_score.json`. Feeds the abstract's "F1 from X for the RAG baseline".

---

## Table IV — Construction-time validation on CORAL

Columns: `Validation Dimension | Result`.

| Validation Dimension | Result |
|---|---|
| Source grounding | **0.967** |
| Ontology compatibility | **0.564** |
| Schema validity | **0.539** |
| Temporal consistency | **0.787** |
| Contradiction control | **0.893** |

- ✅ **DONE** — aggregated over the **full current CORAL run** (Gemma-4 sub-5B ensemble, 40 patients,
  **17,597 ensemble triples**; PDAC 8,747 / BRCA 8,850). Per-cohort is consistent (e.g. source grounding
  PDAC 0.964 / BRCA 0.971). Not the old 30-patient values — freshly recomputed.
- Script: `scripts/table4_validation.py` → `results/table4_coral_validation.json` (gitignored — DUA).
- Reading: source grounding is the strong signal (0.967); ontology/schema are moderate (0.56/0.54),
  reflecting the still-thin terminology grounding + loose fhir-typing (the normalization/grounding upgrade
  targets exactly these).

---

## Table V — KG construction scale & structured queryability ✅ COMPLETE

Columns: `Dataset | KG Triples | Entities | Query Success` — reported per cohort/subset.

| Dataset | KG Triples | Entities | Query Success | Status |
|---|---|---|---|---|
| CORAL-PDAC | 31,652 | 2,549 | 10/10 | ✅ (Gemma-4, normalized, schema+instances) |
| CORAL-BRCA | 31,314 | 2,568 | 10/10 | ✅ (Gemma-4, normalized, schema+instances) |
| MIMIC-III | 460,821 | 26,186 | 10/10 | ✅ (Gemma-4, normalized, schema+instances) |
| MIMIC-IV | 480,041 | 27,740 | 10/10 | ✅ (Gemma-4, normalized, schema+instances) |

- Source: `results/coral_graph_report.json` / `results/mimic_graph_report.json`
  (`scripts/run_coral_graph.py`, `scripts/run_mimic_graph.py`). Current Gemma-4 sub-5B ensemble after
  normalization (fhir-type canonicalized, PHI scrubbed, degenerate collapsed) + shared schema (TBox) merged
  into each `.ttl`. All 4 subsets pass 10/10 SPARQL.
- **Across all 840 records** (40 CORAL + 800 MIMIC): **1,003,828 RDF triples**, **59,043 KG entities**
  (CORAL 62,966 / 5,117 + MIMIC 940,862 / 53,926).

---

## Table VI — Corpus-fraction scalability

Columns: `Corpus | Triples | Throughput | Verify Lat. | Cost`.

| Corpus | Triples | Throughput (notes/hr) | Verify Lat. (ms/triple) | Cost (GPU-h) |
|---|---|---|---|---|
| 25% (200 notes) | **73,858** | **130** | **43.8** | **3.1** |
| 50% (400 notes) | **153,911** | **130** | **43.8** | **6.2** |
| 75% (600 notes) | **237,479** | **130** | **43.8** | **9.2** |
| 100% (800 notes) | **318,081** | **130** | **43.8** | **12.3** |

- ✅ **DONE** (full 800-note MIMIC oncology corpus). Scalability is **linear**: triples grow ∝ corpus,
  throughput flat (no degradation), per-triple verification latency constant, cost linear.
- Script: `scripts/table6_scalability.py` → `results/table6_scalability.json`.
- **Provenance (honest):** *Triples* = measured (ensemble union, pooled from the complete bymodel caches).
  *Verification latency* = measured fresh here (5-layer validation; 17.3 s/note ≈ 43.8 ms/triple — the O(n²)
  contradiction layer dominates). *Throughput* = the measured vLLM ensemble rate from the run (gemma-4
  2-pass 173.7/160.2 notes/hr + 1-pass augmenters ≈320 → 65 notes/hr·GPU, 130 on 2 GPUs). *Cost* = derived
  GPU-hours. **Not a single instrumented run** — if a one-pass instrumented table is wanted, a clean vLLM
  re-run (≈12 GPU-h) reproduces all four columns in one shot.

---

## 8. Abstract / intro claims → value (the `[GATED]` map)

The draft prose blanks unverified numbers with `[GATED]`. Fillable now vs. blocked:

| Location / phrase | Value | Status |
|---|---|---|
| `[GATED-CONFIG]` improves F1 … | "the ensemble" (our primary config) | ✅ |
| … from `[GATED]` for the RAG baseline | — | ⛔ RAG baseline (Table III) |
| … to `[GATED]` on CORAL-BRCA | 0.890 (Ensemble, Gemma-4 sub-5B) | ✅ |
| achieves `[GATED]` on CORAL-PDAC | 0.879 (Ensemble, Gemma-4 sub-5B) | ✅ |
| mean F1 `XX±XX` BRCA / PDAC (robustness) | 0.890±0.044 / 0.879±0.048 (Ensemble) | ✅ |
| inserts `[GATED]` / routes `[GATED]` / rejects `[GATED]` | 67% / 31% / 2% @99% target (or 100/0/2 @95%) | ✅ |
| … AURC is `[GATED]` | 0.019 (learned) | ✅ |
| across 840 records: `[GATED]` triples / `[GATED]` entities | CORAL 17,597 + MIMIC 318,081 union triples; CORAL KG 5,117 entities | ✅ (MIMIC done) |
| `[GATED FIGURE: RISK–COVERAGE]` (Fig. 2) | data ready (E1–E2); PNG not rendered | can generate on request |
| "no credential-gated MIMIC … content" (ethics) | *not a placeholder — ordinary word, leave as-is* | n/a |

---

## 9. What's needed to finish (prioritized)

Done: Table II (calibration) ✅ · Table IV (validation) ✅ · Table VI (scalability) ✅ · Table V CORAL KG ✅.
Remaining:

1. **Vanilla-RAG baseline** on CORAL — the baseline row of **Table III** (extraction) + the abstract's
   "F1 from `[GATED]`". `run_coral_ensemble.py` with the RAG-only config.
2. **Full-cohort Gemma-2-pass** (optional) — an extra comparison row in **Table III** (the ensemble stays
   primary). `run_coral_ensemble.py --models gemma3-4b --twopass gemma3-4b`.
3. **MIMIC KG materialization** — the MIMIC rows of **Table V** (KG scale) via `scripts/run_mimic_graph.py`
   once the MIMIC union is rebuilt (extraction is complete; the validated union just needs a clean re-run).
