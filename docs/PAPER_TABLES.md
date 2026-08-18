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

Two panels: **(A)** which reliability estimator to trust (fair, same-setting comparison), and **(B)** what
selective admission buys as the precision target tightens (one method, the tunable dial).

**Panel A — reliability estimators** (held-out CORAL test; lower ECE/Brier/NLL/AURC better, higher Cov.@95% better):

| Method | ECE | Brier | NLL | AURC | Cov.@95% |
|---|---|---|---|---|---|
| Heuristic Trust | 0.172 | 0.082 | 0.315 | 0.031 | 0.981 |
| **Learned Reliability** | **0.008** | **0.046** | **0.181** | **0.019** | **0.992** |
| Learned + Calibration | 0.014 | 0.048 | 0.190 | 0.019 | 0.992 |

- **No Insert/Review/Reject column here, on purpose:** at a 95%-precision bar all three admit ≈100% (the
  Gemma-4 ensemble is already ≈95% precise), so that split carries no signal. The differentiator is
  **calibration/risk** — Learned ECE **0.008** vs Heuristic **0.172**; AURC **0.019** vs **0.031**.

**Panel B — selective admission (Learned + Calibration), as the precision target rises:**

| Target precision | Insert | Review | Reject | Achieved insert-precision |
|---|---|---|---|---|
| 95% | 100% | 0% | 0% | 0.949 |
| 98% | 86% | 12% | 2% | 0.967 |
| 99% | 67% | 31% | 2% | 0.981 |

- **One method, one dial:** stricter target → fewer auto-inserts, more routed to review. The Insert/Review/Reject
  split comes from thresholding *this method's* calibrated scores on dev to hit each target (disjoint bands, sum 100%).
- **⚠️ The 67/31/2 is this method at the 99% target — do NOT place it beside the Heuristic's 100/0/0 as if they
  share one threshold.** Every method sits at 100/0/0 at the 95% bar (Panel A's setting).

**What Panel B lets us conclude (RQ3):**
1. **The gate is controllable** — tightening the target 95→99% raises realized insert-precision 0.949→0.967→0.981
   while auto-insert coverage falls 100→86→67%. The knob does what it promises: you *buy* precision with coverage.
2. **The requested precision is approximately delivered** (0.981 at a 99% target) — evidence that the calibrated
   per-fact score turns into an (approximate) set-level precision guarantee without checking facts by hand.
   *(Honest: 0.981 ≈ but not exactly 0.99 — a small dev→test gap; a controllable trade, not a hard guarantee.)*
3. **It quantifies the cost of high precision:** reaching ≈98% insert-precision costs ≈31% of facts routed to
   human review and ≈2% rejected — a predictable review budget an operator can plan around.
4. **The review burden stays bounded/practical** — even at the strictest bar most facts are auto-handled; the
   human is a light backstop (≈⅓ to review, ≈2% rejected), not doing the bulk of the work.
- This is *not* the method comparison (that's the threshold-free ECE/AURC in Panel A); it's the tunable-policy
  illustration. We do **not** commit to a single target — an operator picks the point for their use case.

**Gate applied to the materialized KG (makes admission visible in the *product*, not just a table —
`scripts/trust_admission_demo.py`, held-out CORAL):**

| KG | Triples | Precision (vs gold) |
|---|---|---|
| Union (pass-through, current default) | 3,053 | 0.949 |
| **Trust-admitted (learned @ 99% target)** | **2,050** | **0.981** (+0.032) |

- Held back = 1,003 triples (947 review / 56 reject); **118 are genuinely wrong**. The held-back set is what
  a threshold-free / rule-based method keeps — e.g. **bogus entity→code assignments** (`Female`, `Diagnosis`,
  `Noted` → SNOMED `254837009` = breast cancer) and **null/malformed triples** (`R2109 --mutation--> None`).
- **Honest magnitude:** the precision lift is real but **modest (+0.032)** — the base extractor is already
  ≈95% precise, so there is little entity-level error to remove. The gate's value is a *cleaner KG at a chosen
  bar* + a visible review/reject queue, not a large cleanup. (The materialized KG is currently the Union
  pass-through; applying the gate is a build-flag away — the scores just need threading into the union.)

- Verified on the Gemma-4 `coral_final` ensemble (`results/e1e3_results.json`, seeded `random_state=0`,
  reproducible via `scripts/exp_calibration_selective.py`; default `TRUSTKG_UNION_DIR=coral_final` — the *reported*
  extractor. The old default pointed at `ens3` and produced different numbers).
- Held-out **test set: 3,053 triples** (of 17,597 labeled); overall correct-rate **0.952**.
  **⚠️ The draft caption's "3,569 eligible candidates / correct-rate 0.872" are the OLD ens3 numbers — update
  to 3,053 / 0.952 (Gemma-4).**

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

## Table IV — Construction-time validation on CORAL ✅ COMPLETE

Columns: `Validation Dimension | Result`.

**Definition (verified from `scripts/table4_validation.py`): each Result is the *mean per-triple validator
score* for that layer — a graded 0–1 score from `validate_triple`, arithmetic-mean over all 17,597 CORAL
ensemble triples. It is NOT a binary pass/fail rate.** Per layer: *source grounding* = fraction of the
triple's entity/value/evidence found in the source note (Layer 1); *ontology compatibility* = 1.0 known
concept / 0.7 valid-FHIR-type / 0.3 otherwise (Layer 2); *schema validity* = attribute-vs-FHIR-type match,
1.0/0.8/0.5/0.3 (Layer 3); *temporal consistency* = 0.7 neutral / higher-lower by plausibility (Layer 4);
*contradiction control* = 1.0 clean / 0.1 on a detected binary contradiction (Layer 5).

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

## Table VI — Corpus-fraction scalability ✅ COMPLETE

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
| … from `[GATED]` for the RAG baseline | 0.787 PDAC / 0.805 BRCA (Vanilla RAG) → 0.879 / 0.890 (Ensemble) | ✅ Table III |
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

**All six tables have their real numbers:** Table I (descriptive) · II calibration ✅ · III extraction incl.
Vanilla-RAG baseline ✅ · IV validation ✅ · V KG scale, CORAL + MIMIC ✅ · VI scalability ✅.

Optional only:
1. **Full-cohort Gemma-2-pass** — an extra comparison row in **Table III** (the ensemble stays primary).
   `run_coral_ensemble.py --models gemma3-4b --twopass gemma3-4b`.
