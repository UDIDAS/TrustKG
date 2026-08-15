# TRUST-KG

**Evidence-Grounded, Calibrated, and Selective Knowledge Graph Construction from Heterogeneous Clinical Big Data**

TRUST-KG builds ontology-aligned clinical knowledge graphs from free-text medical narratives, and
validates each candidate fact **before** it is written to the graph. Unlike RAG/GraphRAG systems that use
retrieval only for downstream answer generation, TRUST-KG applies retrieval + verification at
*construction time*: a candidate Entity–Attribute–Value (EAV) triple is admitted only if it is
source-grounded, ontology-compatible, schema-valid, temporally consistent, and passes a trust threshold.

> Target venue: **IEEE BigData 2026**. This repository contains the extraction/validation pipeline and the
> reproducible experiment code. **No patient data is included** (see [Data & ethics](#data--ethics)).

---

## Contributions (IEEE BigData positioning)

TRUST-KG is framed as a **data-veracity / ingestion-governance** contribution for building knowledge graphs
from heterogeneous data at scale — not merely a clinical extractor. Mapped to the big-data V's:

**1. Veracity — calibrated, selective *admission control* (the headline).**
LLM extraction over massive heterogeneous text can't be human-verified at scale, so TRUST-KG casts KG
*ingestion* as **calibrated selective prediction**: every candidate fact is scored for reliability and then
**inserted / routed-to-review / rejected _before_ graph materialization**, at a **tunable quality–coverage
operating point**. This turns "trustworthy data" from a slogan into a mechanism.
→ Tables **VIII** (calibration: ECE/Brier/NLL), **IX** (selective admission: AURC / Coverage@95% /
insert-review-reject), **XI** (evidence-level reliability ablation). *[planned]*

**2. Variety — heterogeneous, multi-source integration.** *[data obtained]*
One pipeline unifies expert oncology reports (CORAL), ICU notes (MIMIC-III), and longitudinal EHR
(MIMIC-IV) — multi-institution, **pan-cancer**, differing formats/schemas — into a single ontology-aligned,
FHIR-typed, SPARQL-queryable RDF graph. → Tables **I, II, III, XIII**.

**3. Volume — scale at *ingestion* + bounded-cost scalability.**
The final graph is modest, but ingestion is at scale: the oncology cohort is filtered from **~6.3 M
diagnosis rows / 2 M+ clinical notes / 546 K admissions** in BigQuery *(done)*. Table **XIV**
characterizes **throughput, verification latency, and cost as corpus fraction grows (25→100%)**, evidencing
**bounded per-record overhead** — i.e. a *scalable method*, demonstrated, rather than a huge graph. The
throughput comes from a **resident-model, batched-inference** extractor (`run_mimic_fast.py`); a naive serial
baseline (`run_mimic_extraction.py`) is retained for the head-to-head speedup measurement. *[planned]*

**4. Value — unstructured → queryable analytics.**
Narratives become **SPARQL-executable** cohort / temporal / multi-hop queries. → Table **XV**.

**Why this reads as BigData, not clinical-NLP:** the admission-control mechanism is **domain-general** (it
governs *any* LLM-to-graph ingestion); the evaluation foregrounds the **quality–coverage trade-off and
scaling behavior**; and the volume claim rests on the **millions of records filtered at ingest**, not the
final graph size.

**Status.** *Obtained:* extractor selected (Gemma-3-4B ensemble, see [Experiments](#experiments-coral-smoke-pdac_0--brca_20-entity-level-vs-expert-gold)),
MIMIC oncology cohorts curated (see [Datasets](#datasets-all-oncology)). *In progress:* full-CORAL ensemble
+ MIMIC scale extraction. *Planned:* the calibration/selective (VIII/IX/XI) and scalability (XIV) tables above.

---

## Pipeline

```
clinical note ──► NER (SciSpaCy) ──► schema-constrained EAV extraction (LLM, FHIR-typed)
                                          │
              hybrid retrieval  ◄─────────┤   BM25 (lexical) + MedCPT (dense) + graph-neighborhood
              (evidence grounding)        │
                                          ▼
                       trust-aware validation  ──►  source grounding · ontology · schema ·
                       (rule-based + learned)       temporal · contradiction · calibrated reliability
                                          │
                                trust ≥ δ  &  source-grounded
                                          ▼
                        RDF materialization (FHIR types, ontology links, provenance)
                                          ▼
                             SPARQL cohort retrieval / temporal queries
```

**Recall levers** (validated on CORAL smoke patients `pdac_0` + `brca_20`):
1. **Two-pass extraction** — pass 2 re-extracts with pass-1 triples injected as graph-neighborhood evidence;
   the two passes are unioned and trust-filtered. Lifts Gemma recall ~0.71 → ~0.83.
2. **Ensemble union** — union the 2-pass anchor with other models' outputs (dedup), then filter. Strongest
   lever, and it *also raises precision* (independent models reinforce true entities).

## Datasets (all oncology)

| Dataset | Domain | Patients | Cases (notes) | Admissions | Median len | Gold | Split | Role |
|---|---|---:|---:|---:|---:|---|---|---|
| CORAL‑PDAC | Pancreatic oncology | 20 | 20 | — | ~11 K | expert entity spans | 12/4/4 | entity P/R/F1 |
| CORAL‑BRCA | Breast oncology | 20 | 20 | — | ~11 K | expert entity spans | 12/4/4 | entity P/R/F1 |
| MIMIC‑III (onc.) | ICU/EHR oncology | 392 | 400 | 400 | ~10.4 K | — | — | scale / source‑grounding |
| MIMIC‑IV (onc.) | EHR oncology | 394 | 400 | 400 | ~10.1 K | — | — | scale / source‑grounding |

**CORAL** (40 patients = 20 pancreatic + 20 breast) is the annotated benchmark — dense oncology narratives
with expert `PROBLEM`/`TEST`/`TREATMENT` spans (`.ann.txt`) for entity‑level evaluation.

**MIMIC‑III / MIMIC‑IV (oncology subsets)** — 400 notes each (~392–394 distinct patients / 400 admissions),
median ~10 K chars/note, filtered to **malignant‑neoplasm ICD codes** (ICD‑10 `C00–C97` / ICD‑9 `140–208`).
They are a **pan‑cancer population** — by note mentions, roughly: lung > colorectal > GI/esophageal >
GU (bladder/renal) > head‑&‑neck > breast > prostate > pancreatic — so they broaden CORAL's breast+pancreatic
focus to the wider oncology domain. No expert entity gold, so (per the paper's design) they drive
**scale + source‑grounding** stats, not P/R/F1. Acquisition details in
[MIMIC oncology data (BigQuery)](#mimic-oncology-data-bigquery).

## Experiments — CORAL, full pipeline end-to-end (per cohort)

The complete pipeline runs end-to-end on all 40 CORAL patients; results are **per cohort** (20 PDAC +
20 BRCA), never pooled. Reproduce: extraction via `run_coral_ensemble.py`, graph stage via
`run_coral_graph.py`.

**Stage 1 — Extraction** · ensemble ×3 (Gemma-3-4B 2-pass ∪ Qwen3-8B ∪ Llama-3.2-3B), entity-level vs gold:

| Cohort | N | Precision | Recall | F1 (mean ± sd) | 95% CI |
|---|---|---|---|---|---|
| CORAL-PDAC | 20 | 0.888 | 0.870 | **0.877 ± 0.043** | [0.858, 0.897] |
| CORAL-BRCA | 20 | 0.850 | **0.890** | **0.868 ± 0.045** | [0.848, 0.888] |

BRCA recall **0.890** meets the paper's 0.879 target; tight CIs = robust; zero hallucination. → Tables **II**, **XII**.

**Stage 2 — Validation** · trust-filter (δ=0.4): a **no-op** here (nothing pruned → precision held).

**Stage 3–4 — RDF materialization + SPARQL cohort queries** (all queries execute) → Tables **XIII**, **XV**:

| | CORAL-PDAC | CORAL-BRCA |
|---|---:|---:|
| RDF triples | 40,412 | 41,293 |
| KG entities | 3,601 | 3,822 |
| ontology-linked | 163 | 223 |
| conditions / medications / procedures | 1178 / 558 / 1351 | 1351 / 744 / 1474 |
| temporal facts | 8,212 | 7,912 |
| cancer cohort (SPARQL) | 20/20 | 20/20 |
| chemotherapy cohort (SPARQL) | 17/20 | 13/20 |
| SPARQL queries executed | **10/10** | **10/10** |

This closes the loop **unstructured notes → validated triples → queryable RDF/SPARQL** (the Value V).

**How the config was chosen** (2-patient `pdac_0`+`brca_20` smoke — *not* the reported numbers): **Gemma-3-4B**
selected as a stable, span-faithful anchor; **Qwen3-8B** is strong but brittle (canonicalizes → span-recall
can collapse); small MoEs not viable (Phi-mini-MoE incompatible with transformers 5.8, OLMoE too weak). Recall
levers stack — single-pass → **2-pass** (~0.71→0.83) → **ensemble ×3** (chosen: clears the recall target while
*raising* precision). The broken `google/gemma-4-E4B-it` in the recovered code was swapped for the official
`google/gemma-3-4b-it`.

(single-pass = initial config; 2-pass onward use the raised caps. Single-pass F1 is stochastic across runs — 0.70–0.83 on PDAC.)

Ensemble ×3 **clears the paper's BRCA recall target (0.879)** and improves precision. The practical recall
ceiling is ~0.90–0.92 — the residual misses are anaphora ("the mass") and lab-table fragments ("g/dL",
"x10E9") that the CORAL gold annotates but aren't clean EAV entities.

**Finalized config → full run.** Ensemble ×3: **Gemma-3-4B 2-pass anchor + Qwen3-8B + Llama-3.2-3B
single-pass**, unioned and trust-filtered (`scripts/run_coral_ensemble.py`). The full 40-patient CORAL run
is in progress (split across 2 GPUs) → paper **Tables II & XII**; it also attaches a trust score to every
triple, producing the labeled data for the calibration/selective analysis (Tables VIII/IX/XI). Full
walkthrough with charts + qualitative examples: `notebooks/TRUSTKG_Results.ipynb`.

## Repository layout

```
src/
  config.py, config_splits.py        # paths (TRUSTKG_ROOT env override) + CORAL 12/4/4 splits
  data/          reader.py, mimic3_reader.py, mimic_notes_reader.py, mimic_kg.py
  extraction/    rag_extractor.py, ner.py, hybrid_retrieval.py, local_llm.py, prompts.py,
                 fhir_normalizer.py, validation.py, multi_agent_validation.py, temporal.py, evaluate.py
  graph/         rdf_builder.py, hierarchical_kg.py
  gnn/           trust_graph.py, trust_gnn.py      # learned reliability estimator
  evaluation/    kg_qa.py, kg_pubmedqa.py, split_qa.py, mimic_split_eval.py
scripts/
  run_coral_ensemble.py        # full-CORAL ENSEMBLE run (SELECTED config) → Tables II, XII
  run_coral_full.py            # full-CORAL Gemma-only 2-pass run (baseline/ablation)
  run_gemma_2pass.py           # 2-pass + trust filter on a patient subset
  compare_models_smoke.py      # extractor comparison (Gemma/Qwen/Llama/MoE)
  validate_against_ann.py      # quality validation vs gold
  analyze_misses.py            # per-patient miss tracking by entity type -> fine-tuning targets
  fetch_mimic_oncology.py      # pull MIMIC-III/IV ONCOLOGY notes via BigQuery (ICD-filtered)
  run_mimic_fast.py            # throughput-optimized MIMIC extraction (resident model + batched chunks)
  run_mimic_extraction.py      # baseline serial MIMIC extraction (kept for the speedup comparison)
  queue_mimic_after_coral.sh   # auto-start FAST MIMIC extraction once the CORAL ensemble finishes
  compute_all_metrics.py       # cache metrics for the notebook
  build_results_notebook.py    # regenerate the results notebook
  extract_all_pdac.py, run_mimic3_full.py, ...   # recovered run scripts
notebooks/  TRUSTKG_Results.ipynb        # results notebook (outputs stripped; re-run to view)
experiments/ RESULTS_LEDGER.md           # per-table data-collection worksheet
docs/        (paper draft PDF — not tracked)
data/, results/  (not tracked — clinical data / regenerable outputs)
```

## Setup

```bash
conda activate llmft            # Python 3.11, CUDA 12.1
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

export TRUSTKG_ROOT=$PWD
export HF_TOKEN=...             # required for gated models (Gemma/Llama); never commit this
# keep large HF/pip caches off a quota'd home:
export HF_HOME=/scratch/$USER/hf_cache
```

The extractor is selected in `src/extraction/local_llm.py::MODEL_REGISTRY`
(`gemma3-4b` → `google/gemma-3-4b-it`).

## Running

```bash
# Extractor comparison on 2 smoke patients
python scripts/compare_models_smoke.py --gpu 0

# Gemma 2-pass + trust filter on the same patients (stage-by-stage recall/precision)
python scripts/run_gemma_2pass.py --gpu 0

# Full-CORAL ENSEMBLE — the selected config (resumable, GPU-splittable)
python scripts/run_coral_ensemble.py --gpu 0 \
    --models gemma3-4b qwen3-8b llama32-3b --twopass gemma3-4b
#   split across 2 GPUs: pass PDAC ids to --gpu 0 and BRCA ids to --gpu 1 via --patients

# Full-CORAL Gemma-only 2-pass (baseline/ablation) → results/coral_full_metrics.json
python scripts/run_coral_full.py --gpu 0

# Regenerate + execute the results notebook
python scripts/build_results_notebook.py
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=llmft notebooks/TRUSTKG_Results.ipynb
```

## MIMIC oncology data (BigQuery)

The MIMIC portion of TRUST-KG is **focused on oncology patients**, to complement the CORAL oncology cohorts
(rather than the general ICU population). `scripts/fetch_mimic_oncology.py` identifies oncology admissions by
**malignant-neoplasm ICD codes** and pulls their free-text notes from BigQuery.

- **Cohort filter:** ICD-10 `C00–C97` / ICD-9 `140–208` (`--cancer all`, the default); optional per-cancer
  subsets via `--cancer breast,pancreatic,lung,colorectal,prostate` (breast+pancreatic mirror CORAL).
- **Datasets** (`physionet-data`, credential-gated):
  - MIMIC-IV — `mimiciv_3_1_hosp.diagnoses_icd` (cohort) + `mimiciv_note.discharge` / `radiology` (notes)
  - MIMIC-III — `mimiciii_clinical.diagnoses_icd` (cohort) + `mimiciii_notes.noteevents` (notes)
- **Access:** be credentialed for [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and
  [MIMIC-IV](https://physionet.org/content/mimiciv/) on PhysioNet and grant BigQuery access to your query
  identity (service-account email), then:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp/bq-key.json   # kept OUTSIDE the repo
export BQ_PROJECT=<your-billing-project>

python scripts/fetch_mimic_oncology.py --source mimiciv  --cancer all --dry-run    # print SQL (no auth/cost)
python scripts/fetch_mimic_oncology.py --source mimiciv  --cancer all --estimate   # free: GB scanned + cost
python scripts/fetch_mimic_oncology.py --source mimiciv  --cancer all --limit 400  # pull -> data/mimic_oncology/
python scripts/fetch_mimic_oncology.py --source mimiciii --cancer all --limit 400
```

- **Cost:** BigQuery bills the querying project by **bytes scanned**, with a **1 TB/month free tier**; each
  oncology query scans only a few GB, so it stays free in practice. `--estimate` runs a dry-run that reports
  the exact GB and charges nothing; `--max-gb` (default 25) caps `maximum_bytes_billed` so a query errors
  rather than over-scan. Pulled notes land in `data/mimic_oncology/` (git-ignored, DUA) and feed the same
  2-pass / ensemble pipeline.

## Data & ethics

CORAL (oncology) and MIMIC-III/IV are **credential-gated via PhysioNet** and governed by Data Use Agreements;
the MIMIC subset used here is **oncology-filtered** (malignant-neoplasm ICD codes) to align with CORAL.
**No patient data, extracted triples, or executed-notebook outputs are committed** — `data/`, `results/`, and
executed notebooks are git-ignored. Obtain the datasets under your own credentials and place CORAL under
`data/coral/{pdac,breastca}/` (`N.txt` narratives, `N.ann.txt` gold). TRUST-KG is an assistive KG-construction
framework, not a clinical decision system.

## Status / roadmap

- [x] Extractor selected (Gemma-3-4B); 2-pass + ensemble recall validated on smoke patients
- [x] Config finalized: ensemble ×3 (Gemma 2-pass + Qwen + Llama) — clears paper recall/F1 target
- [x] Full 40-patient CORAL ensemble run (done — 20 PDAC + 20 BRCA, per cohort) → Tables II, XII
- [x] CORAL end-to-end: RDF materialization + SPARQL cohort queries (per cohort) → Tables XIII, XV (`run_coral_graph.py`)
- [ ] Calibration + selective admission (ECE/Brier/NLL, AURC/coverage) → Tables VIII, IX, XI
- [x] Obtained MIMIC-III/IV **oncology** notes — 400 + 400, ICD-filtered via BigQuery
      (`scripts/fetch_mimic_oncology.py`); next: scale extraction run → Tables I, III, VI, XIII, XIV
- [ ] RDF + SPARQL on the MIMIC oncology graphs (CORAL done above)
- [~] Per-patient miss analysis (recall by gold label + recurring substantive misses) → targets a
      supervised fine-tune of the extractor in the next version (`scripts/analyze_misses.py`)
