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

**Two-pass extraction** is the main recall lever: pass 1 extracts candidate triples; pass 2 re-extracts with
pass-1 triples injected as graph-neighborhood evidence, then the two passes are unioned and filtered.

## Current results (Gemma-3-4B, CORAL smoke)

Entity-level vs expert `.ann.txt` gold, `pdac_0` + `brca_20` (single-pass → 2-pass):

| Stage | Recall | Precision | F1 | Hallucination |
|---|---|---|---|---|
| single-pass | 0.60–0.79 | 0.83–0.90 | 0.70–0.84 | ~0 |
| **2-pass union** | **0.83–0.84** | 0.83–0.90 | 0.83–0.86 | ~0 |

Extractor comparison found **Gemma-3-4B** the best trade-off (stable, span-faithful, precision held while
recall rises); Qwen3-8B is brittle (canonicalizes under long prompts), and small MoEs (Phi-mini-MoE,
OLMoE-1B-7B) were not viable. See `notebooks/TRUSTKG_Results.ipynb`.

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
  run_coral_full.py            # full-CORAL Gemma 2-pass run (Tables II, XII)
  run_gemma_2pass.py           # 2-pass + trust filter on a patient subset
  compare_models_smoke.py      # extractor comparison
  validate_against_ann.py      # quality validation vs gold
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

# Full-CORAL 2-pass (all 40 patients; resumable) → results/coral_full_metrics.json
python scripts/run_coral_full.py --gpu 0

# Regenerate + execute the results notebook
python scripts/build_results_notebook.py
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=llmft notebooks/TRUSTKG_Results.ipynb
```

## Data & ethics

CORAL (oncology) and MIMIC-III/IV (ICU/EHR) are **credential-gated via PhysioNet** and governed by Data Use
Agreements. **No patient data, extracted triples, or executed-notebook outputs are committed** — `data/`,
`results/`, and executed notebooks are git-ignored. Obtain the datasets under your own credentials and place
CORAL under `data/coral/{pdac,breastca}/` (`N.txt` narratives, `N.ann.txt` gold). TRUST-KG is an assistive
KG-construction framework, not a clinical decision system.

## Status / roadmap

- [x] Extractor selected (Gemma-3-4B), 2-pass recall validated on smoke patients
- [ ] Full 40-patient CORAL run → Tables II, XII
- [ ] Calibration + selective admission (ECE/Brier/NLL, AURC/coverage) → Tables VIII, IX, XI
- [ ] MIMIC-III/IV scale run (via BigQuery) → Tables I, III, VI, XIII, XIV
- [ ] RDF materialization + SPARQL cohort retrieval → Table XV
