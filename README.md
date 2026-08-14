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

**Recall levers** (validated on CORAL smoke patients `pdac_0` + `brca_20`):
1. **Two-pass extraction** — pass 2 re-extracts with pass-1 triples injected as graph-neighborhood evidence;
   the two passes are unioned and trust-filtered. Lifts Gemma recall ~0.71 → ~0.83.
2. **Ensemble union** — union the 2-pass anchor with other models' outputs (dedup), then filter. Strongest
   lever, and it *also raises precision* (independent models reinforce true entities).

## Experiments (CORAL smoke: `pdac_0` + `brca_20`, entity-level vs expert gold)

**Extractor comparison (single-pass).** Gemma-3-4B and Qwen3-8B lead; Llama-3.2-3B trails on recall.
Qwen3-8B is strong but **brittle** — it canonicalizes entities under long prompts, so its span-level recall
can collapse (pdac 0.917 → 0.402 when caps were raised). Small MoEs are **not viable**: Phi-mini-MoE is
incompatible with transformers 5.8, and OLMoE-1B-7B is too weak (0–7 triples). **Gemma-3-4B is selected**
as the stable, span-faithful anchor (= the paper's "Gemma 3 4B"). Note: the recovered code pointed at a
broken `google/gemma-4-E4B-it`; the registry now uses the official `google/gemma-3-4b-it`.

**Recall progression** (all zero-hallucination / fully source-grounded):

| Config | pdac_0 F1 | brca_20 F1 | BRCA recall | BRCA precision |
|---|---|---|---|---|
| Gemma single-pass | 0.70–0.83 | 0.75–0.86 | 0.656 | 0.885 |
| Gemma **2-pass** | 0.833 | 0.861 | 0.828 | 0.896 |
| Gemma ∪ Llama | 0.854 | 0.876 | 0.844 | 0.910 |
| **Ensemble ×3** (Gemma ∪ Qwen ∪ Llama) | **0.879** | **0.906** | **0.890** | **0.923** |

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

## Data & ethics

CORAL (oncology) and MIMIC-III/IV (ICU/EHR) are **credential-gated via PhysioNet** and governed by Data Use
Agreements. **No patient data, extracted triples, or executed-notebook outputs are committed** — `data/`,
`results/`, and executed notebooks are git-ignored. Obtain the datasets under your own credentials and place
CORAL under `data/coral/{pdac,breastca}/` (`N.txt` narratives, `N.ann.txt` gold). TRUST-KG is an assistive
KG-construction framework, not a clinical decision system.

## Status / roadmap

- [x] Extractor selected (Gemma-3-4B); 2-pass + ensemble recall validated on smoke patients
- [x] Config finalized: ensemble ×3 (Gemma 2-pass + Qwen + Llama) — clears paper recall/F1 target
- [~] Full 40-patient CORAL ensemble run (in progress, split across 2 GPUs) → Tables II, XII
- [ ] Calibration + selective admission (ECE/Brier/NLL, AURC/coverage) → Tables VIII, IX, XI
- [ ] MIMIC-III/IV scale run (via BigQuery) → Tables I, III, VI, XIII, XIV
- [ ] RDF materialization + SPARQL cohort retrieval → Table XV
- [~] Per-patient miss analysis (recall by gold label + recurring substantive misses) → targets a
      supervised fine-tune of the extractor in the next version (`scripts/analyze_misses.py`)
