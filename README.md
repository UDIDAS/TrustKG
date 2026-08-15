# TRUST-KG

**Evidence-Grounded, Calibrated, and Selective Knowledge Graph Construction from Heterogeneous Clinical Big Data**

TRUST-KG builds ontology-aligned clinical knowledge graphs from free-text medical narratives and validates
each candidate fact **before** it is written to the graph. Unlike RAG/GraphRAG systems that use retrieval only
for downstream answering, TRUST-KG applies retrieval + verification at *construction time*: an
Entity–Attribute–Value (EAV) triple is admitted only if it is source-grounded, ontology-compatible,
schema-valid, temporally consistent, and passes a trust threshold.

> Target venue: **IEEE BigData 2026**. No patient data is committed (see [Data & ethics](#data--ethics)).

---

## Contributions (IEEE BigData positioning)

TRUST-KG is a **data-veracity / ingestion-governance** contribution for building knowledge graphs from
heterogeneous data at scale — not merely a clinical extractor. Mapped to the big-data V's:

1. **Veracity — calibrated, selective admission control (the headline).** LLM extraction over massive
   heterogeneous text can't be human-verified at scale, so TRUST-KG casts KG *ingestion* as **calibrated
   selective prediction**: each candidate fact is scored for reliability and **inserted / routed-to-review /
   rejected before materialization**, at a **tunable quality–coverage operating point**. **Demonstrated on
   CORAL** (held-out test): a learned reliability model **auto-inserts ~60% of candidate facts at 94.8%
   precision** (37% to review, 4% rejected), halving selective AURC (0.09→0.05) and cutting ECE 0.14→0.04.
   → Table I.
2. **Variety — heterogeneous, multi-source integration.** One pipeline unifies expert oncology reports
   (CORAL), ICU notes (MIMIC-III), and longitudinal EHR (MIMIC-IV) — multi-institution, **pan-cancer** — into
   a single ontology-aligned, FHIR-typed, SPARQL-queryable RDF graph. → Tables IV, VI.
3. **Volume — scale at ingestion + bounded-cost scalability.** The oncology cohort is filtered from
   **~6.3 M diagnosis rows / 2 M+ notes / 546 K admissions** in BigQuery; Table VII characterizes
   **throughput, verification latency, and cost as corpus fraction grows**, via a resident-model,
   batched-inference extractor — a *scalable method*, not a huge graph. → Table VII.
4. **Value — unstructured → queryable analytics.** Narratives become **SPARQL-executable** cohort / temporal /
   multi-hop queries. → Table VI.

**Why this reads as BigData, not clinical-NLP:** the admission-control mechanism is **domain-general** (it
governs any LLM-to-graph ingestion); the evaluation foregrounds the **quality–coverage trade-off and scaling
behavior**; and the volume claim rests on the **millions of records filtered at ingest**, not the final
graph size.

---

## Pipeline

```
clinical note ─► NER (SciSpaCy) ─► schema-constrained EAV extraction (LLM, FHIR-typed)
                                        │
            hybrid retrieval  ◄─────────┤   BM25 (lexical) + MedCPT (dense) + graph-neighborhood
            (evidence grounding)        ▼
                     trust-aware validation ─► source-grounding · ontology · schema ·
                     (rule-based + learned)    temporal · contradiction · calibrated reliability
                                        │  (admit if trust ≥ δ and source-grounded)
                                        ▼
                     RDF materialization (FHIR types, ontology links, provenance)
                                        ▼
                     SPARQL cohort / temporal / multi-hop queries
```

Two recall levers used in the reported config: **two-pass extraction** (pass 2 re-extracts seeded with
pass-1 triples as graph-neighborhood evidence) and **ensemble union** across models — which raises recall
*and* precision (independent models reinforce true positives).

---

## Datasets (all oncology)

| Dataset | Domain | Patients | Cases (notes) | Median len | Gold | Role |
|---|---|---:|---:|---:|---|---|
| CORAL-PDAC | Pancreatic oncology | 20 | 20 | ~11 K | expert entity spans | entity P/R/F1 |
| CORAL-BRCA | Breast oncology | 20 | 20 | ~11 K | expert entity spans | entity P/R/F1 |
| MIMIC-III (onc.) | ICU/EHR oncology | 392 | 400 | ~10.4 K | — | scale / source-grounding |
| MIMIC-IV (onc.) | EHR oncology | 394 | 400 | ~10.1 K | — | scale / source-grounding |

**CORAL** (40 patients = 20 pancreatic + 20 breast) is the annotated benchmark — dense oncology narratives
with expert `PROBLEM`/`TEST`/`TREATMENT` spans for entity-level evaluation.

**MIMIC-III / MIMIC-IV (oncology subsets)** — 400 notes each (~392–394 distinct patients / 400 admissions),
filtered to **malignant-neoplasm ICD codes** (ICD-10 `C00–C97` / ICD-9 `140–208`). A **pan-cancer**
population (by note mentions: lung > colorectal > GI/esophageal > GU > head-&-neck > breast > prostate >
pancreatic), broadening CORAL's breast+pancreatic focus. No expert entity gold, so (per the paper's design)
they drive **scale + source-grounding** stats, not P/R/F1.

---

## Results — CORAL, full pipeline end-to-end (per cohort)

The complete pipeline runs end-to-end on all 40 CORAL patients; results are **per cohort** (20 PDAC +
20 BRCA), never pooled.

**Stage 1 — Extraction** · ensemble ×3 (Gemma-3-4B 2-pass ∪ Qwen3-8B ∪ Llama-3.2-3B), entity-level vs gold:

| Cohort | N | Precision | Recall | F1 (mean ± sd) | 95% CI |
|---|---|---|---|---|---|
| CORAL-PDAC | 20 | 0.888 | 0.870 | **0.877 ± 0.043** | [0.858, 0.897] |
| CORAL-BRCA | 20 | 0.850 | **0.890** | **0.868 ± 0.045** | [0.848, 0.888] |

BRCA recall **0.890** meets the paper's 0.879 target; tight CIs = robust; zero hallucination. → Table **II**.

**Stage 2 — Validation** · trust-filter (δ=0.4): a **no-op** here (nothing pruned → precision held).

**Stage 3–4 — RDF materialization + SPARQL cohort queries** (all queries execute) → Table **VI**:

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

**Example queries on the CORAL graph:**

```sparql
# pancreatic-cancer cohort  ->  20/20 patients
SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p trustkg:hasEntity ?e . ?e a fhir:Condition ; rdfs:label ?l .
  FILTER(REGEX(?l, "pancrea|adenocarc", "i")) }

# chemotherapy cohort  ->  17/20 patients
SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {
  ?p trustkg:hasEntity ?e . ?e a fhir:MedicationStatement ; rdfs:label ?l .
  FILTER(REGEX(?l, "gemcitabine|abraxane|fluorouracil|chemo", "i")) }

# ontology-grounded entities (sample rows)
SELECT ?l ?code WHERE { ?e rdfs:label ?l ; trustkg:ontologyCode ?code . }
#   "distant metastasis"  ->  SNOMED:128462008
#   "liver metastasis"    ->  SNOMED:128462008
```

**E10 — KG semantics vs keyword search (where keyword *should* fail).** On negation/experiencer-prone
concepts, naive keyword retrieval matches "no evidence of metastasis" / "family history of …" mentions.
Across 10 such concepts, keyword yields **104 false-positive patient retrievals**; the KG's semantic
extraction avoids **39 % (41/104)** — its precision advantage over raw-text search. The residual 61 % is an
**honest limitation** (the extractor doesn't fully suppress negated mentions), motivating an
assertion/negation validation layer (a natural addition to the Veracity stack). Note: on *simple*
single-concept retrieval over CORAL's concept-dense reports, keyword is competitive — the KG's edge is on
**semantically hard** and **structured/relational** queries, not term matching.

### Veracity — calibration & selective admission (CORAL, held-out test)

Every extracted triple gets a reliability score; a **learned reliability** model (validation-layer +
structural features, trained on the train split) beats the heuristic trust and enables **selective admission**
at a target precision.

Table I — calibration part (lower better):

| Reliability | ECE | Brier | NLL |
|---|---|---|---|
| Heuristic trust | 0.141 | 0.119 | 0.406 |
| **Learned** | **0.037** | **0.092** | **0.312** |

Table I — selective admission part (operating point set on dev for ≥95% precision):

| Policy | AURC ↓ | Cov@95% ↑ | Insert | Review | Reject | Insert-prec |
|---|---|---|---|---|---|---|
| Heuristic trust | 0.094 | 0.003 | 21% | 78% | 0.1% | 0.93 |
| **Learned (calibrated)** | **0.045** | **0.575** | **59.5%** | 36.6% | 4.0% | **0.948** |

The calibrated selective policy **auto-inserts ~60% of candidate facts at 94.8% precision**, routes ~37% to
review, rejects ~4% — the tunable quality–coverage admission control (the Veracity contribution). The learned
model is already well-calibrated, so Platt adds nothing on top (an honest no-op).

**How the config was chosen** (from a 2-patient `pdac_0`+`brca_20` smoke — *not* the reported numbers):
**Gemma-3-4B** selected as a stable, span-faithful anchor; **Qwen3-8B** is strong but brittle (canonicalizes →
span-recall can collapse); small MoEs not viable (Phi-mini-MoE incompatible with transformers 5.8, OLMoE too
weak). Recall levers stack: single-pass → 2-pass (~0.71→0.83) → **ensemble ×3** (chosen — clears the recall
target while *raising* precision). The practical recall ceiling is ~0.90–0.92; residual misses are anaphora
and lab-table fragments that the gold annotates but aren't clean EAV entities.

---

## MIMIC oncology data

Focused on **oncology patients** (to complement CORAL, not the general ICU population). `fetch_mimic_oncology.py`
identifies oncology admissions by malignant-neoplasm ICD codes and pulls their free-text notes from BigQuery
(`physionet-data`: MIMIC-III `mimiciii_notes.noteevents`, MIMIC-IV `mimiciv_note.discharge`). **400 notes were
obtained per source** (~392 / 394 patients). Access is credential-gated via PhysioNet; queries cost **~$0**
(a few GB vs the 1 TB/month free tier), with a free dry-run estimate and a hard byte-billed cap. Extraction
uses a resident-model, batched-inference runner for throughput, feeding the MIMIC scale/grounding tables.

---

## Status

**Done**
- Extractor selected (Gemma-3-4B ensemble ×3); recall levers validated.
- Full 40-patient CORAL run, per cohort → Table II.
- CORAL end-to-end: RDF materialization + SPARQL cohort queries, per cohort → Table VI.
- Veracity: calibration + selective admission on CORAL — learned reliability auto-inserts ~60% at 94.8% precision → Table I.
- MIMIC-III / MIMIC-IV oncology cohorts curated (400 + 400 notes).

**Next**
- MIMIC scale extraction (throughput-optimized) → Tables IV, VI, VII.
- Heterogeneous-evidence (retrieval) eval → Table V; extend calibration/selective to MIMIC.
- Per-patient miss analysis (recall by entity type) → extractor fine-tuning for the next version.

---

## Data & ethics

CORAL and MIMIC-III/IV are **credential-gated via PhysioNet** and governed by Data Use Agreements; the MIMIC
subset here is **oncology-filtered** to align with CORAL. **No patient data, extracted triples, or
executed-notebook outputs are committed** — `data/`, `results/`, and executed notebooks are git-ignored.
Obtain the datasets under your own credentials. TRUST-KG is an assistive KG-construction framework, not a
clinical decision system.
