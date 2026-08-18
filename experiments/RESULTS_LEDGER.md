# TRUST-KG — Results Ledger (table-filling worksheet)

> ⚠️ **SUPERSEDED — do not use the numbering or numbers below.** This ledger tracks an older draft
> (9-table scheme, arabic numerals). The **canonical, current-draft-aligned reference is
> [docs/PAPER_TABLES.md](../docs/PAPER_TABLES.md)** (draft's 6 tables, I–VI, with verified values).
> Kept only for history.

Purpose: track every number that must come from a **real, verified run** before the paper is
finalized. Values below are copied from an earlier draft and are **UNVERIFIED / stale** — see
`docs/PAPER_TABLES.md` for the authoritative current numbers.

Status legend: `[ ]` not run · `[~]` run, needs check · `[x]` verified against logged output

> Pipeline has **not been run at scale**. Every cell starts UNVERIFIED.

---

## Pre-run consistency issues found in the current draft (fix regardless of re-run)

- [ ] **Combined entities don't sum.** Table 7: 3,254 + 7,019 + 6,641 = **16,914**, but draft reports
      **14,131** (abstract, intro, Table 7 Combined, discussion). Triples DO sum: 7,704+17,638+15,463 = 40,805 ✓.
      → decide: is 14,131 post-dedup across domains, or an error? Document the reconciliation.
- [ ] **Table 3:** Hybrid Recall@10 (0.996) < MedCPT alone (1.000). Verify fusion/cutoff — adding signals shouldn't drop recall.
- [ ] **Table 5:** "30 CORAL patients" vs CORAL total of 40 (20 BRCA + 20 PDAC). Reconcile with Table 9 (N=20/cohort); state why 10 excluded.
- [ ] **187,462 RDF triples:** no per-domain breakdown anywhere. Add CORAL/MIMIC-III/MIMIC-IV RDF-triple counts that sum to it.
- [ ] **Venue:** filename says *BigData_2026*, page headers say *CIKM 2026, Rome*. Pick one; table/format rules differ.

---

## Which tables the "6–7" aim refers to
Core results tables needing real numbers: **Tables 2, 3, 4, 5, 6, 7, 8** (7 tables).
Table 9 (robustness) is an 8th results table. Table 1 is dataset descriptors (relation counts are the only "results-like" cells).
Headline numbers threaded through abstract/intro/discussion all trace to **Tables 2, 4, 7** — lock those first.

---

## Table 1 — Datasets (descriptive; relation counts need real data stats)
| Dataset | Docs | Patients | Split | Relations (unverified) | Source of number | Verified |
|---|---|---|---|---|---|---|
| CORAL-BRCA | 20 | 20 | 12/4/4 | 4,098 (2-pass, 180–205/pt) | extraction count | [ ] |
| CORAL-PDAC | 20 | 20 | 12/4/4 | 3,606 | extraction count | [ ] |
| MIMIC-III | 400 | 400 | 200/–/200 | 17,638 | extraction count | [ ] |
| MIMIC-III-Ext | 150 | 150 | 80/20/50 | ~6,750 (est.) | **estimate — replace w/ real** | [ ] |
| MIMIC-IV | 400 | 400 | 200/–/200 | 15,463 | extraction count | [ ] |

## Table 2 — Cross-dataset KG construction (CORAL = P/R/F1 vs expert annot; MIMIC = source-grounding)
| Dataset | Method | Prec | Recall | F1 | Unsup | Produced by | Verified |
|---|---|---|---|---|---|---|---|
| CORAL-BRCA | Vanilla RAG | 0.987 | 0.628 | 0.765 | — | baseline run vs gold | [ ] |
| CORAL-BRCA | TRUST-KG 2-pass | 0.969 | 0.879 | 0.922 | 0.000 | 2-pass run vs gold | [ ] |
| CORAL-PDAC | TRUST-KG 2-pass | 0.956 | 0.890 | 0.922 | 0.000 | 2-pass run vs gold | [ ] |
| MIMIC-III | source-grounded filt. | 0.872 (pre-filter) | — | — | 0.000 | grounding check | [ ] |
| MIMIC-IV | filtered | 1.000 (post-filter) | — | — | 0.000 | grounding check | [ ] |
Note: PDAC has no Vanilla RAG baseline row — add one if a transfer claim is made.

## Table 3 — Retrieval grounding, CORAL test (per named-entity mention vs expert annot)
| Method | R@5 | R@10 | MRR | nDCG | EvPrec | OntAcc | Verified |
|---|---|---|---|---|---|---|---|
| BM25 | 0.209 | 0.209 | 0.207 | 0.992 | 0.695 | 1.000 | [ ] |
| MedCPT | 0.996 | 1.000 | 0.874 | 0.897 | 0.188 | 1.000 | [ ] |
| Graph Retrieval | 0.396 | 0.396 | 0.396 | 0.870 | 1.000 | 0.267 | [ ] |
| Hybrid (TRUST-KG) | 0.996 | 0.996 | 0.876 | 0.897 | 0.299 | 0.931 | [ ] |
Check: R@5 == R@10 for BM25/Graph (no new hits between 5–10?); Hybrid R@10 < MedCPT R@10.

## Table 4 — Retrieval ablation, CORAL test
| Config | Prec | Unsup | OntComp | F1 | Verified |
|---|---|---|---|---|---|
| Vanilla RAG | 0.987 | — | 1.000 | 0.765 | [ ] |
| BM25 Only | 0.979 | 0.000 | 1.000 | — | [ ] |
| Dense Only | 0.975 | 0.000 | 1.000 | — | [ ] |
| Graph Only | 0.971 | 0.000 | 1.000 | — | [ ] |
| BM25 + Dense | 0.973 | 0.000 | 1.000 | — | [ ] |
| Dense + Graph | 0.970 | 0.000 | 1.000 | — | [ ] |
| Full Hybrid | 0.969 | 0.000 | 1.000 | 0.922 | [ ] |
Decide: fill intermediate F1 cells (currently "—") or keep blank by design.

## Table 5 — Semantic validation, 30 CORAL patients (mean ± SD)
| Layer | Score | Verified |
|---|---|---|
| Source Grounding | 0.939 ± 0.057 | [ ] |
| Ontology Compliance | 0.779 ± 0.040 | [ ] |
| Schema Validity | 0.678 ± 0.122 | [ ] |
| Temporal Consistency | 0.833 ± 0.105 | [ ] |
| Contradiction Detection | 0.949 ± 0.038 | [ ] |
| Combined Trust Score | 0.827 ± 0.017 | [ ] |
Reconcile "30 patients" with CORAL N=40.

## Table 6 — Filtering effect (unsupported before/after + recall impact)
| Dataset | Unsup Before | Unsup After | Recall Impact | Filter | Verified |
|---|---|---|---|---|---|
| CORAL-BRCA | 0.000 | 0.000 | None | Clean | [ ] |
| CORAL-PDAC | 0.000 | 0.000 | None | Clean | [ ] |
| MIMIC-III | 0.124 | 0.000 | −2.0% | Source | [ ] |
| MIMIC-IV | 0.120 | 0.000 | −2.0% | Source | [ ] |
"Estimated recall impact −2.0%" — document how estimated (no MIMIC gold labels exist).

## Table 7 — Ontology / graph quality (KG triples/entities BEFORE RDF expansion)
| Method | OntComp | InvEdge | SemConsist | KG Triples | KG Entities | SPARQL | Verified |
|---|---|---|---|---|---|---|---|
| Vanilla RAG | 1.000 | 0.000 | 1.000 | – | – | Partial | [ ] |
| TRUST-KG (CORAL) | 1.000 | 0.000 | 1.000 | 7,704 | 3,254 | Yes | [ ] |
| TRUST-KG (MIMIC-III) | 1.000 | 0.000 | 1.000 | 17,638 | 7,019 | Yes | [ ] |
| TRUST-KG (MIMIC-IV) | 1.000 | 0.000 | 1.000 | 15,463 | 6,641 | Yes | [ ] |
| TRUST-KG (Combined) | 1.000 | 0.000 | 1.000 | 40,805 | 14,131 | Yes | [ ] |
Triples sum ✓ (40,805). Entities DON'T (16,914 vs 14,131) — RESOLVE. Also link to 187,462 RDF total (add breakdown).

## Table 8 — SPARQL cohort retrieval
CORAL (40 patients):
| Query | Results | Accuracy | Verified |
|---|---|---|---|
| Breast cancer patients | 19/20 | 95.0% | [ ] |
| Pancreatic cancer patients | 18/20 | 90.0% | [ ] |
| Chemotherapy drug-patient pairs | 31 | – | [ ] |
| Patients with mastectomy | 9 | 100% | [ ] |
| Patients with temporal data | 38/40 | 100% | [ ] |
| Ontology-linked entities | 233 | – | [ ] |
| Cohort precision | – | 92.5% | [ ] |
| Query consistency | 6/6 | – | [ ] |
MIMIC-III (400 patients):
| Query | Results | Verified |
|---|---|---|
| All conditions | 9,109 | [ ] |
| All medications | 2,712 | [ ] |
| All procedures | 3,694 | [ ] |
| Ontology-linked entities | 390 | [ ] |
| Temporal facts | 11,690 | [ ] |
| Unique medical concepts | 7,019 | [ ] |
| Query consistency | 7/7 | [ ] |
Check: MIMIC-III "unique medical concepts" 7,019 == Table 7 MIMIC-III KG entities 7,019 (consistent — keep aligned).

## Table 9 — Patient-level robustness, CORAL (entity-level F1)
| Cohort | N | Mean F1 | SD | 95% CI | Verified |
|---|---|---|---|---|---|
| BRCA All (2-pass) | 20 | 0.911 | 0.049 | [0.888, 0.934] | [ ] |
| BRCA Test (2-pass) | 4 | 0.922 | 0.018 | [0.893, 0.951] | [ ] |
| PDAC All (2-pass) | 20 | 0.930 | 0.031 | [0.916, 0.944] | [ ] |
| PDAC Test (2-pass) | 4 | 0.922 | 0.027 | [0.879, 0.965] | [ ] |
Check: Table 2 F1 (0.922 BRCA/PDAC) are the *test-subset* values here (N=4) — keep them identical.

---

## Inline numbers that must move WITH the tables (update together)
Every occurrence, so nothing is missed when values change:
- **F1 = 0.922** (2-pass) — abstract, intro, §5.1, §5.4, discussion, conclusion  ← from Table 2 / Table 9 test rows
- **0.765 → 0.922, +0.157, relative +20.5%** — abstract, intro, §5.1 (Table 4), discussion, conclusion
- **40,805 triples / 14,131 entities / 187,462 RDF triples / 840 patients** — abstract, intro, §4.1, §5.3 (Table 7), discussion, conclusion
- **87.2% MIMIC-III grounding / 1.000 MIMIC-IV** — abstract-adjacent, §5.1, Table 2
- **~3 min/patient extraction; 3.5 min one-pass; ~7 min two-pass; <10 s NER+retrieval+validation** — §5.4 (Gemma 3 4B, GPU)
- **trust weights β1=.25, β2=.25, β3=.30, β4=.20; δ=0.4** — §3.4, §4.2 (config, not a result, but must match code)

## Runs required to fill each table (once pipeline exists)
- **Extraction (Gemma 3 4B, 1-pass & 2-pass), CORAL BRCA+PDAC vs expert gold** → Tables 2, 4(F1), 5, 9
- **Retrieval eval (BM25 / MedCPT / Graph / Hybrid) on CORAL test** → Tables 3, 4
- **MIMIC-III / MIMIC-IV scale extraction + source-grounding filter** → Tables 2, 6, 7
- **RDF materialization + SPARQL query suite** → Tables 7, 8, 187,462 total
- **Timing harness** → §5.4 latency numbers
Data access prerequisites: CORAL + MIMIC-III/IV are credentialed (PhysioNet DUA); MIMIC-III-Ext annotations.
