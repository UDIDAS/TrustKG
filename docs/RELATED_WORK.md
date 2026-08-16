# TRUST-KG — Related Work & Novelty Positioning

Verified via a multi-source deep-research pass (2026-08-16; 15 primary sources, adversarially
verified). This is internal positioning guidance for the IEEE BigData 2026 submission — a map of
the closest current works, the honest novelty delta, and must-cites. **Verdict: timely and
relevant; lead with the calibration + risk-coverage + tiered admission-control intersection.**

**Note on 2601.01844:** this is *our own prior arXiv preprint* — the preliminary version TRUST-KG
extends. Extending your own preprint into a full venue paper is standard (arXiv is NOT "prior
publication" for IEEE/CS venues), so it is a **self-citation and foundation, not a novelty risk**.
Cite it as the preliminary version and foreground the new contributions. The works to actually
*differentiate against* are the independent ones (DIAL-KG, MedKGent, SAC-KG, SelectLLM, Infherno).

## Must-cite works

| Work (arXiv / venue) | What it does | Bounds our novelty because… |
|---|---|---|
| **2601.01844** (Jan 2026, RAG multi-LLM clinical KG) | Builds clinical KGs on the **same CORAL PDAC/BRCA** data, **same EAV** formulation, multi-LLM consensus, entropy trust, composite trust T(τ)=λ₁R+λ₂C+λ₃J @ δ=0.65, ontology (SNOMED/LOINC/RxNorm/ICD/GO), FHIR templates, OWL/RDFS checks, Grok-3 contradiction | **Our own preliminary arXiv preprint** — TRUST-KG is its extended version. Self-cite as prior work / foundation; foreground the new contributions (below). Not a novelty risk. |
| **DIAL-KG** (2603.20059, Mar 2026) | verify-before-insert "Governance Adjudication" (Evidence + Logical + Evolutionary-Intent) filtering facts BEFORE KG insertion | Owns "construction-time verification" as a named, active direction. |
| **MedKGent** (2508.12393, npj Digital Medicine 2025) | 2-agent LLM KG construction; sampling-based confidence; temporal graph | LLM-KG-construction SOTA w/ confidence scoring. |
| **SAC-KG** (2410.02811, ACL 2024) | Generator–Verifier–Pruner; rule-based error correction | Dedicated verification stage precedent. |
| **SelectLLM** (NeurIPS 2025) | jointly-trained selection head; risk-coverage optimization; calibrated abstention | Our selective-prediction machinery is prior art. |
| **AUGRC** (Traub et al., NeurIPS 2024) | shows AURC's flaws; proposes AUGRC (avg risk of undetected failures) | The correct **evaluation apparatus** for our risk-coverage claim. |
| **Infherno** (2507.12261, EACL 2026) | agent note→HL7 FHIR synthesis; schema/terminology grounding | Closest FHIR-grounded clinical IE neighbor (binary valid/invalid, not calibrated). |
| Surveys: **2510.20345** (LLM-KG construction taxonomy), **2503.05777** (medical hallucination) | Taxonomy has NO category for calibrated admission control / selective prediction; hallucination survey establishes grounding/provenance/calibration rationale | Support paradigm-novelty framing + timeliness (but the rationale itself is prior art). |
| Also surfaced: **GraphMERT** (2510.09580), **SHARP** (2604.04190) | tiny-model KG distillation; agentic triple verification | Adjacent triple-verification methods. |

## Genuine, defensible novelty (lead with this)

No cited competitor combines **all** of:
1. probability **calibration (ECE/Brier/NLL)** +
2. **risk–coverage** selective prediction (AURC/Coverage@95%, ideally AUGRC) +
3. a **tiered Insert / Review / Reject** admission policy at a **tunable quality–coverage operating point**, applied **per-triple BEFORE materialization** +
4. a **sub-5B open-weight ensemble** (vs the predecessor's frontier APIs) +
5. a **MIMIC-III/IV extension** (the predecessor is CORAL-only) +
6. **hybrid BM25 + MedCPT + graph-neighborhood** construction-time retrieval.

The **Big-Data data-veracity / ingestion-governance** framing is timely and defensible; the
**Table I** calibration+selective-admission result (E1-E2) is our strongest, least-contested claim.

## What the paper should do

1. **Position 2601.01844 as the preliminary version and make the new contributions the headline.** Extending your own preprint is standard and legitimate; just foreground the deltas (calibration + Insert/Review/Reject + sub-5B + MIMIC + hybrid retrieval) so the venue paper's added contribution is unmistakable — since arXiv is public, a reviewer may pull the preprint up.
2. **Report a genuine calibrated risk–coverage curve** with a tunable operating point (not a single fixed threshold like the independent competitors). Prefer **AUGRC**, or justify AURC/ECE. This is exactly what Table I does — foreground it.
3. Ideally show **construction-time admission control yields downstream SPARQL/query benefits** over retrieval-time (GraphRAG) filtering.
4. **Differentiate against the independent works** (DIAL-KG, MedKGent, SAC-KG, SelectLLM, Infherno) — that's where related-work positioning matters, not the self-citation.

## Caveats
- Several sources are 2025–2026 preprints; 2601.01844 will still be fresh at review time.
- The ECE/"Evidence Chain Evaluation" fact-check preprint (2607.18240) is unreviewed, tiny (95 claims) — cite only as illustrative of the calibrated-abstain direction.
- Per project convention, keep author-identifying phrasing out of the paper text; 2601.01844 is cited as ordinary prior work.
