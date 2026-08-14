"""FHIR-guided EAV extraction prompt templates.

Two modes:
  1. Bare extraction (baseline): LLM sees only raw text
  2. Retrieval-augmented (TRUST-KG): LLM sees raw text + NER candidates + ontology context
"""

SYSTEM_PROMPT = """\
You are a clinical information extraction system. Your task is to extract \
structured Entity-Attribute-Value (EAV) triples from unstructured clinical \
narratives. Each triple must be grounded in the text — do not hallucinate \
facts not present in the narrative.

For each extracted triple, provide:
- entity: the biomedical entity (diagnosis, lab test, medication, procedure, etc.)
- attribute: the clinical attribute or property
- value: the extracted value
- fhir_type: one of [Condition, Observation, Procedure, MedicationStatement, \
CarePlan, FamilyMemberHistory, AllergyIntolerance]
- temporal_anchor: any date, time reference, or temporal expression associated \
with this fact (e.g. "09/17/16", "after 3 cycles", "at diagnosis"). \
Use null if no temporal information is present.
- evidence_span: the exact text span from the narrative supporting this triple \
(keep under 150 characters)

Output ONLY valid JSON — an array of objects. No markdown, no explanation."""

# ── Baseline (bare LLM) ───────────────────────────────────────

EAV_EXTRACTION_PROMPT = """\
Extract all clinically relevant Entity-Attribute-Value triples from the \
following oncology clinical narrative. Focus on:

1. **Conditions**: diagnoses, cancer type/subtype, staging (TNM, AJCC), \
grade, histology, comorbidities, symptoms
2. **Observations**: lab results (with values and units), biomarkers \
(ER, PR, HER2, Ki-67, CA19-9), vital signs, imaging findings, pathology results
3. **Procedures**: surgeries, biopsies, imaging studies (CT, PET, MRI), \
ERCP, EUS, with dates and findings
4. **Medications**: chemotherapy regimens, supportive meds, dosages, \
cycles completed, drug combinations
5. **CarePlan**: treatment recommendations, follow-up plans, referrals
6. **FamilyHistory**: family cancer history, genetic predispositions
7. **Allergies**: drug allergies, adverse reactions

For Observations with numeric values, always include the value AND unit.
For temporal anchors, extract the most specific date or time expression available.

--- CLINICAL NARRATIVE ---
{narrative}
--- END NARRATIVE ---

Return a JSON array of EAV triples."""


CHUNKED_EXTRACTION_PROMPT = """\
Continue extracting Entity-Attribute-Value triples from this section of the \
same patient's clinical narrative. Maintain consistency with previously \
extracted entities. Focus on new information in this section.

--- NARRATIVE SECTION ---
{narrative}
--- END SECTION ---

Return a JSON array of EAV triples. If this section contains no clinically \
relevant information, return an empty array []."""

# ── Retrieval-Augmented (TRUST-KG) ────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are a clinical information extraction system with retrieval augmentation. \
You will receive:
1. A clinical narrative
2. A list of biomedical entity candidates detected by NER
3. Ontology context for recognized entities

Your task: produce structured EAV triples for EVERY clinically relevant \
candidate entity. Do NOT skip candidates. If a candidate appears in the \
narrative, extract a triple for it. If a candidate is not clinically \
relevant (e.g. "patient", generic words), skip it.

Each triple must include: entity, attribute, value, fhir_type, \
temporal_anchor, evidence_span.

Output ONLY valid JSON array. No markdown, no explanation."""


RAG_EXTRACTION_PROMPT = """\
Extract EAV triples from the clinical narrative below. You are provided:
1. Entity candidates detected by biomedical NER
2. Ontology-grounded concepts retrieved via BM25 and dense semantic search \
(SNOMED CT, RxNorm, RadLex, NCI codes)
3. Prior graph evidence from already-extracted triples (if any)

Use the ontology concepts to GROUND your extractions in standard \
biomedical terminologies. When an NER candidate matches an ontology \
concept, use the standardized term. You MUST produce a triple for each \
clinically relevant candidate.

--- ENTITY CANDIDATES + ONTOLOGY CONTEXT ---
{candidates}
--- END CANDIDATES ---

--- CLINICAL NARRATIVE ---
{narrative}
--- END NARRATIVE ---

For each candidate entity:
1. Determine the correct FHIR type
2. Extract the attribute and value (use ontology-standard terms where possible)
3. Find the temporal anchor (if any)
4. Provide the evidence span from the text

Return a JSON array of EAV triples. Cover ALL relevant candidates."""


RAG_CHUNKED_PROMPT = """\
Continue extracting EAV triples from this narrative section. \
Use the candidate list to ensure comprehensive coverage. Only extract \
triples for entities that appear in THIS section.

--- ENTITY CANDIDATES ---
{candidates}
--- END CANDIDATES ---

--- NARRATIVE SECTION ---
{narrative}
--- END SECTION ---

Return a JSON array. Empty array [] if no relevant candidates in this section."""
