"""Post-processing FHIR type normalization.

LLMs often generate fine-grained types (Finding, Treatment, string, boolean,
BodyStructure, etc.) instead of the 7 standard FHIR resource types.
This module maps all non-standard types to the correct FHIR category.

Standard FHIR types (from Draft §3.2):
  - Condition: diagnoses, findings, symptoms, comorbidities
  - Observation: lab results, vital signs, biomarkers, imaging findings
  - Procedure: surgeries, biopsies, imaging studies
  - MedicationStatement: drugs, chemotherapy, dosing
  - CarePlan: treatment plans, recommendations
  - FamilyMemberHistory: family cancer history
  - AllergyIntolerance: drug allergies, reactions
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Map every non-standard type the LLM might generate to a standard FHIR type
_FHIR_MAP: dict[str, str] = {
    # Already standard
    "condition": "Condition",
    "observation": "Observation",
    "procedure": "Procedure",
    "medicationstatement": "MedicationStatement",
    "careplan": "CarePlan",
    "familymemberhistory": "FamilyMemberHistory",
    "allergyintolerance": "AllergyIntolerance",

    # → Condition
    "finding": "Condition",
    "diagnosis": "Condition",
    "symptom": "Condition",
    "disease": "Condition",
    "comorbidity": "Condition",
    "clinicalcondition": "Condition",
    "problem": "Condition",
    "disorder": "Condition",
    "injury": "Condition",
    "infection": "Condition",
    "cancer": "Condition",
    "tumor": "Condition",
    "neoplasm": "Condition",
    "syndrome": "Condition",
    "sign": "Condition",

    # → Observation
    "labresult": "Observation",
    "lab_result": "Observation",
    "testresult": "Observation",
    "test_result": "Observation",
    "test": "Observation",
    "vitalsign": "Observation",
    "vital_sign": "Observation",
    "biomarker": "Observation",
    "measurement": "Observation",
    "imagingfinding": "Observation",
    "pathologyfinding": "Observation",
    "physicalexam": "Observation",
    "diagnosticreport": "Observation",
    "string": "Observation",
    "boolean": "Observation",
    "quantity": "Observation",
    "value": "Observation",
    "result": "Observation",
    "score": "Observation",
    "ratio": "Observation",

    # → Procedure
    "surgery": "Procedure",
    "biopsy": "Procedure",
    "imagingstudy": "Procedure",
    "imaging": "Procedure",
    "study": "Procedure",
    "radiologystudy": "Procedure",
    "intervention": "Procedure",
    "operation": "Procedure",
    "excision": "Procedure",
    "resection": "Procedure",
    "servicerequest": "Procedure",

    # → MedicationStatement
    "medication": "MedicationStatement",
    "drug": "MedicationStatement",
    "treatment": "MedicationStatement",
    "chemotherapy": "MedicationStatement",
    "medicationorder": "MedicationStatement",
    "medicationadministration": "MedicationStatement",
    "prescription": "MedicationStatement",
    "regimen": "MedicationStatement",
    "therapy": "MedicationStatement",
    "supportivemed": "MedicationStatement",

    # → CarePlan
    "plan": "CarePlan",
    "treatmentplan": "CarePlan",
    "recommendation": "CarePlan",
    "referral": "CarePlan",
    "goal": "CarePlan",
    "instruction": "CarePlan",
    "followup": "CarePlan",
    "discussiontopic": "CarePlan",

    # → FamilyMemberHistory
    "familyhistory": "FamilyMemberHistory",
    "familymember": "FamilyMemberHistory",
    "geneticmarker": "FamilyMemberHistory",

    # → AllergyIntolerance
    "allergy": "AllergyIntolerance",
    "adversereaction": "AllergyIntolerance",
    "substance": "AllergyIntolerance",

    # Anatomy/body → Condition (as finding context)
    "bodystructure": "Condition",
    "bodypart": "Condition",
    "bodysite": "Condition",
    "anatomy": "Condition",
    "location": "Condition",
    "vascularstructure": "Condition",
    "imagingregion": "Procedure",

    # Administrative / other → best guess
    "patient": "Observation",
    "person": "Observation",
    "encounter": "Procedure",
    "organization": "Observation",
    "practitioner": "Observation",
    "document": "Observation",
    "documentreference": "Observation",
    "narrative": "Observation",
    "note": "Observation",
    "section": "Observation",
    "title": "Observation",
    "identifier": "Observation",
    "date": "Observation",
    "datetime": "Observation",
    "timeperiod": "Observation",
    "duration": "Observation",
    "frequency": "Observation",
    "unit": "Observation",
    "qualifier": "Observation",
    "action": "Procedure",
    "event": "Observation",
    "appointment": "CarePlan",
    "consent": "CarePlan",
    "communication": "CarePlan",
    "socialdeterminant": "Observation",
    "socialhabit": "Observation",
    "ethnicity": "Observation",
    "administrativegender": "Observation",
    "relationship": "FamilyMemberHistory",
    "history": "Condition",

    # Radiology-specific
    "contrastagent": "MedicationStatement",
    "contrastroute": "Procedure",
    "imagingreformat": "Procedure",
    "imagingobject": "Procedure",
    "imagingparameter": "Observation",
    "dosimetry": "Observation",
    "dosimetryunit": "Observation",
    "device": "Procedure",

    # Clinical trial
    "clinicaltrial": "CarePlan",

    # Ontology system names that leak into FHIR type
    "snomed_ct": "Condition",
    "loinc": "Observation",
    "rxnorm": "MedicationStatement",
    "icd-10": "Condition",
    "radlex": "Procedure",
    "nci_thesaurus": "Condition",
}

VALID_FHIR_TYPES = {
    "Condition", "Observation", "Procedure", "MedicationStatement",
    "CarePlan", "FamilyMemberHistory", "AllergyIntolerance",
}


def normalize_fhir_type(raw_type: str) -> str:
    """Map a raw FHIR type string to one of the 7 standard types."""
    if not raw_type:
        return "Observation"

    # Already standard
    if raw_type in VALID_FHIR_TYPES:
        return raw_type

    # Lookup in map (case-insensitive, strip whitespace/underscores)
    key = re.sub(r"[\s_\-]", "", raw_type.lower().strip())
    if key in _FHIR_MAP:
        return _FHIR_MAP[key]

    # Partial match: check if any key is a substring
    for map_key, fhir_type in _FHIR_MAP.items():
        if map_key in key or key in map_key:
            return fhir_type

    # Default fallback
    return "Observation"


def normalize_patient_triples(triples: list[dict]) -> list[dict]:
    """Normalize all FHIR types in a patient's extracted triples.

    Modifies triples in-place and returns them.
    Also stores the original type for audit.
    """
    changed = 0
    for triple in triples:
        raw = str(triple.get("fhir_type", ""))
        normalized = normalize_fhir_type(raw)
        if normalized != raw:
            triple["_fhir_original"] = raw
            triple["fhir_type"] = normalized
            changed += 1

    if changed > 0:
        logger.debug("Normalized %d/%d FHIR types", changed, len(triples))

    return triples


def normalize_batch(all_triples: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Normalize FHIR types for all patients."""
    total_changed = 0
    for pid, triples in all_triples.items():
        before = sum(1 for t in triples if t.get("fhir_type") not in VALID_FHIR_TYPES)
        normalize_patient_triples(triples)
        total_changed += before

    logger.info("FHIR normalization: %d non-standard types corrected across %d patients",
                total_changed, len(all_triples))
    return all_triples


def fhir_distribution(triples: list[dict]) -> dict[str, int]:
    """Get FHIR type distribution (after normalization)."""
    from collections import Counter
    return dict(Counter(t.get("fhir_type", "Unknown") for t in triples).most_common())
