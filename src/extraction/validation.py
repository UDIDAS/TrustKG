"""5-layer validation pipeline for extracted EAV triples.

Layers (ordered by strictness):
  1. Source Grounding: entity/value must appear in original text
  2. Ontology Check: entity maps to SNOMED/LOINC/RxNorm/ICD code
  3. Schema Check: relation valid for FHIR types
  4. Temporal Consistency: dates plausible, no future dates, chronological
  5. Contradiction Detection: conflicting values at same timepoint

Each layer produces a validation score. Combined into a trust score:
  T(τ) = β₁·grounding + β₂·ontology + β₃·schema + β₄·temporal - β₅·contradiction

Triples below threshold δ are rejected.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from src.extraction.evaluate import _normalize, _match_score, _check_source_grounding
from src.config import MAX_WORKERS

logger = logging.getLogger(__name__)

# ── FHIR schema: valid (fhir_type, attribute) pairs ───────────
_VALID_FHIR_ATTRS = {
    "Condition": {
        "diagnosis", "grade", "stage", "staging", "tnm", "ajcc",
        "histology", "type", "subtype", "laterality", "site", "location",
        "status", "severity", "onset", "er status", "pr status",
        "her2 status", "ki-67", "ki67", "biomarker", "tumor size",
        "lymph node status", "metastasis", "invasion", "margin",
        "comorbidity", "symptom", "sign", "finding", "impression",
    },
    "Observation": {
        "value", "result", "level", "count", "score", "ratio",
        "measurement", "vital sign", "lab result", "imaging finding",
        "pathology finding", "test result", "interpretation",
    },
    "Procedure": {
        "type", "date", "site", "laterality", "outcome", "finding",
        "technique", "indication", "complication", "status",
    },
    "MedicationStatement": {
        "medication", "drug", "dose", "dosage", "route", "frequency",
        "regimen", "cycles", "duration", "status", "indication",
        "discontinuation", "adverse effect",
    },
    "CarePlan": {
        "recommendation", "plan", "referral", "follow-up", "goal",
        "instruction", "activity",
    },
    "FamilyMemberHistory": {
        "condition", "relation", "age of onset", "status",
    },
    "AllergyIntolerance": {
        "allergen", "reaction", "severity", "type",
    },
}


def _layer1_source_grounding(triple: dict, source_text: str) -> float:
    """Check if entity and value appear in source document. Returns 0 or 1."""
    entity = str(triple.get("entity", ""))
    value = str(triple.get("value", ""))
    evidence = str(triple.get("evidence_span", ""))

    checks = []
    if entity and len(entity) > 2:
        checks.append(_check_source_grounding(entity, source_text))
    if value and len(value) > 2:
        checks.append(_check_source_grounding(value, source_text))
    if evidence and len(evidence) > 5:
        checks.append(_check_source_grounding(evidence, source_text))

    if not checks:
        return 0.5  # Can't verify
    return sum(checks) / len(checks)


def _layer2_ontology_check(triple: dict) -> float:
    """Check if entity maps to a known biomedical concept.

    Uses heuristic matching against common biomedical terms.
    Full ontology API lookup is done separately in ontology_normalizer.
    Returns 0-1 score.
    """
    entity = _normalize(str(triple.get("entity", "")))
    fhir_type = str(triple.get("fhir_type", ""))

    # Known biomedical term lists (fast heuristic)
    known_conditions = {
        "cancer", "carcinoma", "tumor", "adenocarcinoma", "lymphoma",
        "leukemia", "melanoma", "sarcoma", "diabetes", "hypertension",
        "heart failure", "kidney disease", "copd", "asthma", "stroke",
        "infection", "sepsis", "anemia", "thrombosis", "fibrosis",
        "metastasis", "necrosis", "stenosis", "obstruction", "fracture",
    }
    known_labs = {
        "hemoglobin", "hematocrit", "wbc", "platelet", "creatinine",
        "bilirubin", "albumin", "glucose", "sodium", "potassium",
        "calcium", "ast", "alt", "alkaline phosphatase", "ldh",
        "ca19-9", "ca 19-9", "cea", "afp", "psa", "her2", "er",
        "pr", "ki-67", "ki67", "egfr", "brca", "tp53",
    }
    known_meds = {
        "gemcitabine", "abraxane", "paclitaxel", "docetaxel",
        "carboplatin", "cisplatin", "doxorubicin", "cyclophosphamide",
        "fluorouracil", "capecitabine", "trastuzumab", "pembrolizumab",
        "nivolumab", "tamoxifen", "anastrozole", "letrozole",
        "metformin", "insulin", "aspirin", "atorvastatin", "lisinopril",
        "amlodipine", "metoprolol", "furosemide", "omeprazole",
    }
    known_procedures = {
        "mastectomy", "lumpectomy", "biopsy", "resection", "excision",
        "ct", "mri", "pet", "ultrasound", "mammogram", "endoscopy",
        "ercp", "eus", "colonoscopy", "bronchoscopy", "surgery",
        "radiation", "chemotherapy", "immunotherapy", "transplant",
    }

    all_known = known_conditions | known_labs | known_meds | known_procedures
    entity_tokens = set(entity.split())

    if entity in all_known or entity_tokens & all_known:
        return 1.0
    if fhir_type in _VALID_FHIR_ATTRS:
        return 0.7  # Valid FHIR type but entity not in our list
    return 0.3


def _layer3_schema_check(triple: dict) -> float:
    """Check if attribute is valid for the FHIR type. Returns 0 or 1."""
    fhir_type = str(triple.get("fhir_type", ""))
    attribute = _normalize(str(triple.get("attribute", "")))

    if fhir_type not in _VALID_FHIR_ATTRS:
        return 0.5  # Unknown type — can't validate

    valid_attrs = _VALID_FHIR_ATTRS[fhir_type]
    # Check if attribute matches any valid attribute (fuzzy)
    for va in valid_attrs:
        if va in attribute or attribute in va:
            return 1.0
        if _match_score(attribute, va) >= 0.5:
            return 0.8
    return 0.3  # No match — suspicious but not definitive


def _parse_date(text: str) -> datetime | None:
    """Try parsing common clinical date formats."""
    for fmt in [
        "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m/%d/%Y",
        "%B %d, %Y", "%b %d, %Y", "%m-%d-%Y", "%m-%d-%y",
    ]:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def _layer4_temporal_consistency(triple: dict) -> float:
    """Check temporal plausibility. Returns 0-1."""
    anchor = str(triple.get("temporal_anchor", ""))
    if not anchor or anchor.lower() in ("null", "none", "n/a", ""):
        return 0.7  # No temporal info — neutral

    dt = _parse_date(anchor)
    if dt is None:
        return 0.6  # Unparseable date — likely relative ("after 3 cycles")

    # Check plausibility
    now = datetime.now()
    if dt.year < 1900 or dt > now:
        return 0.1  # Future date or implausible past
    if dt.year < 1980:
        return 0.3  # Suspiciously old for clinical data
    return 1.0


def _layer5_contradiction_detection(
    triple: dict, all_triples: list[dict]
) -> float:
    """Check for contradictions with other triples. Returns 0 (contradiction) to 1 (clean)."""
    entity = _normalize(str(triple.get("entity", "")))
    attr = _normalize(str(triple.get("attribute", "")))
    value = _normalize(str(triple.get("value", "")))
    anchor = str(triple.get("temporal_anchor", ""))

    for other in all_triples:
        if other is triple:
            continue
        o_entity = _normalize(str(other.get("entity", "")))
        o_attr = _normalize(str(other.get("attribute", "")))
        o_value = _normalize(str(other.get("value", "")))
        o_anchor = str(other.get("temporal_anchor", ""))

        # Same entity + same attribute but different value at same time
        if (_match_score(entity, o_entity) >= 0.6 and
            _match_score(attr, o_attr) >= 0.6 and
            value != o_value and
            anchor == o_anchor and anchor):
            # Potential contradiction — but some are valid (e.g., multiple symptoms)
            # Only flag binary contradictions (positive/negative)
            binary_pairs = [
                ("positive", "negative"), ("yes", "no"),
                ("present", "absent"), ("normal", "abnormal"),
            ]
            for pos, neg in binary_pairs:
                if (pos in value and neg in o_value) or (neg in value and pos in o_value):
                    return 0.1  # Clear contradiction
            return 0.5  # Suspicious but not definitive

    return 1.0


def validate_triple(
    triple: dict,
    source_text: str,
    all_triples: list[dict],
    weights: tuple[float, ...] = (0.30, 0.20, 0.15, 0.15, 0.20),
) -> dict[str, Any]:
    """Run all 5 validation layers on a single triple.

    Returns the triple enriched with validation scores and overall trust score.
    """
    β1, β2, β3, β4, β5 = weights

    s1 = _layer1_source_grounding(triple, source_text)
    s2 = _layer2_ontology_check(triple)
    s3 = _layer3_schema_check(triple)
    s4 = _layer4_temporal_consistency(triple)
    s5 = _layer5_contradiction_detection(triple, all_triples)

    trust = β1 * s1 + β2 * s2 + β3 * s3 + β4 * s4 + β5 * s5

    return {
        **triple,
        "_validation": {
            "source_grounding": round(s1, 3),
            "ontology_check": round(s2, 3),
            "schema_check": round(s3, 3),
            "temporal_consistency": round(s4, 3),
            "contradiction_score": round(s5, 3),
            "trust_score": round(trust, 4),
        },
    }


def validate_patient_triples(
    triples: list[dict],
    source_text: str,
    trust_threshold: float = 0.4,
) -> dict[str, Any]:
    """Validate all triples for one patient.

    Returns accepted triples, rejected triples, and validation stats.
    """
    validated = [
        validate_triple(t, source_text, triples) for t in triples
    ]

    accepted = [t for t in validated if t["_validation"]["trust_score"] >= trust_threshold]
    rejected = [t for t in validated if t["_validation"]["trust_score"] < trust_threshold]

    # Compute per-layer stats
    layer_names = [
        "source_grounding", "ontology_check", "schema_check",
        "temporal_consistency", "contradiction_score",
    ]
    layer_stats = {}
    for layer in layer_names:
        scores = [t["_validation"][layer] for t in validated]
        layer_stats[layer] = {
            "mean": round(sum(scores) / max(len(scores), 1), 3),
            "min": round(min(scores) if scores else 0, 3),
            "below_threshold": sum(1 for s in scores if s < 0.5),
        }

    trust_scores = [t["_validation"]["trust_score"] for t in validated]

    return {
        "accepted": accepted,
        "rejected": rejected,
        "stats": {
            "total_input": len(triples),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "acceptance_rate": round(len(accepted) / max(len(triples), 1), 3),
            "mean_trust": round(sum(trust_scores) / max(len(trust_scores), 1), 4),
            "layer_stats": layer_stats,
        },
    }
