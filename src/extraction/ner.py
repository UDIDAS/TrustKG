"""Clinical NER and section detection using medspacy/scispacy.

Extracts candidate biomedical entity mentions from clinical narratives.
These candidates are then passed to the ontology retrieval and LLM
structuring stages of the TRUST-KG pipeline.

Uses en_core_sci_lg for entity detection and medspacy for section/context.
CPU parallelism via ThreadPoolExecutor for batch processing.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import spacy

from src.config import MAX_WORKERS

logger = logging.getLogger(__name__)

_nlp = None


def _get_nlp():
    """Lazy-load scispacy pipeline (CPU only, shared across threads)."""
    global _nlp
    if _nlp is None:
        logger.info("Loading en_core_sci_lg NER pipeline...")
        _nlp = spacy.load("en_core_sci_lg")
        # Disable components we don't need for speed
        _nlp.disable_pipes("lemmatizer")
        logger.info("NER pipeline loaded (components: %s)", _nlp.pipe_names)
    return _nlp


@dataclass
class EntityMention:
    text: str
    start: int
    end: int
    label: str
    context: str       # surrounding sentence
    section: str       # clinical section (HPI, Labs, Meds, etc.)
    category: str      # PROBLEM/TEST/TREATMENT/OTHER (heuristic)


def _detect_section(text: str, char_offset: int) -> str:
    """Detect clinical section for a character offset using regex patterns."""
    section_patterns = [
        (r"(?i)\b(history of present illness|hpi)\b", "HPI"),
        (r"(?i)\b(assessment|impression|plan)\b", "ASSESSMENT"),
        (r"(?i)\b(past medical history|pmh)\b", "PMH"),
        (r"(?i)\b(past surgical history)\b", "PSH"),
        (r"(?i)\b(family history)\b", "FAMILY_HISTORY"),
        (r"(?i)\b(social history)\b", "SOCIAL_HISTORY"),
        (r"(?i)\b(medications?|current medications?|prescriptions?)\b", "MEDICATIONS"),
        (r"(?i)\b(allergies?|allergy)\b", "ALLERGIES"),
        (r"(?i)\b(laboratory|lab results?|labs)\b", "LABS"),
        (r"(?i)\b(pathology|pathologic|surgical pathology)\b", "PATHOLOGY"),
        (r"(?i)\b(imaging|radiology|ct |mri |pet )", "IMAGING"),
        (r"(?i)\b(physical exam|vital signs|review of systems)\b", "EXAM"),
        (r"(?i)\b(recommendations?|treatment plan)\b", "PLAN"),
    ]

    # Find the last section header before this offset
    best_section = "UNKNOWN"
    best_pos = -1

    for pattern, section_name in section_patterns:
        for match in re.finditer(pattern, text[:char_offset]):
            if match.start() > best_pos:
                best_pos = match.start()
                best_section = section_name

    return best_section


def _categorize_entity(text: str, section: str) -> str:
    """Heuristic categorization into PROBLEM/TEST/TREATMENT/OTHER.

    Uses section context + keyword patterns to approximate the annotation
    categories in the CORAL ground truth.
    """
    text_lower = text.lower().strip()

    # TREATMENT indicators
    treatment_kw = [
        "therapy", "treatment", "chemotherapy", "radiation", "surgery",
        "mastectomy", "lumpectomy", "biopsy", "resection", "excision",
        "regimen", "cycle", "dose", "infusion", "stent", "drain",
    ]
    # Common drug suffixes
    drug_suffixes = [
        "mab", "nib", "ine", "cin", "ole", "ide", "fen", "pam",
        "tin", "xel", "sol", "pril", "tan", "lol", "azole",
    ]
    if any(kw in text_lower for kw in treatment_kw):
        return "TREATMENT"
    if section == "MEDICATIONS":
        return "TREATMENT"
    if any(text_lower.endswith(s) for s in drug_suffixes) and len(text_lower) > 4:
        return "TREATMENT"

    # TEST indicators
    test_kw = [
        "test", "scan", "ct ", "mri", "pet", "ultrasound", "mammogram",
        "biopsy", "aspiration", "echocardiogram", "ekg", "ercp", "eus",
        "lab", "level", "count", "marker", "stain", "assay", "panel",
    ]
    biomarkers = [
        "er ", "pr ", "her2", "ki-67", "ki67", "ca19-9", "ca 19-9",
        "ca125", "brca", "egfr", "alk", "pdl1", "pd-l1", "cea",
        "hemoglobin", "hematocrit", "wbc", "platelet", "creatinine",
        "bilirubin", "albumin", "glucose", "sodium", "potassium",
    ]
    if any(kw in text_lower for kw in test_kw):
        return "TEST"
    if any(bm in text_lower for bm in biomarkers):
        return "TEST"
    if section in ("LABS", "IMAGING", "PATHOLOGY"):
        return "TEST"

    # PROBLEM indicators
    problem_kw = [
        "cancer", "carcinoma", "tumor", "mass", "lesion", "metasta",
        "malignant", "disease", "syndrome", "disorder", "failure",
        "obstruction", "infection", "pain", "nausea", "vomiting",
        "dyspnea", "edema", "bleeding", "necrosis", "stenosis",
        "hypertension", "diabetes", "anemia", "thrombosis",
        "positive", "negative", "elevated", "decreased", "abnormal",
        "stage", "grade", "invasive", "recurrent", "metastatic",
    ]
    if any(kw in text_lower for kw in problem_kw):
        return "PROBLEM"
    if section in ("HPI", "ASSESSMENT", "PMH"):
        return "PROBLEM"

    return "OTHER"


def extract_entities(text: str) -> list[EntityMention]:
    """Extract biomedical entity mentions from clinical text.

    Uses scispacy en_core_sci_lg for NER, then enriches with
    section detection and heuristic categorization.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    mentions: list[EntityMention] = []
    seen_spans: set[tuple[int, int]] = set()

    for ent in doc.ents:
        # Skip very short or very long entities
        if len(ent.text.strip()) < 2 or len(ent.text) > 200:
            continue
        # Skip generic entities
        if ent.text.lower().strip() in {
            "patient", "patients", "history", "results", "report",
            "date", "time", "status", "findings", "conclusion",
        }:
            continue
        # Deduplicate by span
        span_key = (ent.start_char, ent.end_char)
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)

        # Get sentence context
        sent = ent.sent if ent.sent else doc[max(0, ent.start - 10):ent.end + 10]
        context = sent.text[:200]

        section = _detect_section(text, ent.start_char)
        category = _categorize_entity(ent.text, section)

        mentions.append(EntityMention(
            text=ent.text.strip(),
            start=ent.start_char,
            end=ent.end_char,
            label=ent.label_,
            context=context,
            section=section,
            category=category,
        ))

    return mentions


def extract_entities_batch(
    texts: list[str],
) -> list[list[EntityMention]]:
    """Extract entities from multiple texts using CPU ThreadPool."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(extract_entities, texts))
    return results


def group_candidates(mentions: list[EntityMention]) -> dict[str, list[EntityMention]]:
    """Group entity mentions by category for structured candidate lists."""
    groups: dict[str, list[EntityMention]] = defaultdict(list)
    for m in mentions:
        groups[m.category].append(m)
    return dict(groups)


def format_candidate_list(mentions: list[EntityMention], max_candidates: int | None = None) -> str:
    """Format entity mentions as a structured candidate list for LLM prompts.

    Groups by category and deduplicates by normalized text.
    Caps total candidates to max_candidates to prevent prompt overflow / OOM.
    """
    from src.config import MAX_CANDIDATES_PER_CHUNK
    if max_candidates is None:
        max_candidates = MAX_CANDIDATES_PER_CHUNK

    groups = group_candidates(mentions)
    seen: set[str] = set()
    lines: list[str] = []

    total_added = 0
    for category in ["PROBLEM", "TEST", "TREATMENT", "OTHER"]:
        if category not in groups:
            continue
        lines.append(f"\n## {category} candidates:")
        for m in groups[category]:
            if total_added >= max_candidates:
                break
            norm = m.text.lower().strip()
            if norm in seen:
                continue
            seen.add(norm)
            total_added += 1
            lines.append(f'  - "{m.text}" (section: {m.section})')

    return "\n".join(lines)
