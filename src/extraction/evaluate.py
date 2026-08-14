"""Evaluate extracted EAV triples against .ann.txt ground truth.

Evaluation levels:
  1. Entity-level (unique): deduplicate GT mentions, match unique entities
  2. Mention-level: match every GT mention independently
  3. Source grounding: % of extractions traceable to source text

Matching uses multi-strategy: exact substring, normalized containment,
token overlap, and abbreviation expansion.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.reader import load_ground_truth

logger = logging.getLogger(__name__)

# ── Common clinical abbreviations ──────────────────────────────
_ABBREV = {
    "bp": "blood pressure", "hr": "heart rate", "rr": "respiratory rate",
    "bmi": "body mass index", "chf": "congestive heart failure",
    "ckd": "chronic kidney disease", "dm": "diabetes mellitus",
    "htn": "hypertension", "gerd": "gastroesophageal reflux disease",
    "sbo": "small bowel obstruction", "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism", "copd": "chronic obstructive pulmonary disease",
    "er": "estrogen receptor", "pr": "progesterone receptor",
    "her2": "human epidermal growth factor receptor 2",
    "tnbc": "triple negative breast cancer", "idc": "invasive ductal carcinoma",
    "dcis": "ductal carcinoma in situ", "lcis": "lobular carcinoma in situ",
    "wbc": "white blood cell", "rbc": "red blood cell", "hgb": "hemoglobin",
    "hct": "hematocrit", "plt": "platelet", "bun": "blood urea nitrogen",
    "ast": "aspartate transaminase", "alt": "alanine transaminase",
    "alp": "alkaline phosphatase", "ldh": "lactate dehydrogenase",
    "egfr": "estimated glomerular filtration rate",
    "ct": "computed tomography", "mri": "magnetic resonance imaging",
    "pet": "positron emission tomography", "ercp": "endoscopic retrograde cholangiopancreatography",
    "eus": "endoscopic ultrasound", "ekg": "electrocardiogram",
    "lvef": "left ventricular ejection fraction",
    "ac/t": "adriamycin cyclophosphamide taxol", "tc": "taxotere cyclophosphamide",
    "cmf": "cyclophosphamide methotrexate fluorouracil",
}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_abbrev(text: str) -> set[str]:
    """Return set of normalized forms including abbreviation expansions."""
    norm = _normalize(text)
    forms = {norm}
    # Try expanding abbreviation
    if norm in _ABBREV:
        forms.add(_normalize(_ABBREV[norm]))
    # Try matching expanded forms
    for abbr, expansion in _ABBREV.items():
        if norm == _normalize(expansion):
            forms.add(abbr)
    return forms


def _match_score(extracted_text: str, gt_text: str) -> float:
    """Multi-strategy matching score between extracted and GT text.

    Returns score in [0, 1]. Higher = better match.
    Strategies (in order of reliability):
      1. Exact normalized match → 1.0
      2. One contains the other → 0.9
      3. Abbreviation match → 0.85
      4. High token overlap → Jaccard score
      5. Core token containment → fraction of GT tokens found
    """
    norm_ext = _normalize(extracted_text)
    norm_gt = _normalize(gt_text)

    if not norm_ext or not norm_gt:
        return 0.0

    # 1. Exact match
    if norm_ext == norm_gt:
        return 1.0

    # 2. Substring containment (either direction)
    if norm_gt in norm_ext or norm_ext in norm_gt:
        return 0.9

    # 3. Abbreviation expansion match
    ext_forms = _expand_abbrev(extracted_text)
    gt_forms = _expand_abbrev(gt_text)
    if ext_forms & gt_forms:
        return 0.85

    # Check containment with expanded forms
    for ef in ext_forms:
        for gf in gt_forms:
            if gf in ef or ef in gf:
                return 0.85

    # 4. Token overlap (Jaccard)
    tokens_ext = set(norm_ext.split())
    tokens_gt = set(norm_gt.split())
    if tokens_ext and tokens_gt:
        intersection = tokens_ext & tokens_gt
        union = tokens_ext | tokens_gt
        jaccard = len(intersection) / len(union)
        if jaccard >= 0.5:
            return jaccard

    # 5. Core token containment — what fraction of GT tokens appear in extracted
    if tokens_gt:
        containment = len(tokens_ext & tokens_gt) / len(tokens_gt)
        if containment >= 0.6:
            return containment * 0.8  # Discount slightly vs Jaccard

    return 0.0


def _check_source_grounding(extracted_text: str, source_document: str) -> bool:
    """Check if extracted entity/value appears in the source document."""
    norm_ext = _normalize(extracted_text)
    norm_src = _normalize(source_document)

    if norm_ext in norm_src:
        return True

    # Token-level check
    ext_tokens = set(norm_ext.split())
    src_tokens = set(norm_src.split())
    if ext_tokens and len(ext_tokens & src_tokens) / len(ext_tokens) >= 0.6:
        return True

    return False


def _get_triple_texts(triple: dict) -> list[str]:
    """Extract all text fields from a triple for matching."""
    texts = []
    for field in ["entity", "value", "evidence_span"]:
        val = triple.get(field, "")
        if isinstance(val, str) and len(val.strip()) > 1:
            texts.append(val.strip())
    # Also try entity+value combined
    ent = triple.get("entity", "")
    val = triple.get("value", "")
    if ent and val:
        texts.append(f"{ent} {val}")
    return texts


def evaluate_single_model(
    extraction_path: Path,
    ann_path: Path,
    source_text: str,
    patient_id: str,
) -> dict[str, Any]:
    """Evaluate one model's extraction for one patient.

    Reports both entity-level (unique) and mention-level metrics.
    """
    with open(extraction_path) as f:
        extraction = json.load(f)

    triples = extraction.get("triples", [])
    gt_entities = load_ground_truth(ann_path)

    if not triples or not gt_entities:
        return {
            "patient_id": patient_id,
            "num_extracted": len(triples),
            "num_gt_mentions": len(gt_entities),
            "num_gt_unique": len(set(e["text"].lower().strip() for e in gt_entities)),
            "entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0,
            "mention_precision": 0.0, "mention_recall": 0.0, "mention_f1": 0.0,
            "hallucination_rate": 0.0,
            "source_grounded": 0, "not_grounded": 0,
        }

    # Deduplicate GT to unique entities
    gt_unique: dict[str, dict] = {}
    for e in gt_entities:
        key = e["text"].lower().strip()
        if key not in gt_unique:
            gt_unique[key] = e

    # Match triples to GT — a triple can match MULTIPLE GT entities
    # (e.g., evidence_span covering several entities)
    matched_gt_unique: set[str] = set()
    matched_gt_mentions: set[int] = set()
    tp_triples = 0
    grounded = 0
    not_grounded = 0

    for triple in triples:
        triple_texts = _get_triple_texts(triple)

        # Source grounding check
        is_grounded = any(
            _check_source_grounding(t, source_text) for t in triple_texts
        )
        if is_grounded:
            grounded += 1
        else:
            not_grounded += 1

        # Match against ALL GT entities (one triple can cover multiple)
        triple_matched_any = False
        for gt_key, gt_ent in gt_unique.items():
            for t_text in triple_texts:
                if _match_score(t_text, gt_ent["text"]) >= 0.4:
                    matched_gt_unique.add(gt_key)
                    triple_matched_any = True
                    # Mark mention-level matches
                    for i, e in enumerate(gt_entities):
                        if e["text"].lower().strip() == gt_key:
                            matched_gt_mentions.add(i)
                    break  # This GT entity matched, move to next GT

        if triple_matched_any:
            tp_triples += 1

    n_gt_unique = len(gt_unique)
    n_gt_mentions = len(gt_entities)

    # Entity-level metrics (unique)
    e_precision = tp_triples / len(triples) if triples else 0.0
    e_recall = len(matched_gt_unique) / n_gt_unique if n_gt_unique else 0.0
    e_f1 = (2 * e_precision * e_recall / (e_precision + e_recall)
            if (e_precision + e_recall) > 0 else 0.0)

    # Mention-level metrics
    m_precision = tp_triples / len(triples) if triples else 0.0
    m_recall = len(matched_gt_mentions) / n_gt_mentions if n_gt_mentions else 0.0
    m_f1 = (2 * m_precision * m_recall / (m_precision + m_recall)
            if (m_precision + m_recall) > 0 else 0.0)

    hallucination_rate = not_grounded / len(triples) if triples else 0.0

    return {
        "patient_id": patient_id,
        "num_extracted": len(triples),
        "num_gt_mentions": n_gt_mentions,
        "num_gt_unique": n_gt_unique,
        "tp_triples": tp_triples,
        "gt_unique_matched": len(matched_gt_unique),
        "gt_mentions_matched": len(matched_gt_mentions),
        "entity_precision": round(e_precision, 4),
        "entity_recall": round(e_recall, 4),
        "entity_f1": round(e_f1, 4),
        "mention_precision": round(m_precision, 4),
        "mention_recall": round(m_recall, 4),
        "mention_f1": round(m_f1, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "source_grounded": grounded,
        "not_grounded": not_grounded,
    }


def evaluate_pilot(
    extraction_dir: Path,
    coral_dir: Path,
    patient_ids: list[str],
    source_texts: dict[str, str],
) -> dict[str, Any]:
    """Evaluate all models for pilot patients."""
    from src.extraction.local_llm import MODEL_REGISTRY
    results = {}

    # Check all subdirectories for model results
    model_dirs = [d for d in extraction_dir.iterdir() if d.is_dir()]

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_results = []

        for pid in patient_ids:
            extraction_path = model_dir / f"{pid}.json"
            if not extraction_path.exists():
                continue

            cohort = pid.split("_")[0]
            num = pid.split("_")[1]
            ann_path = coral_dir / cohort.replace("brca", "breastca") / f"{num}.ann.txt"
            if not ann_path.exists():
                ann_path = coral_dir / cohort / f"{num}.ann.txt"
            if not ann_path.exists():
                continue

            source = source_texts.get(pid, "")
            metrics = evaluate_single_model(extraction_path, ann_path, source, pid)
            metrics["model"] = model_name
            model_results.append(metrics)

        if model_results:
            n = len(model_results)
            results[model_name] = {
                "patients": model_results,
                "aggregate": {
                    "avg_entity_precision": round(sum(m["entity_precision"] for m in model_results) / n, 4),
                    "avg_entity_recall": round(sum(m["entity_recall"] for m in model_results) / n, 4),
                    "avg_entity_f1": round(sum(m["entity_f1"] for m in model_results) / n, 4),
                    "avg_mention_f1": round(sum(m["mention_f1"] for m in model_results) / n, 4),
                    "avg_hallucination_rate": round(sum(m["hallucination_rate"] for m in model_results) / n, 4),
                    "total_extracted": sum(m["num_extracted"] for m in model_results),
                    "total_gt_unique": sum(m["num_gt_unique"] for m in model_results),
                    "total_gt_mentions": sum(m["num_gt_mentions"] for m in model_results),
                    "num_patients": n,
                },
            }

    return results


def print_evaluation_table(results: dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'Model':<20} {'E-Prec':>8} {'E-Rec':>8} {'E-F1':>8} {'M-F1':>8} {'Halluc%':>8} {'Extr':>6} {'GT-U':>6}")
    print("-" * 84)
    for model_name, data in sorted(results.items()):
        a = data["aggregate"]
        print(
            f"{model_name:<20} "
            f"{a['avg_entity_precision']:>8.3f} "
            f"{a['avg_entity_recall']:>8.3f} "
            f"{a['avg_entity_f1']:>8.3f} "
            f"{a['avg_mention_f1']:>8.3f} "
            f"{a['avg_hallucination_rate']:>8.3f} "
            f"{a['total_extracted']:>6} "
            f"{a['total_gt_unique']:>6}"
        )

    print(f"\n{'Model':<20} {'Patient':<12} {'E-Prec':>8} {'E-Rec':>8} {'E-F1':>8} {'Halluc%':>8} {'Extr':>6} {'GT-U':>6}")
    print("-" * 92)
    for model_name, data in sorted(results.items()):
        for pm in data["patients"]:
            print(
                f"{model_name:<20} "
                f"{pm['patient_id']:<12} "
                f"{pm['entity_precision']:>8.3f} "
                f"{pm['entity_recall']:>8.3f} "
                f"{pm['entity_f1']:>8.3f} "
                f"{pm['hallucination_rate']:>8.3f} "
                f"{pm['num_extracted']:>6} "
                f"{pm['num_gt_unique']:>6}"
            )
