"""Read clinical narratives from CORAL sources.

Supports:
  - BRCA annotated (.txt files, patients 20-39)
  - PDAC annotated (.txt files, patients 0-19)

.ann.txt files are used ONLY for evaluation (ground truth entity annotations).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.config import CORAL_DIR


@dataclass
class ClinicalDocument:
    patient_id: str
    cohort: str              # "pdac" or "brca"
    source: str              # "annotated"
    text: str
    metadata: dict = field(default_factory=dict)


def _read_cohort(cohort_dir: Path, cohort: str) -> Iterator[ClinicalDocument]:
    """Read .txt clinical narratives from a cohort directory."""
    for txt_file in sorted(cohort_dir.glob("*.txt")):
        if ".ann" in txt_file.name:
            continue
        pid = txt_file.stem
        text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 100:
            continue
        yield ClinicalDocument(
            patient_id=f"{cohort}_{pid}",
            cohort=cohort,
            source="annotated",
            text=text,
            metadata={"coral_idx": int(pid), "file": str(txt_file)},
        )


# ── Public interface ───────────────────────────────────────────
def load_coral_documents(
    cohorts: list[str] | None = None,
    annotated_only: bool = False,
) -> list[ClinicalDocument]:
    """Load clinical documents from CORAL.

    Both PDAC and BRCA have .txt source narratives.
    Args:
        cohorts: subset of ["pdac", "brca"]. None loads both.
        annotated_only: ignored (kept for backward compat).
    """
    cohorts = cohorts or ["pdac", "brca"]
    docs: list[ClinicalDocument] = []

    if "pdac" in cohorts:
        docs.extend(_read_cohort(CORAL_DIR / "pdac", "pdac"))
    if "brca" in cohorts:
        docs.extend(_read_cohort(CORAL_DIR / "breastca", "brca"))

    docs.sort(key=lambda d: d.patient_id)
    return docs


def load_ground_truth(ann_path: Path) -> list[dict]:
    """Parse ENTITY annotations from .ann.txt for evaluation ONLY.

    Reads PROBLEM/TEST/TREATMENT labels — never used as extraction input.
    Returns list of dicts: {label, start, end, text}
    """
    entities: list[dict] = []
    valid_labels = {
        "PROBLEM", "TEST", "TREATMENT", "ClinicalCondition",
        "Observation", "Condition", "Medication", "Procedure",
    }

    for line in ann_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or not line.startswith("T"):
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        tokens = parts[1].split()
        if not tokens:
            continue
        label = tokens[0]
        if label not in valid_labels:
            continue
        offset_str = " ".join(tokens[1:])
        nums = re.findall(r"\d+", offset_str)
        if len(nums) >= 2:
            entities.append({
                "label": label,
                "start": int(nums[0]),
                "end": int(nums[1]),
                "text": parts[2],
            })

    return entities
