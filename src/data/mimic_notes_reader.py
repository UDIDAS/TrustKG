"""Read MIMIC-IV-Note discharge summaries for TRUST-KG extraction.

MIMIC-IV-Note contains 331K discharge summaries. We process a curated
subset for cross-domain evaluation:
  - MIMIC-Onc: oncology patients (filtered by ICD cancer codes)
  - MIMIC-ICU: ICU patients (for domain diversity)
  - MIMIC-Temporal: patients with 3+ admissions (temporal evolution)

Reads directly from zip. Processes in batches. Results pushed to GDrive.

The structured MIMIC data serves as ground truth for evaluating
LLM extraction quality (compare extracted KG vs coded diagnoses).
"""
from __future__ import annotations

import gzip
import logging
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.data.reader import ClinicalDocument

logger = logging.getLogger(__name__)


def _find_notes_zip() -> Path | None:
    """Find the MIMIC-IV-Note zip file."""
    candidates = [
        Path("/tmp/ud3d4_mimic/mimic-iv-note-deidentified-free-text-clinical-notes-2.2.zip"),
        Path("/tmp/ud3d4_mimic/mimic-iv-note-2.2.zip"),
    ]
    # Also check for any note-related zip
    mimic_dir = Path("/tmp/ud3d4_mimic")
    if mimic_dir.exists():
        for f in mimic_dir.iterdir():
            if "note" in f.name.lower() and f.suffix == ".zip":
                return f

    for c in candidates:
        if c.exists():
            return c
    return None


def _read_notes_from_zip(
    notes_zip: Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read discharge summaries from MIMIC-IV-Note zip."""
    with zipfile.ZipFile(str(notes_zip)) as zf:
        # Find the discharge file
        discharge_files = [
            f.filename for f in zf.filelist
            if "discharge" in f.filename.lower() and f.filename.endswith(".csv.gz")
        ]
        if not discharge_files:
            # Try plain csv
            discharge_files = [
                f.filename for f in zf.filelist
                if "discharge" in f.filename.lower() and f.filename.endswith(".csv")
            ]

        if not discharge_files:
            raise FileNotFoundError(
                f"No discharge file found in {notes_zip}. "
                f"Contents: {[f.filename for f in zf.filelist[:20]]}"
            )

        fname = discharge_files[0]
        logger.info("Reading %s from %s", fname, notes_zip.name)

        with zf.open(fname) as f:
            if fname.endswith(".gz"):
                with gzip.open(f) as gz:
                    df = pd.read_csv(gz, nrows=nrows, low_memory=False)
            else:
                df = pd.read_csv(f, nrows=nrows, low_memory=False)

    logger.info("Loaded %d discharge summaries", len(df))
    return df


def load_mimic_notes_subset(
    subset: str = "oncology",
    max_patients: int = 500,
    max_notes: int = 1000,
) -> list[ClinicalDocument]:
    """Load a curated subset of MIMIC notes as ClinicalDocuments.

    Subsets:
      "oncology": patients with cancer ICD codes
      "icu": ICU patients
      "temporal": patients with 3+ discharge summaries (longitudinal)
    """
    notes_zip = _find_notes_zip()
    if notes_zip is None:
        logger.warning("MIMIC-IV-Note zip not found. Run poll_mimic_notes.sh or download manually.")
        return []

    # Load notes
    notes_df = _read_notes_from_zip(notes_zip)

    # Identify text column (varies by version)
    text_col = None
    for col in ["text", "note", "discharge_text", "TEXT"]:
        if col in notes_df.columns:
            text_col = col
            break
    if text_col is None:
        # Try the longest string column
        str_cols = notes_df.select_dtypes(include="object").columns
        if len(str_cols) > 0:
            text_col = max(str_cols, key=lambda c: notes_df[c].str.len().mean())
    if text_col is None:
        raise ValueError(f"Cannot find text column. Columns: {notes_df.columns.tolist()}")

    # Filter by subset
    if subset == "oncology":
        # Load diagnoses to find cancer patients
        import gzip as gz_mod
        with zipfile.ZipFile(str(Path("/tmp/ud3d4_mimic/mimic-iv-3.1.zip"))) as zf:
            with zf.open("mimic-iv-3.1/hosp/diagnoses_icd.csv.gz") as f:
                with gz_mod.open(f) as g:
                    diag = pd.read_csv(g, low_memory=False)
        cancer = diag[diag["icd_code"].str.match(r"^C\d", na=False) & (diag["icd_version"] == 10)]
        onc_subjects = set(cancer["subject_id"].unique()[:max_patients])
        notes_df = notes_df[notes_df["subject_id"].isin(onc_subjects)]

    elif subset == "icu":
        with zipfile.ZipFile(str(Path("/tmp/ud3d4_mimic/mimic-iv-3.1.zip"))) as zf:
            with zf.open("mimic-iv-3.1/icu/icustays.csv.gz") as f:
                with gzip.open(f) as g:
                    icu = pd.read_csv(g, low_memory=False)
        icu_subjects = set(icu["subject_id"].unique()[:max_patients])
        notes_df = notes_df[notes_df["subject_id"].isin(icu_subjects)]

    elif subset == "temporal":
        multi = notes_df.groupby("subject_id").size()
        temporal_subjects = set(multi[multi >= 3].index[:max_patients])
        notes_df = notes_df[notes_df["subject_id"].isin(temporal_subjects)]

    # Limit total notes
    notes_df = notes_df.head(max_notes)

    # Convert to ClinicalDocuments
    docs = []
    for _, row in notes_df.iterrows():
        text = str(row[text_col])
        if len(text) < 100:
            continue
        sid = int(row["subject_id"])
        hadm_id = int(row.get("hadm_id", 0)) if pd.notna(row.get("hadm_id")) else 0

        docs.append(ClinicalDocument(
            patient_id=f"mimic_{sid}_{hadm_id}",
            cohort="mimic",
            source=f"mimic_note_{subset}",
            text=text,
            metadata={
                "subject_id": sid,
                "hadm_id": hadm_id,
                "charttime": str(row.get("charttime", "")),
                "storetime": str(row.get("storetime", "")),
            },
        ))

    logger.info("Loaded %d MIMIC %s notes (%d patients)", len(docs), subset, len(set(d.metadata["subject_id"] for d in docs)))
    return docs


def check_notes_available() -> bool:
    """Check if MIMIC-IV-Note zip is available."""
    return _find_notes_zip() is not None
