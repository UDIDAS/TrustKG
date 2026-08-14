"""Read MIMIC-III structured data directly from zip.

Same approach as MIMIC-IV: direct zip → temporal KG, no LLM.
MIMIC-III uses different table names and ICD-9 codes (vs ICD-10 in MIMIC-IV).

Key differences from MIMIC-IV:
  - ICD-9 codes (not ICD-10)
  - Table names: UPPERCASE (DIAGNOSES_ICD vs diagnoses_icd)
  - NOTEEVENTS contains clinical notes (2M+ records)
  - Older data (2001-2012)
"""
from __future__ import annotations

import gzip
import logging
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MIMIC3_ZIP = Path("/tmp/ud3d4_mimic/mimic-iii-clinical-database-1.4.zip")


def _find_file_in_zip(zf: zipfile.ZipFile, pattern: str) -> str | None:
    """Find a file in the zip matching a case-insensitive pattern."""
    for f in zf.filelist:
        if pattern.lower() in f.filename.lower() and f.file_size > 0:
            return f.filename
    return None


def read_table(table_name: str, nrows: int | None = None) -> pd.DataFrame:
    """Read a MIMIC-III table directly from zip.

    Args:
        table_name: e.g. "ADMISSIONS", "DIAGNOSES_ICD", "NOTEEVENTS"
    """
    with zipfile.ZipFile(str(MIMIC3_ZIP)) as zf:
        fname = _find_file_in_zip(zf, table_name)
        if fname is None:
            raise FileNotFoundError(f"Table {table_name} not found in {MIMIC3_ZIP}")

        with zf.open(fname) as f:
            if fname.endswith(".gz"):
                with gzip.open(f) as gz:
                    df = pd.read_csv(gz, nrows=nrows, low_memory=False)
            else:
                df = pd.read_csv(f, nrows=nrows, low_memory=False)

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    logger.info("Read MIMIC-III %s: %d rows", table_name, len(df))
    return df


def get_mimic3_stats() -> dict[str, Any]:
    """Get MIMIC-III dataset statistics for paper Table 2."""
    patients = read_table("PATIENTS")
    admissions = read_table("ADMISSIONS")
    diag = read_table("DIAGNOSES_ICD")

    return {
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_diagnoses": len(diag),
        "unique_icd_codes": diag["icd9_code"].nunique() if "icd9_code" in diag.columns else diag["icd_code"].nunique() if "icd_code" in diag.columns else 0,
        "gender_distribution": patients["gender"].value_counts().to_dict() if "gender" in patients.columns else {},
    }


def build_mimic3_temporal_kgs(
    max_patients: int = 2000,
    min_admissions: int = 3,
) -> dict[str, Any]:
    """Build temporal KGs from MIMIC-III structured data.

    Same approach as MIMIC-IV — no LLM, direct structured transform.
    """
    admissions = read_table("ADMISSIONS")
    diag = read_table("DIAGNOSES_ICD")
    procedures = read_table("PROCEDURES_ICD")

    # Normalize column names
    icd_col = "icd9_code" if "icd9_code" in diag.columns else "icd_code"
    proc_icd_col = "icd9_code" if "icd9_code" in procedures.columns else "icd_code"

    # Find longitudinal patients
    adm_counts = admissions.groupby("subject_id").size()
    longitudinal = adm_counts[adm_counts >= min_admissions].index[:max_patients]

    # Filter to longitudinal patients
    adm_f = admissions[admissions["subject_id"].isin(longitudinal)].copy()
    if "admittime" in adm_f.columns:
        adm_f["admittime"] = pd.to_datetime(adm_f["admittime"])
    adm_f = adm_f.sort_values(["subject_id", "admittime"])

    diag_f = diag[diag["subject_id"].isin(longitudinal)]
    proc_f = procedures[procedures["subject_id"].isin(longitudinal)]

    # Build temporal KGs
    import numpy as np
    from collections import defaultdict

    all_stats = []
    for sid in longitudinal:
        pat_adm = adm_f[adm_f["subject_id"] == sid]
        pat_diag = diag_f[diag_f["subject_id"] == sid]
        pat_proc = proc_f[proc_f["subject_id"] == sid]

        cum_diag = set()
        cum_proc = set()
        snapshots = []

        for _, row in pat_adm.iterrows():
            hadm = row.get("hadm_id")
            a_diag = pat_diag[pat_diag["hadm_id"] == hadm]
            new_d = set()
            for _, d in a_diag.iterrows():
                code = str(d.get(icd_col, ""))
                if code not in cum_diag:
                    new_d.add(code)
                cum_diag.add(code)

            a_proc = pat_proc[pat_proc["hadm_id"] == hadm]
            new_p = set()
            for _, p in a_proc.iterrows():
                code = str(p.get(proc_icd_col, ""))
                if code not in cum_proc:
                    new_p.add(code)
                cum_proc.add(code)

            snapshots.append({
                "new_diag": len(new_d),
                "new_proc": len(new_p),
                "cum_size": len(cum_diag) + len(cum_proc),
            })

        if snapshots:
            growth = (snapshots[-1]["cum_size"] - snapshots[0]["cum_size"]) / max(snapshots[0]["cum_size"], 1)
            all_stats.append({
                "n_admissions": len(snapshots),
                "final_kg_size": snapshots[-1]["cum_size"],
                "growth_rate": growth,
                "total_diag": len(cum_diag),
                "total_proc": len(cum_proc),
            })

    if not all_stats:
        return {"n_patients": 0}

    kg_sizes = [s["final_kg_size"] for s in all_stats]

    return {
        "n_patients": len(all_stats),
        "total_admissions": len(adm_f),
        "avg_admissions_per_patient": round(len(adm_f) / len(all_stats), 1),
        "total_relations": sum(s["final_kg_size"] for s in all_stats),
        "avg_relations_per_patient": round(np.mean(kg_sizes), 1),
        "median_kg_size": round(np.median(kg_sizes), 1),
        "avg_growth_rate": round(np.mean([s["growth_rate"] for s in all_stats]), 3),
        "avg_diagnoses": round(np.mean([s["total_diag"] for s in all_stats]), 1),
        "avg_procedures": round(np.mean([s["total_proc"] for s in all_stats]), 1),
        "temporal_coverage": 1.0,
    }
