"""Build knowledge graphs from MIMIC-IV structured data.

Constructs KGs from structured EHR tables (diagnoses, procedures,
admissions) for:
  1. Cross-domain evaluation (compare structured vs LLM-extracted KGs)
  2. Oncology cohort KG for downstream QA
  3. Temporal KG from admission timelines
  4. Ground truth reference for validation

Memory-efficient: reads from zip, processes one patient at a time.
"""
from __future__ import annotations

import gzip
import json
import logging
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MAX_WORKERS

logger = logging.getLogger(__name__)

MIMIC_ZIP = Path("/tmp/ud3d4_mimic/mimic-iv-3.1.zip")


def _read_table(table_name: str, nrows: int | None = None) -> pd.DataFrame:
    """Read table from MIMIC zip without extraction."""
    with zipfile.ZipFile(str(MIMIC_ZIP), "r") as zf:
        path = f"mimic-iv-3.1/{table_name}.csv.gz"
        with zf.open(path) as f:
            with gzip.open(f) as gz:
                return pd.read_csv(gz, nrows=nrows, low_memory=False)


# Cache diagnosis code descriptions (small table, loaded once)
_diag_desc: dict[str, str] = {}


def _get_diag_descriptions() -> dict[str, str]:
    global _diag_desc
    if not _diag_desc:
        df = _read_table("hosp/d_icd_diagnoses")
        for _, row in df.iterrows():
            key = f"{row['icd_code']}_{row['icd_version']}"
            _diag_desc[key] = str(row.get("long_title", row["icd_code"]))
    return _diag_desc


def build_patient_kg(
    subject_id: int,
    diagnoses: pd.DataFrame,
    procedures: pd.DataFrame,
    admissions: pd.DataFrame,
) -> dict[str, Any]:
    """Build a structured KG for one MIMIC patient.

    Returns dict with triples, temporal info, and stats.
    """
    desc = _get_diag_descriptions()
    triples = []

    # Diagnosis triples
    pat_diag = diagnoses[diagnoses["subject_id"] == subject_id]
    seen_diag = set()
    for _, row in pat_diag.iterrows():
        code = str(row["icd_code"])
        ver = int(row.get("icd_version", 10))
        key = f"{code}_{ver}"
        if key in seen_diag:
            continue
        seen_diag.add(key)

        title = desc.get(key, code)
        triples.append({
            "entity": title,
            "attribute": "diagnosis",
            "value": code,
            "fhir_type": "Condition",
            "icd_version": ver,
            "source": "mimic_structured",
        })

    # Procedure triples
    pat_proc = procedures[procedures["subject_id"] == subject_id]
    seen_proc = set()
    for _, row in pat_proc.iterrows():
        code = str(row["icd_code"])
        if code in seen_proc:
            continue
        seen_proc.add(code)
        triples.append({
            "entity": code,
            "attribute": "procedure",
            "value": code,
            "fhir_type": "Procedure",
            "source": "mimic_structured",
        })

    # Admission triples (temporal)
    pat_adm = admissions[admissions["subject_id"] == subject_id].sort_values("admittime")
    for _, row in pat_adm.iterrows():
        triples.append({
            "entity": f"Admission_{row['hadm_id']}",
            "attribute": "admission_type",
            "value": str(row.get("admission_type", "")),
            "fhir_type": "Procedure",
            "temporal_anchor": str(row.get("admittime", "")),
            "discharge_time": str(row.get("dischtime", "")),
            "source": "mimic_structured",
        })

    return {
        "subject_id": subject_id,
        "num_triples": len(triples),
        "num_diagnoses": len(seen_diag),
        "num_procedures": len(seen_proc),
        "num_admissions": len(pat_adm),
        "triples": triples,
    }


def build_cohort_kgs(
    cohort: str = "oncology",
    max_patients: int = 5000,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build KGs for a MIMIC cohort.

    Args:
        cohort: "oncology" or "icu"
        max_patients: limit for memory safety
        output_dir: save per-patient KGs as JSON

    Returns:
        Summary stats for paper tables.
    """
    logger.info("Building MIMIC %s cohort KGs (max %d patients)...", cohort, max_patients)

    # Load tables once
    diagnoses = _read_table("hosp/diagnoses_icd")
    procedures = _read_table("hosp/procedures_icd")
    admissions = _read_table("hosp/admissions")

    # Select cohort subjects
    if cohort == "oncology":
        icd9_mask = diagnoses["icd_code"].str.match(r"^(1[4-9]\d|2[0-3]\d)", na=False) & (diagnoses["icd_version"] == 9)
        icd10_mask = diagnoses["icd_code"].str.match(r"^[CD]\d", na=False) & (diagnoses["icd_version"] == 10)
        cohort_diag = diagnoses[icd9_mask | icd10_mask]
    elif cohort == "icu":
        icu = _read_table("icu/icustays")
        icu_subjects = set(icu["subject_id"].unique())
        cohort_diag = diagnoses[diagnoses["subject_id"].isin(icu_subjects)]
    else:
        cohort_diag = diagnoses

    subject_ids = sorted(cohort_diag["subject_id"].unique()[:max_patients])
    logger.info("Processing %d %s patients...", len(subject_ids), cohort)

    # Build KGs in parallel (CPU-bound)
    results = []

    def _build(sid):
        return build_patient_kg(sid, diagnoses, procedures, admissions)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for kg in pool.map(_build, subject_ids):
            results.append(kg)
            if output_dir and kg["num_triples"] > 0:
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(output_dir / f"mimic_{kg['subject_id']}.json", "w") as f:
                    json.dump(kg, f, indent=2, default=str)

    # Aggregate stats
    total_triples = sum(r["num_triples"] for r in results)
    total_diag = sum(r["num_diagnoses"] for r in results)
    total_proc = sum(r["num_procedures"] for r in results)
    avg_triples = total_triples / max(len(results), 1)

    stats = {
        "cohort": cohort,
        "num_patients": len(results),
        "total_triples": total_triples,
        "total_diagnoses": total_diag,
        "total_procedures": total_proc,
        "avg_triples_per_patient": round(avg_triples, 1),
        "patients_with_triples": sum(1 for r in results if r["num_triples"] > 0),
    }

    logger.info(
        "MIMIC %s KG: %d patients, %d triples (%.1f avg/patient)",
        cohort, len(results), total_triples, avg_triples,
    )
    return stats
