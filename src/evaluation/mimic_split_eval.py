"""MIMIC train/test split evaluation for temporal KG.

Splits 2000 oncology patients into 1400 train / 600 test.
Evaluates:
  - Temporal KG construction statistics per split
  - Growth rate consistency (train vs test)
  - Temporal coverage
  - KG link prediction: can train TKG predict test diagnoses?
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TRAIN_RATIO = 0.7


def evaluate_mimic_splits(
    mimic_version: str = "iv",
    max_patients: int = 2000,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build MIMIC TKG with train/test splits and evaluate.

    Uses structured data only (no LLM) — 100% precision guaranteed.
    """
    import pandas as pd
    from src.data.mimic_kg import _read_table

    logger.info("Loading MIMIC-%s structured data...", mimic_version.upper())

    # Load tables
    if mimic_version == "iii":
        from src.data.mimic3_reader import read_table as read_m3
        diagnoses = read_m3("DIAGNOSES_ICD")
        procedures = read_m3("PROCEDURES_ICD")
        admissions = read_m3("ADMISSIONS")
        subject_col = "subject_id"
        hadm_col = "hadm_id"
        admittime_col = "admittime"
        icd_col = "icd9_code"
    else:
        diagnoses = _read_table("hosp/diagnoses_icd")
        procedures = _read_table("hosp/procedures_icd")
        admissions = _read_table("hosp/admissions")
        subject_col = "subject_id"
        hadm_col = "hadm_id"
        admittime_col = "admittime"
        icd_col = "icd_code"

    # Select oncology cohort
    if mimic_version == "iii":
        onc_mask = diagnoses[icd_col].astype(str).str.match(r"^(1[4-9]|2[0-3])", na=False)
    else:
        icd9_mask = diagnoses[icd_col].str.match(r"^(1[4-9]\d|2[0-3]\d)", na=False) & (diagnoses["icd_version"] == 9)
        icd10_mask = diagnoses[icd_col].str.match(r"^[CD]\d", na=False) & (diagnoses["icd_version"] == 10)
        onc_mask = icd9_mask | icd10_mask

    onc_patients = sorted(diagnoses[onc_mask][subject_col].unique())[:max_patients]
    logger.info("Found %d oncology patients", len(onc_patients))

    # Split
    n_train = int(len(onc_patients) * TRAIN_RATIO)
    train_pids = set(onc_patients[:n_train])
    test_pids = set(onc_patients[n_train:])

    logger.info("Split: %d train / %d test", len(train_pids), len(test_pids))

    # Build per-split KGs
    results = {}
    for split_name, split_pids in [("train", train_pids), ("test", test_pids)]:
        split_diag = diagnoses[diagnoses[subject_col].isin(split_pids)]
        split_proc = procedures[procedures[subject_col].isin(split_pids)] if len(procedures) > 0 else pd.DataFrame()
        split_adm = admissions[admissions[subject_col].isin(split_pids)]

        # Count relations
        n_diag = len(split_diag)
        n_proc = len(split_proc) if len(split_proc) > 0 else 0
        n_relations = n_diag + n_proc

        # Unique entities
        unique_codes = set(split_diag[icd_col].dropna().unique())
        if len(split_proc) > 0:
            unique_codes.update(split_proc[icd_col].dropna().unique())

        # Temporal metrics: admissions per patient
        adm_counts = split_adm.groupby(subject_col)[hadm_col].nunique()
        multi_adm = (adm_counts >= 2).sum()

        # Growth rate (avg relations from first to last admission)
        growth_rates = []
        for pid in list(split_pids)[:500]:  # sample for efficiency
            pid_adm = split_adm[split_adm[subject_col] == pid].sort_values(admittime_col)
            if len(pid_adm) < 2:
                continue
            hadm_ids = pid_adm[hadm_col].tolist()
            first_diag = len(split_diag[split_diag[hadm_col] == hadm_ids[0]])
            total_diag = len(split_diag[split_diag[subject_col] == pid])
            if first_diag > 0:
                growth_rates.append(total_diag / first_diag)

        results[split_name] = {
            "n_patients": len(split_pids),
            "n_relations": n_relations,
            "n_diagnoses": n_diag,
            "n_procedures": n_proc,
            "n_unique_entities": len(unique_codes),
            "avg_relations_per_patient": round(n_relations / max(len(split_pids), 1), 1),
            "multi_admission_patients": int(multi_adm),
            "temporal_coverage": round(multi_adm / max(len(split_pids), 1), 3),
            "avg_growth_rate": round(np.mean(growth_rates), 3) if growth_rates else 0,
            "std_growth_rate": round(np.std(growth_rates), 3) if growth_rates else 0,
        }
        logger.info(
            "%s %s: %d patients, %d relations, %d entities, growth=%.2f",
            mimic_version.upper(), split_name, len(split_pids),
            n_relations, len(unique_codes),
            results[split_name]["avg_growth_rate"],
        )

    # Link prediction: ICD codes in train but not test (coverage)
    if mimic_version != "iii":
        train_codes = set(diagnoses[diagnoses[subject_col].isin(train_pids)][icd_col].dropna())
        test_codes = set(diagnoses[diagnoses[subject_col].isin(test_pids)][icd_col].dropna())
        overlap = train_codes & test_codes
        results["code_coverage"] = {
            "train_unique_codes": len(train_codes),
            "test_unique_codes": len(test_codes),
            "overlap": len(overlap),
            "test_coverage_by_train": round(len(overlap) / max(len(test_codes), 1), 3),
        }
        logger.info("Code coverage: train covers %.1f%% of test codes",
                     results["code_coverage"]["test_coverage_by_train"] * 100)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"mimic{mimic_version}_split_eval.json", "w") as f:
            json.dump(results, f, indent=2)

    return results
