"""Fetch MIMIC oncology clinical notes via BigQuery for TRUST-KG.

Identifies ONCOLOGY patients by malignant-neoplasm ICD codes, then pulls their
free-text notes (discharge summaries; optionally radiology) and writes them to
data/mimic_oncology/<source>/ as JSONL — one record per note.

DUA: the pulled notes are patient data. data/ is git-ignored; never commit them.

Auth (both required):
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/physionet-sa-key.json
    export BQ_PROJECT=<your-billing-project>          # e.g. gen-lang-client-0951480354

Oncology filter (malignant neoplasms):
    ICD-10  C00–C97      (icd_code LIKE 'C%')
    ICD-9   140–208
  --cancer breast     adds ICD-10 C50 / ICD-9 174,233.0    (matches CORAL-BRCA)
  --cancer pancreatic adds ICD-10 C25 / ICD-9 157          (matches CORAL-PDAC)
  --cancer all        any malignant neoplasm (default)

    python scripts/fetch_mimic_oncology.py --source mimiciv --cancer all --limit 400
    python scripts/fetch_mimic_oncology.py --source mimiciii --cancer breast,pancreatic --limit 400
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

# ── dataset/table + column mapping per source ─────────────────────────────────
SOURCES = {
    "mimiciv": {
        "dx": "`physionet-data.mimiciv_3_1_hosp.diagnoses_icd`",
        "notes": "`physionet-data.mimiciv_note.discharge`",
        "radiology": "`physionet-data.mimiciv_note.radiology`",
        "note_cols": "note_id, subject_id, hadm_id, charttime AS note_time, text",
        "icd_col": "icd_code", "icd_ver_col": "icd_version", "join": "subject_id, hadm_id",
    },
    # MIMIC-III (contingent on physionet-data.mimiciii_* being visible to your project)
    "mimiciii": {
        "dx": "`physionet-data.mimiciii_clinical.diagnoses_icd`",
        "notes": "`physionet-data.mimiciii_notes.noteevents`",
        "radiology": None,
        "note_cols": "row_id AS note_id, subject_id, hadm_id, chartdate AS note_time, text",
        "icd_col": "icd9_code", "icd_ver_col": None, "join": "subject_id",
        "note_filter": "category = 'Discharge summary'",
    },
}

CANCER_ICD = {  # (icd10_prefixes, icd9_3digit_codes)
    "breast":     (["C50"], [174, 175, 233]),
    "pancreatic": (["C25"], [157]),
    "lung":       (["C34"], [162]),
    "colorectal": (["C18", "C19", "C20"], [153, 154]),
    "prostate":   (["C61"], [185]),
}


def onc_predicate(src, cancers):
    """Build the malignant-neoplasm WHERE predicate for the source."""
    ic, iv = src["icd_col"], src["icd_ver_col"]
    if cancers == ["all"]:
        if iv:  # mimic-iv: mixed ICD-9/10
            return (f"(({iv}=10 AND STARTS_WITH({ic},'C')) OR "
                    f"({iv}=9 AND SAFE_CAST(SUBSTR({ic},1,3) AS INT64) BETWEEN 140 AND 208))")
        return f"SAFE_CAST(SUBSTR({ic},1,3) AS INT64) BETWEEN 140 AND 208"
    pref10, codes9 = [], []
    for c in cancers:
        p10, c9 = CANCER_ICD[c]; pref10 += p10; codes9 += c9
    like10 = " OR ".join([f"STARTS_WITH({ic},'{p}')" for p in pref10])
    in9 = ",".join(str(x) for x in codes9)
    if iv:
        return (f"(({iv}=10 AND ({like10})) OR "
                f"({iv}=9 AND SAFE_CAST(SUBSTR({ic},1,3) AS INT64) IN ({in9})))")
    return f"SAFE_CAST(SUBSTR({ic},1,3) AS INT64) IN ({in9})"


def build_query(src, cancers, limit):
    pred = onc_predicate(src, cancers)
    conds = ["LENGTH(n.text) > 200"]
    if src.get("note_filter"):
        conds.insert(0, src["note_filter"])
    where_clause = "WHERE " + " AND ".join(conds)
    return f"""
WITH onc AS (
  SELECT DISTINCT {src['join']}
  FROM {src['dx']}
  WHERE {pred}
)
SELECT {src['note_cols']}
FROM {src['notes']} n
JOIN onc USING ({src['join']})
{where_clause}
LIMIT {int(limit)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="mimiciv")
    ap.add_argument("--cancer", default="all", help="all | comma list: breast,pancreatic,lung,...")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--dry-run", action="store_true", help="print the SQL and exit (no auth needed)")
    args = ap.parse_args()
    cancers = ["all"] if args.cancer == "all" else args.cancer.split(",")
    src = SOURCES[args.source]
    sql = build_query(src, cancers, args.limit)

    if args.dry_run:
        print(sql); return

    key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("BQ_PROJECT")
    if not key or not project:
        sys.exit("Set GOOGLE_APPLICATION_CREDENTIALS (SA key) and BQ_PROJECT. "
                 "Use --dry-run to inspect the SQL without auth.")
    from google.cloud import bigquery
    client = bigquery.Client(project=project)
    print(f"[fetch] {args.source} oncology={cancers} limit={args.limit}", flush=True)
    rows = list(client.query(sql).result())

    out = Path(f"data/mimic_oncology/{args.source}"); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"notes_{'_'.join(cancers)}.jsonl"
    with open(fp, "w") as w:
        for r in rows:
            w.write(json.dumps({k: (str(v) if v is not None else None) for k, v in dict(r).items()}) + "\n")
    print(f"[fetch] wrote {len(rows)} notes -> {fp}")


if __name__ == "__main__":
    main()
