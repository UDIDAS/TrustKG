"""Central configuration for TRUST-KG pipeline."""
import os
from pathlib import Path

# ── Project paths ──────────────────────────────────────────────
# Recovered project: original root ~/Desktop/CIKM 26 was lost; default to the
# rebuilt TrustKG project. Override with TRUSTKG_ROOT if needed.
PROJECT_ROOT = Path(os.environ.get("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG"))
DATA_DIR = PROJECT_ROOT / "data"
CORAL_DIR = DATA_DIR / "coral"
MIMIC_DIR = DATA_DIR / "mimic"
RESULTS_DIR = PROJECT_ROOT / "results"
EXTRACTION_DIR = RESULTS_DIR / "extraction"

# Legacy CORAL project (ontologies, schemas, prior results)
CORAL_LEGACY = Path("/home/ud3d4/Desktop/Projects/CORAL")
CORAL_UNANNOTATED = CORAL_LEGACY / "coral" / "unannotated" / "data"

# ── API keys (loaded from project .env, fallback to legacy) ────
_env_candidates = [
    PROJECT_ROOT / ".env",
    CORAL_LEGACY / "src" / ".env",
    Path("/home/ud3d4/Desktop/.env"),  # recovered creds live here
]
_env_path = next((p for p in _env_candidates if p.exists()), _env_candidates[0])
_env_vars = {}
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            _env_vars[k.strip()] = v.strip()

GEMINI_API_KEY = _env_vars.get("GEMINI_API_KEY", "")
HF_TOKEN = _env_vars.get("HF_TOKEN", "")
LOINC_UID = _env_vars.get("LOINC_UID", "")
LOINC_PWD = _env_vars.get("LOINC_PWD", "")
ICD_TOKEN = _env_vars.get("ICD_CLIENT_TOKEN", "")

# ── Extraction settings ────────────────────────────────────────
MAX_WORKERS = 4            # ThreadPool concurrency (cluster-safe)
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_RPM_LIMIT = 14      # requests per minute (free tier safe)
CHUNK_MAX_CHARS = 6_000    # max chars per LLM call (raised on L40S 48GB; was 4000 for A6000 OOM safety)
MAX_CANDIDATES_PER_CHUNK = 200  # cap NER candidates in prompt (raised from 80: brca notes have ~940 candidates -> ~half were dropped, hurting recall)
EXTRACT_MAX_NEW_TOKENS = 8_192  # generation budget for EAV JSON (raised from 4096 to stop dense-note truncation)
EXTRACTION_BATCH_SIZE = 5  # patients per batch (memory-safe)

# ── FHIR entity categories ─────────────────────────────────────
FHIR_CATEGORIES = [
    "Condition",       # diagnoses, findings, symptoms
    "Observation",     # lab results, vital signs, biomarkers
    "Procedure",       # surgeries, biopsies, imaging
    "MedicationStatement",  # drugs, chemotherapy, dosing
    "CarePlan",        # treatment plans, recommendations
    "FamilyMemberHistory",
    "AllergyIntolerance",
]

# ── Ontology references ────────────────────────────────────────
ONTOLOGIES = ["SNOMED_CT", "LOINC", "RxNorm", "ICD-10", "Gene_Ontology"]
