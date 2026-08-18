"""Deterministic triple normalization — structural cleanup before RDF materialization.

Fixes the *structural* failure modes surveyed in scripts/mimic_failure_survey.py
(invalid fhir_type, degenerate entity==value, vacuous/generic entity, PHI leak) with
NO model calls and NO fine-tuning. This is the stage between the ensemble union and
rdf_builder; it does not touch the semantic residual (wrong-relation triples) — that is
the trust gate's job.

Pipeline per triple:
  1. drop PHI de-id placeholders          (compliance)
  2. recast vacuous demographic entities  ('year --age--> 84' -> Patient.age = 84)
     or drop other vacuous entities
  3. drop administrative non-clinical types
  4. collapse degenerate (entity==value)  -> concept node with a status value
  5. canonicalize fhir_type               -> a real FHIR resource
  6. dedup by canonical (entity, attribute, value, type)   -> merges mention variants

normalize_triples(triples) -> (kept: list[dict], report: Counter)
"""
from __future__ import annotations

import re
from collections import Counter

CANONICAL_FHIR = {
    "Patient", "Condition", "Observation", "Procedure", "MedicationStatement",
    "MedicationRequest", "AllergyIntolerance", "FamilyMemberHistory", "DiagnosticReport",
    "Encounter", "Immunization", "BodyStructure", "Specimen", "CarePlan", "Device",
    "ServiceRequest",
}

# loose LLM fhir_type (lowercased) -> canonical FHIR resource
_FHIR_MAP = {
    "person": "Patient", "age": "Patient", "language": "Patient", "gender": "Patient",
    "administrativegender": "Patient", "maritalstatus": "Patient", "race": "Patient",
    "ethnicity": "Patient", "demographics": "Patient",
    "condition": "Condition", "diagnosis": "Condition", "problem": "Condition",
    "disease": "Condition", "symptom": "Condition", "comorbidity": "Condition",
    "finding": "Observation", "observation": "Observation", "quantitative": "Observation",
    "test": "Observation", "labresult": "Observation", "laboratory": "Observation",
    "lab": "Observation", "vitalsign": "Observation", "measurement": "Observation",
    "value": "Observation",
    "diagnosticreport": "DiagnosticReport", "imaging": "DiagnosticReport", "report": "DiagnosticReport",
    "medication": "MedicationStatement", "medicationstatement": "MedicationStatement",
    "drug": "MedicationStatement", "medicationorder": "MedicationRequest",
    "medicationrequest": "MedicationRequest", "prescription": "MedicationRequest",
    "procedure": "Procedure", "intervention": "Procedure", "surgery": "Procedure",
    "treatment": "Procedure", "operation": "Procedure",
    "encounter": "Encounter", "admission": "Encounter", "caresetting": "Encounter",
    "service": "Encounter", "visit": "Encounter",
    "allergy": "AllergyIntolerance", "allergyintolerance": "AllergyIntolerance",
    "familymemberhistory": "FamilyMemberHistory", "familyhistory": "FamilyMemberHistory",
    "device": "Device", "bodystructure": "BodyStructure", "bodysite": "BodyStructure",
    "anatomy": "BodyStructure", "immunization": "Immunization", "vaccine": "Immunization",
    "specimen": "Specimen", "careplan": "CarePlan", "plan": "CarePlan",
}

# administrative / non-clinical types whose entity is usually not a fact worth a node
_ADMIN_TYPES = {"role", "relationship", "practitioner", "duration", "quantity", "text",
                "careteam", "organization", "location", "provider"}

# attributes carrying a Patient-level demographic fact (recast vacuous entity -> Patient)
_DEMOGRAPHIC_ATTRS = {"age", "gender", "sex", "language", "race", "ethnicity",
                      "marital_status", "maritalstatus", "date_of_birth", "dob", "deceased"}

# function-word / non-concept entities
_VACUOUS = {
    "multiple", "year", "years", "course", "courses", "myself", "colleague", "colleagues",
    "none", "unknown", "patient", "several", "various", "other", "yes", "no", "normal",
    "stable", "day", "days", "time", "times", "gentleman", "lady", "man", "woman", "he",
    "she", "they", "it", "provider", "attending", "physician", "nurse", "team", "week",
    "weeks", "month", "months", "hour", "hours", "history",
}

_PHI = re.compile(
    r"\[|\*\*\*\*\*|known (lastname|firstname)|last name \(stitle\)|month/day/year|"
    r"\bhospital\d|\bde-?identified\b", re.IGNORECASE)

# attribute -> status literal used when collapsing degenerate (entity==value) triples
_STATUS_FROM_ATTR = {
    "has_history": "history", "history": "history", "past_medical_history": "history",
    "reason_for_admission": "admission-reason", "present": "present", "has_finding": "present",
}


def _norm(s) -> str:
    if isinstance(s, (list, tuple)):
        s = " ".join(map(str, s))
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def is_phi(text) -> bool:
    return bool(_PHI.search(str(text or "")))


def is_vacuous(entity) -> bool:
    e = _norm(entity)
    return (e in _VACUOUS) or (len(e) <= 2) or e.isdigit()


def canonical_fhir_type(raw, attribute="", value="") -> str:
    """Map a loose LLM fhir_type to a real FHIR resource (infer from attribute if unknown)."""
    if raw in CANONICAL_FHIR:
        return raw
    r = _norm(raw)
    if r in _FHIR_MAP:
        return _FHIR_MAP[r]
    a = _norm(attribute)
    if any(k in a for k in ("dose", "medication", "drug", "route", "mg")):
        return "MedicationStatement"
    if any(k in a for k in ("procedure", "surgery", "performed", "operation")):
        return "Procedure"
    if any(k in a for k in ("lab", "level", "result", "count", "value", "measure")):
        return "Observation"
    if any(k in a for k in ("diagnosis", "condition", "history", "finding")):
        return "Condition"
    return "Observation"   # default clinical bucket


def normalize_triples(triples: list[dict]) -> tuple[list[dict], Counter]:
    """Clean + canonicalize + dedup a patient's triples. Returns (kept, report)."""
    kept: list[dict] = []
    report: Counter = Counter()
    seen: set = set()
    report["input"] = len(triples)
    for t in triples:
        if not isinstance(t, dict):
            report["dropped_malformed"] += 1
            continue
        e = str(t.get("entity", "")); a = str(t.get("attribute", "")); v = str(t.get("value", ""))
        ft = str(t.get("fhir_type", ""))

        # 1. PHI -> drop
        if is_phi(e) or is_phi(v):
            report["dropped_phi"] += 1
            continue

        out = dict(t)

        # 2. vacuous entity -> recast demographic, else drop
        if is_vacuous(e):
            if _norm(a) in _DEMOGRAPHIC_ATTRS and v:
                out["entity"] = "Patient"; out["fhir_type"] = "Patient"; e = "Patient"
                report["recast_demographic"] += 1
            else:
                report["dropped_vacuous"] += 1
                continue

        # 3. administrative non-clinical type -> drop (unless demographic)
        if _norm(ft) in _ADMIN_TYPES and _norm(a) not in _DEMOGRAPHIC_ATTRS:
            report["dropped_admin"] += 1
            continue

        # 4. degenerate entity==value -> concept node with a status value
        if _norm(e) == _norm(v) and e:
            v = _STATUS_FROM_ATTR.get(_norm(a), "present")
            out["value"] = v
            report["collapsed_degenerate"] += 1

        # 5. canonicalize fhir_type
        cft = canonical_fhir_type(ft, a, v)
        if cft != ft:
            report["retyped_fhir"] += 1
        out["fhir_type"] = cft
        out["_norm_entity"] = _norm(e)

        # 6. dedup by canonical (entity, attribute, value, type)
        key = (_norm(e), _norm(a), _norm(v), cft)
        if key in seen:
            report["deduped"] += 1
            continue
        seen.add(key)
        kept.append(out)

    report["kept"] = len(kept)
    return kept, report


if __name__ == "__main__":   # quick before/after demo on real extracted triples
    import glob
    import json
    import sys
    coh = sys.argv[1] if len(sys.argv) > 1 else "mimiciii"
    model = sys.argv[2] if len(sys.argv) > 2 else "gemma4-e4b"
    files = sorted(glob.glob(f"results/extraction/mimic_{coh}/bymodel/{model}/*.json"))[:120]
    agg = Counter(); valid_before = valid_after = tot_before = tot_after = 0
    for f in files:
        tr = [t for t in json.load(open(f)).get("triples", []) if isinstance(t, dict)]
        tot_before += len(tr)
        valid_before += sum(1 for t in tr if t.get("fhir_type") in CANONICAL_FHIR)
        kept, rep = normalize_triples(tr)
        for k, val in rep.items():
            agg[k] += val
        tot_after += len(kept)
        valid_after += sum(1 for t in kept if t.get("fhir_type") in CANONICAL_FHIR)
    print(f"normalize demo: {coh}/{model}, {len(files)} notes")
    print(f"  triples: {tot_before} -> {tot_after} kept "
          f"({100 * tot_after / max(tot_before,1):.0f}%)")
    print(f"  valid FHIR type: {100*valid_before/max(tot_before,1):.0f}% -> "
          f"{100*valid_after/max(tot_after,1):.0f}%")
    for k in ("dropped_phi", "dropped_vacuous", "dropped_admin", "recast_demographic",
              "collapsed_degenerate", "retyped_fhir", "deduped"):
        print(f"  {k:22s} {agg.get(k,0)}")
