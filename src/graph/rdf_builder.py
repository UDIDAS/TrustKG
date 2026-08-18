"""Build RDF/OWL semantic graphs from validated EAV triples.

Produces:
  - RDF triples linked to biomedical ontology URIs
  - OWL class hierarchy for FHIR resource types
  - SPARQL-queryable graph for clinical cohort retrieval
  - Temporal annotations as quintuplets (s,p,o,t_start,t_end)

Uses rdflib for graph construction and serialization.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, XSD, URIRef, BNode
from rdflib.namespace import DCTERMS

logger = logging.getLogger(__name__)

# ── Namespace definitions ──────────────────────────────────────
TRUSTKG = Namespace("http://trustkg.org/ontology/")
PATIENT = Namespace("http://trustkg.org/patient/")
SNOMED = Namespace("http://snomed.info/id/")
LOINC = Namespace("http://loinc.org/")
RXNORM = Namespace("http://rxnorm.nlm.nih.gov/")
FHIR = Namespace("http://hl7.org/fhir/")
SCHEMA = Namespace("http://schema.org/")

# FHIR type to OWL class mapping
FHIR_CLASS_MAP = {
    "Condition": FHIR.Condition,
    "Observation": FHIR.Observation,
    "Procedure": FHIR.Procedure,
    "MedicationStatement": FHIR.MedicationStatement,
    "CarePlan": FHIR.CarePlan,
    "FamilyMemberHistory": FHIR.FamilyMemberHistory,
    "AllergyIntolerance": FHIR.AllergyIntolerance,
}

# Models emit free-form FHIR types (Medication, Finding, Symptom, Diagnosis, ...);
# normalize the common variants to the 7 canonical resource types above.
_FHIR_ALIASES = {
    "medication": "MedicationStatement", "medicationstatement": "MedicationStatement", "drug": "MedicationStatement",
    "condition": "Condition", "diagnosis": "Condition", "problem": "Condition", "symptom": "Condition",
    "disease": "Condition", "disorder": "Condition", "clinicalcondition": "Condition",
    "observation": "Observation", "finding": "Observation", "obs": "Observation", "test": "Observation",
    "laboratorytest": "Observation", "labtest": "Observation", "lab": "Observation", "measurement": "Observation",
    "vitalsign": "Observation", "vitalsigns": "Observation", "codeableconcept": "Observation",
    "procedure": "Procedure", "surgery": "Procedure", "intervention": "Procedure",
    "imagingstudy": "Procedure", "imaging": "Procedure",
    "careplan": "CarePlan", "familymemberhistory": "FamilyMemberHistory", "familyhistory": "FamilyMemberHistory",
    "allergyintolerance": "AllergyIntolerance", "allergy": "AllergyIntolerance",
}


def _normalize_fhir_type(ft: str) -> str:
    """Map a model-emitted fhir_type string to a canonical FHIR resource type."""
    if not ft:
        return ""
    key = str(ft).strip().lower()
    if key in _FHIR_ALIASES:
        return _FHIR_ALIASES[key]
    for canon in FHIR_CLASS_MAP:          # already canonical (any case)
        if key == canon.lower():
            return canon
    return str(ft)                        # unknown -> passthrough (stays untyped)


def _safe_uri(text: str) -> str:
    """Convert text to a safe URI component."""
    safe = re.sub(r"[^\w\-]", "_", text.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return quote(safe[:100], safe="")


def filter_by_trust(triples: list[dict], threshold: float = 0.4) -> tuple[list[dict], list[dict]]:
    """Filter triples by trust score T(τ) ≥ δ (Draft §3.4).

    Returns (accepted, rejected) triples.
    """
    accepted = []
    rejected = []
    for t in triples:
        val = t.get("_validation", {})
        trust = val.get("trust_score", t.get("_gnn_trust", 0.5))
        if trust >= threshold:
            accepted.append(t)
        else:
            rejected.append(t)
    return accepted, rejected


# ── Ontology URI mapping (Gap 3: link to SNOMED/RxNorm/LOINC) ──

_SNOMED_MAP = {
    "breast cancer": "254837009", "invasive ductal carcinoma": "408643008",
    "pancreatic adenocarcinoma": "363418001", "adenocarcinoma": "35917007",
    "metastasis": "128462008", "hypertension": "38341003",
    "diabetes": "73211009", "anemia": "271737000", "sepsis": "91302008",
    "nausea": "422587007", "pain": "22253000", "dyspnea": "267036007",
    "mastectomy": "172043006", "chemotherapy": "367336001",
    "radiation therapy": "108290001", "biopsy": "86273004",
}

_RXNORM_MAP = {
    "gemcitabine": "12574", "paclitaxel": "56946", "carboplatin": "40048",
    "cisplatin": "2555", "doxorubicin": "3639", "tamoxifen": "10324",
    "trastuzumab": "224905", "anastrozole": "84857", "metformin": "6809",
    "furosemide": "4603", "aspirin": "1191", "insulin": "5856",
}

_LOINC_MAP = {
    "hemoglobin": "718-7", "wbc": "6690-2", "platelet": "777-3",
    "creatinine": "2160-0", "bilirubin": "1975-2", "albumin": "1751-7",
    "glucose": "2345-7", "ca 19-9": "24108-3", "cea": "2039-6",
    "her2": "48676-1", "estrogen receptor": "16112-5",
}


def _resolve_ontology_uri(entity: str) -> tuple[URIRef | None, str]:
    """Map an entity to an ontology URI if known."""
    e = entity.lower().strip()
    for term, code in _SNOMED_MAP.items():
        if term in e:
            return SNOMED[code], f"SNOMED:{code}"
    for term, code in _RXNORM_MAP.items():
        if term in e:
            return RXNORM[code], f"RxNorm:{code}"
    for term, code in _LOINC_MAP.items():
        if term in e:
            return LOINC[code], f"LOINC:{code}"
    return None, ""


def build_patient_graph(
    patient_id: str,
    triples: list[dict],
    include_trust: bool = True,
    include_temporal: bool = True,
    trust_threshold: float = 0.0,
    normalize: bool = True,
) -> Graph:
    """Build an RDF graph from validated triples for one patient.

    Args:
        patient_id: e.g. "brca_20"
        triples: validated triples (with _validation and optionally _gnn_trust)
        include_trust: add trust score annotations
        include_temporal: add temporal annotations
        trust_threshold: filter triples below this trust (0.0 = keep all)

    Returns:
        rdflib.Graph with RDF/OWL triples
    """
    g = Graph()
    g.bind("trustkg", TRUSTKG)
    g.bind("patient", PATIENT)
    g.bind("snomed", SNOMED)
    g.bind("loinc", LOINC)
    g.bind("rxnorm", RXNORM)
    g.bind("fhir", FHIR)
    g.bind("schema", SCHEMA)

    # Deterministic structural normalization before materialization: canonicalize fhir_type,
    # drop PHI / vacuous / administrative, collapse degenerate, dedup (triple_normalizer.py).
    if normalize:
        from src.graph.triple_normalizer import normalize_triples
        triples, _ = normalize_triples(triples)

    # Trust-aware filtering (§3.4): only retain triples with T(τ) ≥ δ
    if trust_threshold > 0:
        triples, rejected = filter_by_trust(triples, trust_threshold)
        if rejected:
            logger.info("  %s: filtered %d/%d triples (trust < %.2f)",
                       patient_id, len(rejected), len(rejected) + len(triples), trust_threshold)

    # Patient node
    patient_uri = PATIENT[_safe_uri(patient_id)]
    g.add((patient_uri, RDF.type, FHIR.Patient))
    g.add((patient_uri, RDFS.label, Literal(patient_id)))

    for i, triple in enumerate(triples):
        entity = str(triple.get("entity", "")).strip()
        attribute = str(triple.get("attribute", "")).strip()
        value = str(triple.get("value", "")).strip()
        fhir_type = str(triple.get("fhir_type", "Unknown"))
        temporal = str(triple.get("temporal_anchor", ""))
        evidence = str(triple.get("evidence_span", ""))

        if not entity or not value:
            continue

        # Entity URI
        entity_uri = TRUSTKG[_safe_uri(entity)]

        # Add entity type
        fhir_class = FHIR_CLASS_MAP.get(_normalize_fhir_type(fhir_type))
        if fhir_class:
            g.add((entity_uri, RDF.type, fhir_class))
        g.add((entity_uri, RDFS.label, Literal(entity)))

        # Ontology URI linking (Gap 3: resolve to SNOMED/RxNorm/LOINC)
        onto_uri, onto_code = _resolve_ontology_uri(entity)
        if onto_uri:
            g.add((entity_uri, OWL.sameAs, onto_uri))
            g.add((entity_uri, TRUSTKG.ontologyCode, Literal(onto_code)))

        # Relation predicate
        predicate = TRUSTKG[_safe_uri(attribute)] if attribute else TRUSTKG.hasValue

        # Value — typed float literal only if it truly parses as a number
        # (the regex ^[\d.]+$ also matches '...' / '1.2.3', so guard the float() cast)
        try:
            value_node = Literal(float(value), datatype=XSD.float)
        except (ValueError, TypeError):
            value_node = Literal(value)

        # Core triple
        g.add((entity_uri, predicate, value_node))

        # Link entity to patient
        g.add((patient_uri, TRUSTKG.hasEntity, entity_uri))

        # Evidence provenance
        if evidence:
            g.add((entity_uri, TRUSTKG.evidenceSpan, Literal(evidence[:200])))

        # Trust annotations
        if include_trust:
            val_data = triple.get("_validation", {})
            trust = triple.get("_gnn_trust", val_data.get("trust_score", 0.5))
            g.add((entity_uri, TRUSTKG.trustScore, Literal(trust, datatype=XSD.float)))

            consensus = triple.get("_consensus_level", "")
            if consensus:
                g.add((entity_uri, TRUSTKG.consensusLevel, Literal(consensus)))

        # Temporal annotations
        if include_temporal and temporal and temporal.lower() not in ("null", "none", ""):
            g.add((entity_uri, TRUSTKG.temporalAnchor, Literal(temporal)))

    logger.info(
        "Built RDF graph for %s: %d triples in graph from %d input triples",
        patient_id, len(g), len(triples),
    )
    return g


def serialize_graph(
    graph: Graph,
    output_path: Path,
    fmt: str = "turtle",
) -> None:
    """Serialize RDF graph to file."""
    graph.serialize(destination=str(output_path), format=fmt)
    logger.info("Serialized graph to %s (%s)", output_path, fmt)


def run_sparql_queries(
    graph: Graph,
    queries: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """Run clinical SPARQL queries against the graph.

    Default queries test oncology cohort retrieval scenarios.
    """
    if queries is None:
        queries = {
            # §4.10: Clinical cohort queries
            "all_conditions": """
                SELECT ?patient ?entity ?label WHERE {
                    ?entity a fhir:Condition .
                    ?entity rdfs:label ?label .
                    ?patient trustkg:hasEntity ?entity .
                }
            """,
            "all_medications": """
                SELECT ?patient ?entity ?label WHERE {
                    ?entity a fhir:MedicationStatement .
                    ?entity rdfs:label ?label .
                    ?patient trustkg:hasEntity ?entity .
                }
            """,
            "all_procedures": """
                SELECT ?patient ?entity ?label WHERE {
                    ?entity a fhir:Procedure .
                    ?entity rdfs:label ?label .
                    ?patient trustkg:hasEntity ?entity .
                }
            """,
            "ontology_linked_entities": """
                SELECT ?entity ?label ?code WHERE {
                    ?entity trustkg:ontologyCode ?code .
                    ?entity rdfs:label ?label .
                }
            """,
            "high_trust_entities": """
                SELECT ?entity ?label ?trust WHERE {
                    ?entity trustkg:trustScore ?trust .
                    ?entity rdfs:label ?label .
                    FILTER(?trust >= 0.8)
                }
            """,
            "temporal_facts": """
                SELECT ?entity ?time WHERE {
                    ?entity trustkg:temporalAnchor ?time .
                }
            """,
            "entity_count": """
                SELECT (COUNT(DISTINCT ?entity) AS ?count) WHERE {
                    ?patient trustkg:hasEntity ?entity .
                }
            """,
        }

    results = {}
    for name, query in queries.items():
        try:
            qres = graph.query(query)
            results[name] = [
                {str(k): str(v) for k, v in zip(qres.vars, row)}
                for row in qres
            ]
        except Exception as e:
            logger.warning("SPARQL query '%s' failed: %s", name, e)
            results[name] = []

    return results


def build_cohort_graph(
    patient_graphs: dict[str, Graph],
) -> Graph:
    """Merge multiple patient graphs into a cohort-level graph."""
    cohort = Graph()
    cohort.bind("trustkg", TRUSTKG)
    cohort.bind("patient", PATIENT)
    cohort.bind("fhir", FHIR)

    for pid, pg in patient_graphs.items():
        for s, p, o in pg:
            cohort.add((s, p, o))

    logger.info(
        "Built cohort graph: %d patients, %d total triples",
        len(patient_graphs), len(cohort),
    )
    return cohort
