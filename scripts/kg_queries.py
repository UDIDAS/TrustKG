"""Interesting, medically-relevant SPARQL queries over a TRUST-KG cohort graph.

Demonstrates the KG's value beyond scale counts: patient-level clinical summaries
(each patient is a subgraph in the one dataset-level .ttl) + cohort-level patterns.

    python scripts/kg_queries.py results/rdf/coral_pdac.ttl PDAC
    python scripts/kg_queries.py results/rdf/mimic_mimiciii.ttl MIMIC-III
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rdflib import Graph

PFX = """
PREFIX trustkg: <http://trustkg.org/ontology/>
PREFIX patient: <http://trustkg.org/patient/>
PREFIX fhir: <http://hl7.org/fhir/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
"""

CHEMO = "gemcitabine|folfirinox|abraxane|nab-paclitaxel|paclitaxel|fluorouracil|5-fu|oxaliplatin|irinotecan|capecitabine|cisplatin|carboplatin|doxorubicin|cyclophosphamide|docetaxel|trastuzumab|pembrolizumab"
CANCER = "cancer|carcinoma|adenocarc|malignan|neoplas|tumou?r|metasta"

QUERIES = [
    ("Q1  Patient-level: diagnosis + chemo received (per-patient clinical summary)", f"""
        SELECT ?patient (SAMPLE(?cl) AS ?diagnosis)
               (GROUP_CONCAT(DISTINCT ?ml; separator=", ") AS ?chemotherapy) WHERE {{
            ?patient trustkg:hasEntity ?c, ?m .
            ?c a fhir:Condition ; rdfs:label ?cl . FILTER(REGEX(?cl, "{CANCER}", "i"))
            ?m a fhir:MedicationStatement ; rdfs:label ?ml . FILTER(REGEX(?ml, "{CHEMO}", "i"))
        }} GROUP BY ?patient LIMIT 8"""),

    ("Q2  Cohort: diagnosed cancer patients who received chemotherapy (treated cohort)", f"""
        SELECT (COUNT(DISTINCT ?p) AS ?treated_patients) WHERE {{
            ?p trustkg:hasEntity ?c, ?m .
            ?c a fhir:Condition ; rdfs:label ?cl . FILTER(REGEX(?cl, "{CANCER}", "i"))
            ?m a fhir:MedicationStatement ; rdfs:label ?ml . FILTER(REGEX(?ml, "{CHEMO}", "i"))
        }}"""),

    ("Q3  Cohort: most-prescribed medications (drug utilization)", """
        SELECT ?med (COUNT(DISTINCT ?p) AS ?patients) WHERE {
            ?p trustkg:hasEntity ?m . ?m a fhir:MedicationStatement ; rdfs:label ?med .
        } GROUP BY ?med ORDER BY DESC(?patients) LIMIT 10"""),

    ("Q4  Comorbidity: conditions co-occurring in cancer patients", f"""
        SELECT ?condition (COUNT(DISTINCT ?p) AS ?patients) WHERE {{
            ?p trustkg:hasEntity ?cancer, ?other .
            ?cancer a fhir:Condition ; rdfs:label ?cl . FILTER(REGEX(?cl, "{CANCER}", "i"))
            ?other a fhir:Condition ; rdfs:label ?condition .
            FILTER(!REGEX(?condition, "{CANCER}", "i"))
        }} GROUP BY ?condition ORDER BY DESC(?patients) LIMIT 10"""),

    ("Q5  Standardized: ontology-grounded facts (SNOMED / RxNorm / LOINC)", """
        SELECT ?label ?code WHERE {
            ?e rdfs:label ?label ; trustkg:ontologyCode ?code .
        } LIMIT 12"""),

    ("Q6  Temporal: time-anchored clinical events (timeline signal)", """
        SELECT ?patient ?event ?when WHERE {
            ?patient trustkg:hasEntity ?e .
            ?e rdfs:label ?event ; trustkg:temporalAnchor ?when .
            FILTER(STRLEN(?when) > 3 && !REGEX(?when, "REDACTED"))
        } LIMIT 12"""),
]


def short(uri):
    return str(uri).split("/")[-1].split("#")[-1][:40]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/rdf/coral_pdac.ttl"
    name = sys.argv[2] if len(sys.argv) > 2 else Path(path).stem
    g = Graph().parse(path, format="turtle")
    print(f"\n{'='*74}\n  INTERESTING QUERIES — {name}   ({len(g)} statements)\n{'='*74}")
    for title, q in QUERIES:
        print(f"\n{title}")
        try:
            rows = list(g.query(PFX + q))
        except Exception as e:
            print(f"   (query error: {e})"); continue
        if not rows:
            print("   (no rows)"); continue
        for r in rows[:10]:
            print("   " + " | ".join(short(v) for v in r))


if __name__ == "__main__":
    main()
