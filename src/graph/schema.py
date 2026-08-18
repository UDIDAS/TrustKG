"""TRUST-KG schema (TBox) — the type-level ontology the instance graphs conform to.

We build KGs as **schema + instances**: this module emits the shared TBox (classes +
properties + FHIR alignment + ontology-grounding vocabulary); rdf_builder emits the ABox
(patient/entity instances). Type-level (not entity-level like a closed anatomy vocabulary),
because the clinical vocabulary is open. Merged into each cohort graph so every `.ttl` is a
self-contained schema+instances artifact, and also written standalone as `schema.ttl`.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rdflib import Graph, Literal, RDF, RDFS, OWL, XSD, URIRef
from rdflib.namespace import SKOS, DCTERMS

from src.graph.rdf_builder import TRUSTKG, FHIR, SNOMED, LOINC, RXNORM, PATIENT, SCHEMA
from src.graph.triple_normalizer import CANONICAL_FHIR

ONTOLOGY_URI = URIRef("http://trustkg.org/ontology")


def build_schema_graph() -> Graph:
    """Return the TRUST-KG TBox as an rdflib Graph."""
    g = Graph()
    for pfx, ns in [("trustkg", TRUSTKG), ("fhir", FHIR), ("patient", PATIENT),
                    ("snomed", SNOMED), ("loinc", LOINC), ("rxnorm", RXNORM),
                    ("owl", OWL), ("rdfs", RDFS), ("skos", SKOS), ("dcterms", DCTERMS)]:
        g.bind(pfx, ns)

    # ── ontology header ──
    g.add((ONTOLOGY_URI, RDF.type, OWL.Ontology))
    g.add((ONTOLOGY_URI, RDFS.label, Literal("TRUST-KG clinical schema (type-level TBox)")))
    g.add((ONTOLOGY_URI, DCTERMS.description, Literal(
        "FHIR-aligned classes + provenance / trust / temporal predicates that the "
        "instance graphs conform to. Instances link to SNOMED CT / RxNorm / LOINC.")))

    # ── root class ──
    root = TRUSTKG.ClinicalEntity
    g.add((root, RDF.type, OWL.Class))
    g.add((root, RDFS.label, Literal("Clinical entity")))
    g.add((root, RDFS.comment, Literal("Any extracted clinical fact node (super-class of the FHIR types).")))

    # ── FHIR-aligned classes (from the normalizer's canonical resource set) ──
    for cls in sorted(CANONICAL_FHIR):
        c = FHIR[cls]
        g.add((c, RDF.type, OWL.Class))
        g.add((c, RDFS.label, Literal(cls)))
        g.add((c, RDFS.isDefinedBy, URIRef("http://hl7.org/fhir/")))
        g.add((c, RDFS.seeAlso, URIRef(f"http://hl7.org/fhir/{cls}")))
        if cls != "Patient":
            g.add((c, RDFS.subClassOf, root))

    # ── ontology-grounding concept class ──
    oc = TRUSTKG.OntologyConcept
    g.add((oc, RDF.type, OWL.Class))
    g.add((oc, RDFS.label, Literal("Ontology concept")))
    g.add((oc, RDFS.comment, Literal("A SNOMED CT / RxNorm / LOINC concept an entity is grounded to.")))

    # ── object properties ──
    def obj(name, dom, rng, label, comment=None):
        p = TRUSTKG[name]
        g.add((p, RDF.type, OWL.ObjectProperty))
        if dom: g.add((p, RDFS.domain, dom))
        if rng: g.add((p, RDFS.range, rng))
        g.add((p, RDFS.label, Literal(label)))
        if comment: g.add((p, RDFS.comment, Literal(comment)))
        return p

    obj("hasEntity", FHIR.Patient, root, "has entity", "Links a patient to an extracted clinical entity.")
    clinical_attr = obj("clinicalAttribute", root, None, "clinical attribute",
                        "Super-property of the open-vocabulary extracted attributes (entity -> value).")

    # ── datatype properties ──
    def dat(name, rng, label, comment=None, sub_of=None):
        p = TRUSTKG[name]
        g.add((p, RDF.type, OWL.DatatypeProperty))
        g.add((p, RDFS.domain, root))
        if rng: g.add((p, RDFS.range, rng))
        g.add((p, RDFS.label, Literal(label)))
        if comment: g.add((p, RDFS.comment, Literal(comment)))
        if sub_of: g.add((p, RDFS.subPropertyOf, sub_of))

    dat("hasValue", None, "has value", "Literal value of an entity's attribute.", clinical_attr)
    dat("temporalAnchor", None, "temporal anchor", "When the fact holds (date / relative phrase).")
    dat("evidenceSpan", None, "evidence span", "Verbatim supporting text from the source note (de-id scrubbed).")
    dat("ontologyCode", None, "ontology code", "SNOMED/RxNorm/LOINC code the entity is grounded to.")
    dat("consensusLevel", XSD.integer, "consensus level", "Number of ensemble models that produced the fact.")
    dat("trustScore", XSD.float, "trust score",
        "Calibrated reliability T(τ) driving the Insert / Review / Reject admission gate.")
    return g


def write_schema(path="results/rdf/schema.ttl") -> str:
    from pathlib import Path
    g = build_schema_graph()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=path, format="turtle")
    return path


if __name__ == "__main__":
    g = build_schema_graph()
    p = write_schema()
    print(f"schema.ttl: {len(g)} TBox statements -> {p}")
    print(f"  classes: {sum(1 for _ in g.subjects(RDF.type, OWL.Class))}"
          f"  objProps: {sum(1 for _ in g.subjects(RDF.type, OWL.ObjectProperty))}"
          f"  dataProps: {sum(1 for _ in g.subjects(RDF.type, OWL.DatatypeProperty))}")
