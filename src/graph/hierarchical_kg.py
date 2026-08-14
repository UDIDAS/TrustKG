"""Hierarchical Knowledge Graph with schema-level and data-level layers.

Architecture:
  SCHEMA LEVEL (TBox):
    - OWL class hierarchy auto-generated from extracted entity types
    - Property definitions with domain/range constraints
    - Temporal property annotations
    - Evolves as new entity types are discovered

  DATA LEVEL (ABox):
    - Patient-specific entity instances
    - Hierarchical relations between instances
    - Temporal validity intervals on each fact
    - Links to schema-level classes

The schema level is DYNAMIC — it evolves as new data is extracted,
capturing patterns across patients and cohorts.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, XSD, URIRef
from rdflib.namespace import DCTERMS

logger = logging.getLogger(__name__)

# Namespaces
TKG = Namespace("http://trustkg.org/ontology/")
TKGI = Namespace("http://trustkg.org/instance/")
TKGT = Namespace("http://trustkg.org/temporal/")
FHIR = Namespace("http://hl7.org/fhir/")
SNOMED = Namespace("http://snomed.info/id/")
TIME = Namespace("http://www.w3.org/2006/time#")


def _safe_uri(text: str) -> str:
    text = re.sub(r"[^\w\-]", "_", text.strip())
    return re.sub(r"_+", "_", text).strip("_")[:100]


# ═══════════════════════════════════════════════════════════════
# SCHEMA LEVEL (TBox) — auto-generated from data
# ═══════════════════════════════════════════════════════════════

class DynamicSchema:
    """Dynamically evolving schema built from extracted relations.

    As new patients are processed, the schema grows:
    - New entity types → new OWL classes
    - New predicates → new OWL properties
    - Observed type co-occurrences → subclass relations
    """

    def __init__(self):
        self.classes: dict[str, dict] = {}     # {name: {parent, count, description}}
        self.properties: dict[str, dict] = {}  # {name: {domain, range, count}}
        self.type_hierarchy: dict[str, str] = {}  # {child: parent}

        # Seed with FHIR base classes
        self._seed_fhir_hierarchy()

    def _seed_fhir_hierarchy(self):
        """Initialize with standard FHIR resource hierarchy."""
        base = {
            "Resource": {"parent": None, "count": 0},
            "Condition": {"parent": "Resource", "count": 0},
            "Observation": {"parent": "Resource", "count": 0},
            "Procedure": {"parent": "Resource", "count": 0},
            "MedicationStatement": {"parent": "Resource", "count": 0},
            "CarePlan": {"parent": "Resource", "count": 0},
            "FamilyMemberHistory": {"parent": "Resource", "count": 0},
            "AllergyIntolerance": {"parent": "Resource", "count": 0},
            # Clinical subtypes
            "Cancer": {"parent": "Condition", "count": 0},
            "BreastCancer": {"parent": "Cancer", "count": 0},
            "PancreaticCancer": {"parent": "Cancer", "count": 0},
            "Comorbidity": {"parent": "Condition", "count": 0},
            "Symptom": {"parent": "Condition", "count": 0},
            "Biomarker": {"parent": "Observation", "count": 0},
            "LabTest": {"parent": "Observation", "count": 0},
            "VitalSign": {"parent": "Observation", "count": 0},
            "ImagingStudy": {"parent": "Procedure", "count": 0},
            "Surgery": {"parent": "Procedure", "count": 0},
            "Biopsy": {"parent": "Procedure", "count": 0},
            "Chemotherapy": {"parent": "MedicationStatement", "count": 0},
            "SupportiveMed": {"parent": "MedicationStatement", "count": 0},
        }
        self.classes = base
        for name, info in base.items():
            if info["parent"]:
                self.type_hierarchy[name] = info["parent"]

    def update_from_relations(self, relations: list[dict]) -> None:
        """Update schema from extracted data-level relations."""
        for rel in relations:
            s_type = str(rel.get("subject_type", ""))
            o_type = str(rel.get("object_type", ""))
            pred = str(rel.get("predicate", ""))
            subject = str(rel.get("subject", "")).lower()

            # Auto-classify entities into schema classes
            if s_type:
                if s_type not in self.classes:
                    self.classes[s_type] = {"parent": "Resource", "count": 0}
                self.classes[s_type]["count"] += 1

            # Track predicate usage
            if pred:
                if pred not in self.properties:
                    self.properties[pred] = {
                        "domain": Counter(),
                        "range": Counter(),
                        "count": 0,
                    }
                self.properties[pred]["count"] += 1
                if s_type:
                    self.properties[pred]["domain"][s_type] += 1
                if o_type:
                    self.properties[pred]["range"][o_type] += 1

            # Infer subclass from entity names
            self._infer_subclass(subject, s_type)

    def _infer_subclass(self, entity_name: str, fhir_type: str):
        """Heuristic subclass inference from entity names."""
        cancer_keywords = {
            "breast cancer": "BreastCancer",
            "pancreatic": "PancreaticCancer",
            "ductal carcinoma": "BreastCancer",
            "adenocarcinoma": "Cancer",
            "carcinoma": "Cancer",
            "lymphoma": "Cancer",
            "melanoma": "Cancer",
        }
        for kw, cls in cancer_keywords.items():
            if kw in entity_name:
                if cls not in self.classes:
                    self.classes[cls] = {"parent": "Cancer", "count": 0}
                break

        biomarker_kw = ["er ", "pr ", "her2", "ki-67", "ki67", "ca19-9", "brca", "egfr"]
        if any(bm in entity_name for bm in biomarker_kw):
            if "Biomarker" not in self.classes:
                self.classes["Biomarker"] = {"parent": "Observation", "count": 0}

    def to_owl_graph(self) -> Graph:
        """Serialize schema as OWL ontology."""
        g = Graph()
        g.bind("tkg", TKG)
        g.bind("owl", OWL)
        g.bind("fhir", FHIR)

        for cls_name, info in self.classes.items():
            cls_uri = TKG[_safe_uri(cls_name)]
            g.add((cls_uri, RDF.type, OWL.Class))
            g.add((cls_uri, RDFS.label, Literal(cls_name)))

            parent = info.get("parent")
            if parent:
                g.add((cls_uri, RDFS.subClassOf, TKG[_safe_uri(parent)]))

        for prop_name, info in self.properties.items():
            prop_uri = TKG[_safe_uri(prop_name)]
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            g.add((prop_uri, RDFS.label, Literal(prop_name)))

            # Most common domain/range
            if info["domain"]:
                top_domain = info["domain"].most_common(1)[0][0]
                g.add((prop_uri, RDFS.domain, TKG[_safe_uri(top_domain)]))
            if info["range"]:
                top_range = info["range"].most_common(1)[0][0]
                g.add((prop_uri, RDFS.range, TKG[_safe_uri(top_range)]))

        return g

    def summary(self) -> dict[str, Any]:
        """Schema summary statistics."""
        return {
            "num_classes": len(self.classes),
            "num_properties": len(self.properties),
            "class_hierarchy_depth": self._max_depth(),
            "top_classes": sorted(
                [(k, v["count"]) for k, v in self.classes.items() if v["count"] > 0],
                key=lambda x: -x[1]
            )[:15],
            "top_properties": sorted(
                [(k, v["count"]) for k, v in self.properties.items()],
                key=lambda x: -x[1]
            )[:15],
        }

    def _max_depth(self) -> int:
        depth = 0
        for cls in self.classes:
            d = 0
            current = cls
            while current in self.type_hierarchy and d < 20:
                current = self.type_hierarchy[current]
                d += 1
            depth = max(depth, d)
        return depth


# ═══════════════════════════════════════════════════════════════
# DATA LEVEL (ABox) — patient instances with temporal validity
# ═══════════════════════════════════════════════════════════════

class TemporalDataGraph:
    """Patient-level data graph with temporal validity tracking.

    Each fact has a validity interval [t_start, t_end).
    Facts with t_end=None are considered currently valid.
    """

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.relations: list[dict] = []
        self.entity_index: dict[str, list[int]] = defaultdict(list)
        self.temporal_index: dict[str, list[int]] = defaultdict(list)

    def add_relation(self, rel: dict) -> int:
        """Add a relation to the data graph. Returns relation index."""
        idx = len(self.relations)
        self.relations.append(rel)

        subject = str(rel.get("subject", "")).lower()
        obj = str(rel.get("object", "")).lower()
        self.entity_index[subject].append(idx)
        self.entity_index[obj].append(idx)

        t_start = str(rel.get("temporal_start", ""))
        if t_start and t_start.lower() not in ("null", "none", ""):
            self.temporal_index[t_start].append(idx)

        return idx

    def add_relations_bulk(self, relations: list[dict]) -> None:
        """Add multiple relations efficiently."""
        for rel in relations:
            self.add_relation(rel)

    def get_entity_relations(self, entity: str) -> list[dict]:
        """Get all relations involving an entity."""
        entity = entity.lower()
        indices = self.entity_index.get(entity, [])
        return [self.relations[i] for i in indices]

    def get_temporal_snapshot(self, timestamp: str) -> list[dict]:
        """Get facts valid at a specific timestamp."""
        return [
            r for r in self.relations
            if self._fact_valid_at(r, timestamp)
        ]

    def get_temporal_evolution(self, entity: str) -> list[dict]:
        """Get the temporal evolution of an entity — all facts sorted by time."""
        rels = self.get_entity_relations(entity)
        # Sort by temporal_start
        return sorted(
            rels,
            key=lambda r: str(r.get("temporal_start", "9999")),
        )

    def detect_temporal_changes(self) -> list[dict]:
        """Find entities whose values change over time (evolution, not contradiction)."""
        changes = []
        # Group by (subject, predicate)
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for rel in self.relations:
            key = (
                str(rel.get("subject", "")).lower(),
                str(rel.get("predicate", "")).lower(),
            )
            groups[key].append(rel)

        for (subj, pred), rels in groups.items():
            if len(rels) < 2:
                continue
            # Check if values differ across time
            values = set()
            for r in rels:
                val = str(r.get("object", "")).lower()
                values.add(val)
            if len(values) > 1:
                changes.append({
                    "entity": subj,
                    "predicate": pred,
                    "values": list(values),
                    "timeline": sorted(
                        [(str(r.get("temporal_start", "")), str(r.get("object", ""))) for r in rels]
                    ),
                })

        return changes

    @staticmethod
    def _fact_valid_at(rel: dict, timestamp: str) -> bool:
        t_start = str(rel.get("temporal_start", ""))
        t_end = str(rel.get("temporal_end", ""))

        if not t_start or t_start.lower() in ("null", "none"):
            return True  # No temporal info — assume always valid

        if t_start > timestamp:
            return False  # Not yet valid

        if t_end and t_end.lower() not in ("null", "none", "ongoing", ""):
            if t_end <= timestamp:
                return False  # Expired

        return True

    def to_rdf(self, schema: DynamicSchema) -> Graph:
        """Convert data graph to RDF with temporal reification."""
        g = Graph()
        g.bind("tkg", TKG)
        g.bind("tkgi", TKGI)
        g.bind("tkgt", TKGT)
        g.bind("time", TIME)
        g.bind("fhir", FHIR)

        patient_uri = TKGI[_safe_uri(self.patient_id)]
        g.add((patient_uri, RDF.type, FHIR.Patient))
        g.add((patient_uri, RDFS.label, Literal(self.patient_id)))

        for i, rel in enumerate(self.relations):
            subj = str(rel.get("subject", ""))
            pred = str(rel.get("predicate", ""))
            obj = str(rel.get("object", ""))
            s_type = str(rel.get("subject_type", ""))
            t_start = str(rel.get("temporal_start", ""))
            t_end = str(rel.get("temporal_end", ""))

            if not subj or not obj:
                continue

            subj_uri = TKGI[_safe_uri(f"{self.patient_id}_{subj}")]
            pred_uri = TKG[_safe_uri(pred)] if pred else TKG.relatedTo

            # Object: URI for entities, Literal for values
            o_type = str(rel.get("object_type", ""))
            if o_type == "Literal" or not o_type:
                obj_node = Literal(obj)
            else:
                obj_node = TKGI[_safe_uri(f"{self.patient_id}_{obj}")]
                g.add((obj_node, RDF.type, TKG[_safe_uri(o_type)]))

            # Core triple
            g.add((subj_uri, pred_uri, obj_node))

            # Entity type
            if s_type:
                g.add((subj_uri, RDF.type, TKG[_safe_uri(s_type)]))

            # Link to patient
            g.add((patient_uri, TKG.hasEntity, subj_uri))

            # Temporal reification (if temporal info exists)
            if t_start and t_start.lower() not in ("null", "none", ""):
                stmt_uri = TKGT[_safe_uri(f"stmt_{self.patient_id}_{i}")]
                g.add((stmt_uri, RDF.type, TKG.TemporalStatement))
                g.add((stmt_uri, TKG.hasSubject, subj_uri))
                g.add((stmt_uri, TKG.hasPredicate, pred_uri))
                g.add((stmt_uri, TKG.hasObject, obj_node if isinstance(obj_node, Literal) else obj_node))
                g.add((stmt_uri, TKG.validFrom, Literal(t_start)))
                if t_end and t_end.lower() not in ("null", "none", ""):
                    g.add((stmt_uri, TKG.validUntil, Literal(t_end)))

            # Trust score
            trust = rel.get("_gnn_trust") or rel.get("_validation", {}).get("trust_score")
            if trust is not None:
                g.add((subj_uri, TKG.trustScore, Literal(float(trust), datatype=XSD.float)))

        return g

    def stats(self) -> dict[str, Any]:
        """Data graph statistics."""
        predicates = Counter(str(r.get("predicate", "")) for r in self.relations)
        s_types = Counter(str(r.get("subject_type", "")) for r in self.relations)
        temporal = sum(
            1 for r in self.relations
            if r.get("temporal_start") and str(r["temporal_start"]).lower() not in ("null", "none", "")
        )
        changes = self.detect_temporal_changes()

        return {
            "patient_id": self.patient_id,
            "num_relations": len(self.relations),
            "num_unique_entities": len(self.entity_index),
            "num_predicates": len(predicates),
            "temporal_coverage": round(temporal / max(len(self.relations), 1), 3),
            "temporal_changes": len(changes),
            "top_predicates": dict(predicates.most_common(10)),
            "type_distribution": dict(s_types.most_common()),
            "changes": changes[:5],  # Sample
        }


# ═══════════════════════════════════════════════════════════════
# CONVERSION: flat triples → hierarchical relations
# ═══════════════════════════════════════════════════════════════

def flat_triples_to_hierarchical(triples: list[dict]) -> list[dict]:
    """Convert flat EAV triples to hierarchical relation format.

    Maps existing fields to the hierarchical schema:
      entity → subject
      attribute → predicate
      value → object
      fhir_type → subject_type
      temporal_anchor → temporal_start
    """
    relations = []
    for t in triples:
        rel = {
            "subject": str(t.get("entity", "")),
            "predicate": str(t.get("attribute", "hasValue")),
            "object": str(t.get("value", "")),
            "subject_type": str(t.get("fhir_type", "")),
            "object_type": "Literal",
            "temporal_start": str(t.get("temporal_anchor", "")),
            "temporal_end": None,
            "evidence_span": str(t.get("evidence_span", "")),
        }
        # Preserve validation/trust metadata
        if "_validation" in t:
            rel["_validation"] = t["_validation"]
        if "_gnn_trust" in t:
            rel["_gnn_trust"] = t["_gnn_trust"]
        if "_consensus_level" in t:
            rel["_consensus_level"] = t["_consensus_level"]

        relations.append(rel)

    return relations


def build_hierarchical_kg(
    patient_id: str,
    triples: list[dict],
    schema: DynamicSchema | None = None,
) -> tuple[TemporalDataGraph, DynamicSchema]:
    """Build a hierarchical KG from extracted triples.

    Returns (data_graph, updated_schema).
    """
    if schema is None:
        schema = DynamicSchema()

    # Convert flat triples to hierarchical relations
    relations = flat_triples_to_hierarchical(triples)

    # Update schema from data
    schema.update_from_relations(relations)

    # Build data graph
    data_graph = TemporalDataGraph(patient_id)
    data_graph.add_relations_bulk(relations)

    temporal_changes = data_graph.detect_temporal_changes()
    if temporal_changes:
        logger.info(
            "%s: %d temporal changes detected (e.g., %s)",
            patient_id, len(temporal_changes),
            temporal_changes[0]["entity"] if temporal_changes else "none",
        )

    return data_graph, schema
