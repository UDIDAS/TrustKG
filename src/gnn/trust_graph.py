"""Build PyG graph from validated EAV triples for GNN trust propagation.

Converts patient-level extracted triples into a heterogeneous graph where:
  - Nodes = unique biomedical entities
  - Edges = relations between entities (from triples)
  - Node features = [entity_embedding, initial_trust, fhir_type_onehot]
  - Edge features = [relation_embedding, validation_scores]

Memory-efficient: processes one patient at a time.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
import numpy as np

logger = logging.getLogger(__name__)

# FHIR type to index mapping
FHIR_TYPES = [
    "Condition", "Observation", "Procedure", "MedicationStatement",
    "CarePlan", "FamilyMemberHistory", "AllergyIntolerance", "Unknown",
]
FHIR_TO_IDX = {t: i for i, t in enumerate(FHIR_TYPES)}


def _get_entity_key(triple: dict) -> str:
    """Normalized entity key for node deduplication."""
    entity = str(triple.get("entity", "")).lower().strip()
    return entity


def _compute_temporal_recency(triple: dict) -> float:
    """Compute temporal recency score [0, 1]. Recent facts score higher.

    Uses normalized date if available, falls back to temporal_anchor.
    Facts without dates get 0.5 (neutral).
    """
    import re
    date_str = triple.get("temporal_normalized") or str(triple.get("temporal_anchor", ""))
    if not date_str or date_str.lower() in ("null", "none", ""):
        return 0.5

    # Try to parse year from date
    year_match = re.search(r"(19|20)\d{2}", date_str)
    if not year_match:
        return 0.5

    year = int(year_match.group())
    # Exponential decay: 2026=1.0, 2020=0.7, 2015=0.5, 2010=0.35, 2000=0.15
    recency = max(0.0, min(1.0, 1.0 - (2026 - year) * 0.05))
    return recency


def _get_value_key(triple: dict) -> str:
    """Normalized value key — values can also be nodes."""
    value = str(triple.get("value", "")).lower().strip()
    return value


def build_trust_graph(
    triples: list[dict],
    entity_embeddings: dict[str, np.ndarray] | None = None,
    embed_dim: int = 64,
) -> dict[str, Any]:
    """Build a PyG-compatible graph from validated triples.

    Args:
        triples: list of validated triples (with _validation scores)
        entity_embeddings: optional pre-computed entity embeddings
        embed_dim: embedding dimension for random init if no embeddings

    Returns:
        dict with:
          - node_features: Tensor [N, feat_dim]
          - edge_index: Tensor [2, E]
          - edge_features: Tensor [E, edge_feat_dim]
          - initial_trust: Tensor [N]
          - node_labels: list of entity names
          - triple_to_edge: mapping from triple index to edge index
    """
    # Step 1: Collect unique entities (nodes)
    entity_to_idx: dict[str, int] = {}
    node_labels: list[str] = []

    for triple in triples:
        for key_fn in [_get_entity_key, _get_value_key]:
            key = key_fn(triple)
            if key and key not in entity_to_idx and len(key) > 1:
                entity_to_idx[key] = len(entity_to_idx)
                node_labels.append(key)

    n_nodes = len(entity_to_idx)
    if n_nodes == 0:
        return _empty_graph()

    # Step 2: Build edges from triples
    src_nodes: list[int] = []
    dst_nodes: list[int] = []
    edge_features_list: list[list[float]] = []
    triple_to_edge: dict[int, int] = {}

    for t_idx, triple in enumerate(triples):
        e_key = _get_entity_key(triple)
        v_key = _get_value_key(triple)

        if e_key not in entity_to_idx or v_key not in entity_to_idx:
            continue

        src = entity_to_idx[e_key]
        dst = entity_to_idx[v_key]

        if src == dst:
            continue

        # Edge features from validation scores + temporal recency
        val = triple.get("_validation", {})
        temporal_recency = _compute_temporal_recency(triple)
        edge_feat = [
            val.get("source_grounding", 0.5),
            val.get("ontology_check", 0.5),
            val.get("schema_check", 0.5),
            val.get("temporal_consistency", 0.5),
            val.get("contradiction_score", 0.5),
            val.get("trust_score", 0.5),
            temporal_recency,  # newer facts get higher weight
        ]

        # Bidirectional edges
        src_nodes.extend([src, dst])
        dst_nodes.extend([dst, src])
        edge_features_list.extend([edge_feat, edge_feat])
        triple_to_edge[t_idx] = len(src_nodes) - 2

    # Step 3: Node features
    # [entity_embedding (embed_dim) | fhir_type_onehot (8) | initial_trust (1)]
    feat_dim = embed_dim + len(FHIR_TYPES) + 1
    node_features = torch.zeros(n_nodes, feat_dim)

    # Assign FHIR types and initial trust from triples
    node_trust = torch.full((n_nodes,), 0.5)
    node_fhir_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for triple in triples:
        e_key = _get_entity_key(triple)
        if e_key in entity_to_idx:
            idx = entity_to_idx[e_key]
            fhir = triple.get("fhir_type", "Unknown")
            node_fhir_counts[idx][fhir] += 1
            # Update trust with validation score
            val = triple.get("_validation", {})
            trust = val.get("trust_score", 0.5)
            node_trust[idx] = max(node_trust[idx], trust)

    for idx in range(n_nodes):
        # Entity embedding: random init or pre-computed
        if entity_embeddings and node_labels[idx] in entity_embeddings:
            emb = entity_embeddings[node_labels[idx]][:embed_dim]
            node_features[idx, :len(emb)] = torch.tensor(emb, dtype=torch.float)
        else:
            node_features[idx, :embed_dim] = torch.randn(embed_dim) * 0.1

        # FHIR type one-hot (majority vote if multiple)
        if idx in node_fhir_counts:
            top_fhir = max(node_fhir_counts[idx], key=node_fhir_counts[idx].get)
            fhir_idx = FHIR_TO_IDX.get(top_fhir, FHIR_TO_IDX["Unknown"])
            node_features[idx, embed_dim + fhir_idx] = 1.0

        # Initial trust score
        node_features[idx, -1] = node_trust[idx]

    # Step 4: Assemble tensors
    if src_nodes:
        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 7, dtype=torch.float)

    logger.info(
        "Built graph: %d nodes, %d edges from %d triples",
        n_nodes, edge_index.shape[1], len(triples),
    )

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "initial_trust": node_trust,
        "node_labels": node_labels,
        "entity_to_idx": entity_to_idx,
        "triple_to_edge": triple_to_edge,
        "n_nodes": n_nodes,
        "n_edges": edge_index.shape[1],
    }


def _empty_graph() -> dict[str, Any]:
    return {
        "node_features": torch.zeros(0, 73),
        "edge_index": torch.zeros(2, 0, dtype=torch.long),
        "edge_attr": torch.zeros(0, 7),
        "initial_trust": torch.zeros(0),
        "node_labels": [],
        "entity_to_idx": {},
        "triple_to_edge": {},
        "n_nodes": 0,
        "n_edges": 0,
    }
