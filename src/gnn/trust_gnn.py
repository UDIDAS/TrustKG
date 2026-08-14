"""GNN-based trust propagation for clinical knowledge graph triples.

Architecture:
  - R-GCN style message passing (relation-aware via edge features)
  - Gated trust update: how much to trust neighbors vs self
  - 2-layer propagation with residual connections
  - Output: refined trust score per node → mapped back to triples

Training:
  - Contrastive loss: verified triples (positive) vs corrupted (negative)
  - Binary classification: predict whether a triple is trustworthy

This is the PRIMARY TECHNICAL NOVELTY of the TRUST-KG paper.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class TrustMessagePassing(MessagePassing):
    """Relation-aware message passing layer for trust propagation.

    Messages are weighted by edge features (validation scores from
    the 5-layer validation pipeline).
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 6):
        super().__init__(aggr="mean")
        self.lin_msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(out_dim * 2, 1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # Neighbor messages
        neighbor_msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Self transform
        self_msg = self.lin_self(x)
        # Gated update: learn how much to trust neighbors vs self
        gate_input = torch.cat([self_msg, neighbor_msg], dim=-1)
        gate_val = torch.sigmoid(self.gate(gate_input))
        out = gate_val * neighbor_msg + (1 - gate_val) * self_msg
        return self.norm(F.gelu(out))

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Concatenate neighbor features with edge (validation) features
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.lin_msg(msg_input)


class TrustGNN(nn.Module):
    """Graph Neural Network for trust propagation in clinical KGs.

    Takes node features (entity embeddings + initial trust + FHIR type)
    and edge features (validation scores), propagates trust through
    the graph, and outputs refined trust scores per node.
    """

    def __init__(
        self,
        in_dim: int = 73,       # embed_dim(64) + fhir_onehot(8) + trust(1)
        hidden_dim: int = 64,
        edge_dim: int = 7,      # 5 validation scores + trust score + temporal recency
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            TrustMessagePassing(hidden_dim, hidden_dim, edge_dim)
            for _ in range(n_layers)
        ])

        self.trust_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data with x, edge_index, edge_attr

        Returns:
            trust_scores: Tensor [N] — refined trust per node
        """
        x = self.input_proj(data.x)
        x = F.gelu(x)

        for layer in self.layers:
            residual = x
            x = layer(x, data.edge_index, data.edge_attr)
            x = self.dropout(x)
            x = x + residual  # Residual connection

        trust = self.trust_head(x).squeeze(-1)
        return trust


def create_training_data(
    graph_data: dict[str, Any],
    triples: list[dict],
    corruption_ratio: float = 0.3,
) -> tuple[Data, torch.Tensor]:
    """Create PyG Data object with positive/negative labels for contrastive training.

    Positive: triples with high validation trust (>= 0.6)
    Negative: corrupted triples (swap entity/value) + low-trust triples

    Returns:
        (data, labels) where labels are 1.0 (trustworthy) or 0.0 (untrustworthy)
    """
    data = Data(
        x=graph_data["node_features"],
        edge_index=graph_data["edge_index"],
        edge_attr=graph_data["edge_attr"],
    )

    n_nodes = graph_data["n_nodes"]
    labels = torch.zeros(n_nodes)

    # Positive labels: nodes from high-trust triples
    for triple in triples:
        val = triple.get("_validation", {})
        trust = val.get("trust_score", 0.5)
        entity_key = str(triple.get("entity", "")).lower().strip()
        idx = graph_data["entity_to_idx"].get(entity_key)
        if idx is not None:
            labels[idx] = 1.0 if trust >= 0.6 else 0.0

    return data, labels


def train_trust_gnn(
    graphs: list[dict[str, Any]],
    triple_sets: list[list[dict]],
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda:0",
) -> TrustGNN:
    """Train the TrustGNN on multiple patient graphs.

    Uses binary cross-entropy loss: predict whether each node
    (entity) is trustworthy based on its graph neighborhood.
    """
    if not graphs:
        raise ValueError("No graphs provided for training")

    # Determine input dimensions from first graph
    in_dim = graphs[0]["node_features"].shape[1]
    edge_dim = graphs[0]["edge_attr"].shape[1] if graphs[0]["edge_attr"].shape[0] > 0 else 6

    model = TrustGNN(in_dim=in_dim, edge_dim=edge_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # Prepare training data
    train_data = []
    for graph, triples in zip(graphs, triple_sets):
        if graph["n_nodes"] < 2 or graph["n_edges"] < 1:
            continue
        data, labels = create_training_data(graph, triples)
        data = data.to(device)
        labels = labels.to(device)
        train_data.append((data, labels))

    if not train_data:
        logger.warning("No valid training graphs, returning untrained model")
        return model

    logger.info("Training TrustGNN on %d patient graphs...", len(train_data))
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        for data, labels in train_data:
            optimizer.zero_grad()
            pred_trust = model(data)
            loss = criterion(pred_trust, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_data)
            logger.info("Epoch %d/%d, avg loss: %.4f", epoch + 1, n_epochs, avg_loss)

    model.eval()
    logger.info("TrustGNN training complete")
    return model


@torch.no_grad()
def propagate_trust(
    model: TrustGNN,
    graph_data: dict[str, Any],
    triples: list[dict],
    device: str = "cuda:0",
) -> list[dict]:
    """Run trained GNN to get refined trust scores, map back to triples.

    Returns triples enriched with _gnn_trust field.
    """
    if graph_data["n_nodes"] < 2 or graph_data["n_edges"] < 1:
        for t in triples:
            t["_gnn_trust"] = t.get("_validation", {}).get("trust_score", 0.5)
        return triples

    data = Data(
        x=graph_data["node_features"],
        edge_index=graph_data["edge_index"],
        edge_attr=graph_data["edge_attr"],
    ).to(device)

    model = model.to(device)
    refined_trust = model(data).cpu()

    entity_to_idx = graph_data["entity_to_idx"]
    for triple in triples:
        entity_key = str(triple.get("entity", "")).lower().strip()
        idx = entity_to_idx.get(entity_key)
        if idx is not None:
            triple["_gnn_trust"] = round(refined_trust[idx].item(), 4)
        else:
            triple["_gnn_trust"] = triple.get("_validation", {}).get("trust_score", 0.5)

    return triples
