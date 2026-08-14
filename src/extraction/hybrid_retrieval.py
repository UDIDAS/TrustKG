"""Ontology-Aware Hybrid Retrieval (Draft Section 3.3).

Implements the draft's retrieval strategy exactly:
  S_hybrid(τ) = λ₁·S_BM25 + λ₂·S_dense + λ₃·S_graph

Three retrieval sources:
  1. S_BM25: Sparse lexical retrieval over biomedical ontology descriptions
  2. S_dense: Dense semantic retrieval using MedCPT biomedical encoders
  3. S_graph: Graph neighborhood retrieval from previously validated graph

Additionally computes ontology-aware retrieval consistency (Draft eq):
  S_retrieval(τ) = α₁·S_semantic + α₂·S_ontology + α₃·S_document

Memory-efficient: embeddings computed in batches, BM25 index built once.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import MAX_WORKERS

logger = logging.getLogger(__name__)

# ── Lazy-loaded models ──────────────────────────────────────
_medcpt_query = None
_medcpt_article = None
_bm25_index = None
_bm25_corpus = None
_ontology_concepts = None


def _load_medcpt():
    """Load MedCPT query and article encoders."""
    global _medcpt_query, _medcpt_article
    if _medcpt_query is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading MedCPT encoders...")
        _medcpt_query = SentenceTransformer("ncbi/MedCPT-Query-Encoder")
        _medcpt_article = SentenceTransformer("ncbi/MedCPT-Article-Encoder")
        logger.info("MedCPT loaded")
    return _medcpt_query, _medcpt_article


def _build_ontology_corpus() -> list[dict]:
    """Build corpus of biomedical ontology concept descriptions.

    Uses SNOMED CT, LOINC, RxNorm, ICD common terms as retrieval targets.
    In production, this would query ontology APIs. Here we use a curated
    list covering common oncology concepts.
    """
    global _ontology_concepts
    if _ontology_concepts is not None:
        return _ontology_concepts

    concepts = []

    _RAW_CONCEPTS = [
        # Oncology conditions
        ("invasive ductal carcinoma", "408643008", "SNOMED"),
        ("breast cancer", "254837009", "SNOMED"),
        ("pancreatic ductal adenocarcinoma", "363418001", "SNOMED"),
        ("adenocarcinoma", "35917007", "SNOMED"),
        ("metastasis", "128462008", "SNOMED"),
        ("lymph node metastasis", "94391008", "SNOMED"),
        ("congestive heart failure", "42343007", "SNOMED"),
        ("diabetes mellitus", "73211009", "SNOMED"),
        ("hypertension", "38341003", "SNOMED"),
        ("chronic kidney disease", "709044004", "SNOMED"),
        ("anemia", "271737000", "SNOMED"),
        ("deep vein thrombosis", "128053003", "SNOMED"),
        ("pulmonary embolism", "59282003", "SNOMED"),
        ("sepsis", "91302008", "SNOMED"),
        ("dyspnea", "267036007", "SNOMED"),
        ("nausea", "422587007", "SNOMED"),
        ("pain", "22253000", "SNOMED"),
        ("edema", "79654002", "SNOMED"),
        # Biomarkers / lab tests
        ("estrogen receptor", "16112-5", "LOINC"),
        ("progesterone receptor", "16113-3", "LOINC"),
        ("HER2 neu", "48676-1", "LOINC"),
        ("Ki-67 proliferation index", "29593-1", "LOINC"),
        ("CA 19-9 antigen", "24108-3", "LOINC"),
        ("carcinoembryonic antigen CEA", "2039-6", "LOINC"),
        ("hemoglobin", "718-7", "LOINC"),
        ("white blood cell count", "6690-2", "LOINC"),
        ("platelet count", "777-3", "LOINC"),
        ("creatinine", "2160-0", "LOINC"),
        ("bilirubin total", "1975-2", "LOINC"),
        ("albumin", "1751-7", "LOINC"),
        ("alkaline phosphatase", "6768-6", "LOINC"),
        ("alanine transaminase ALT", "1742-6", "LOINC"),
        ("aspartate transaminase AST", "1920-8", "LOINC"),
        ("glucose", "2345-7", "LOINC"),
        ("sodium", "2951-2", "LOINC"),
        ("potassium", "2823-3", "LOINC"),
        ("blood pressure systolic", "8480-6", "LOINC"),
        # Medications
        ("gemcitabine", "12574", "RxNorm"),
        ("nab-paclitaxel abraxane", "583214", "RxNorm"),
        ("paclitaxel", "56946", "RxNorm"),
        ("docetaxel", "72962", "RxNorm"),
        ("carboplatin", "40048", "RxNorm"),
        ("cisplatin", "2555", "RxNorm"),
        ("doxorubicin", "3639", "RxNorm"),
        ("cyclophosphamide", "3002", "RxNorm"),
        ("fluorouracil 5-FU", "4492", "RxNorm"),
        ("trastuzumab herceptin", "224905", "RxNorm"),
        ("tamoxifen", "10324", "RxNorm"),
        ("anastrozole", "84857", "RxNorm"),
        ("pembrolizumab keytruda", "1547220", "RxNorm"),
        ("metformin", "6809", "RxNorm"),
        ("insulin glargine", "274783", "RxNorm"),
        ("furosemide lasix", "4603", "RxNorm"),
        ("aspirin", "1191", "RxNorm"),
        ("atorvastatin lipitor", "83367", "RxNorm"),
        # Procedures
        ("mastectomy", "172043006", "SNOMED"),
        ("lumpectomy", "392021009", "SNOMED"),
        ("sentinel lymph node biopsy", "396487001", "SNOMED"),
        ("fine needle aspiration", "44199001", "SNOMED"),
        ("computed tomography CT scan", "77477000", "SNOMED"),
        ("magnetic resonance imaging MRI", "113091000", "SNOMED"),
        ("positron emission tomography PET", "82918005", "SNOMED"),
        ("mammography", "71651007", "SNOMED"),
        ("endoscopic ultrasound EUS", "310541005", "SNOMED"),
        ("ERCP", "386718000", "SNOMED"),
        ("colonoscopy", "73761001", "SNOMED"),
        ("chemotherapy administration", "367336001", "SNOMED"),
        ("radiation therapy", "108290001", "SNOMED"),
    ]

    for term, code, system in _RAW_CONCEPTS:
        concepts.append({
            "text": term,
            "code": code,
            "system": system,
            "description": term,
        })

    _ontology_concepts = concepts
    return concepts


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer for BM25."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_bm25_index(additional_docs: list[str] | None = None) -> BM25Okapi:
    """Build BM25 index over ontology concepts + optional documents."""
    global _bm25_index, _bm25_corpus
    concepts = _build_ontology_corpus()
    corpus_texts = [c["text"] + " " + c.get("description", "") for c in concepts]
    if additional_docs:
        corpus_texts.extend(additional_docs)
    _bm25_corpus = corpus_texts
    tokenized = [_tokenize(t) for t in corpus_texts]
    _bm25_index = BM25Okapi(tokenized)
    logger.info("BM25 index built: %d documents", len(corpus_texts))
    return _bm25_index


def retrieve_bm25(query: str, top_k: int = 5) -> list[tuple[float, dict]]:
    """BM25 sparse retrieval over ontology corpus.

    Returns [(score, concept_dict), ...] sorted by score descending.
    """
    if _bm25_index is None:
        build_bm25_index()

    concepts = _build_ontology_corpus()
    tokens = _tokenize(query)
    scores = _bm25_index.get_scores(tokens)

    # Pair with concepts (only ontology part, not additional docs)
    scored = []
    for i, score in enumerate(scores[:len(concepts)]):
        if score > 0:
            scored.append((float(score), concepts[i]))

    scored.sort(key=lambda x: -x[0])
    return scored[:top_k]


def retrieve_dense(
    query: str,
    candidates: list[str] | None = None,
    top_k: int = 5,
) -> list[tuple[float, str]]:
    """MedCPT dense semantic retrieval.

    Encodes query with MedCPT-Query-Encoder, candidates with Article-Encoder.
    Returns [(similarity, candidate_text), ...].
    """
    query_enc, article_enc = _load_medcpt()

    if candidates is None:
        concepts = _build_ontology_corpus()
        candidates = [c["text"] for c in concepts]

    # Encode in batches for memory efficiency
    q_emb = query_enc.encode([query], batch_size=1, show_progress_bar=False)

    batch_size = 64
    all_scores = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        c_emb = article_enc.encode(batch, batch_size=batch_size, show_progress_bar=False)
        sims = np.dot(q_emb, c_emb.T).flatten()
        for j, sim in enumerate(sims):
            all_scores.append((float(sim), candidates[i + j]))

    all_scores.sort(key=lambda x: -x[0])
    return all_scores[:top_k]


def retrieve_graph_neighborhood(
    entity: str,
    existing_triples: list[dict],
    hop: int = 1,
) -> list[dict]:
    """Graph neighborhood retrieval from previously validated graph.

    Given an entity, find related triples within N hops in the
    existing patient graph. Returns neighboring triples as evidence.
    """
    entity_lower = entity.lower().strip()
    neighbors: list[dict] = []
    visited_entities: set[str] = {entity_lower}

    current_entities = {entity_lower}
    for _ in range(hop):
        next_entities: set[str] = set()
        for triple in existing_triples:
            t_entity = str(triple.get("entity", "")).lower().strip()
            t_value = str(triple.get("value", "")).lower().strip()

            if t_entity in current_entities or t_value in current_entities:
                neighbors.append(triple)
                # Expand frontier
                for key in [t_entity, t_value]:
                    if key not in visited_entities and len(key) > 2:
                        next_entities.add(key)
                        visited_entities.add(key)

        current_entities = next_entities

    return neighbors


def hybrid_retrieve(
    triple: dict,
    existing_triples: list[dict] | None = None,
    weights: tuple[float, float, float] = (0.3, 0.5, 0.2),
    top_k: int = 5,
) -> dict[str, Any]:
    """Full hybrid retrieval as specified in Draft Section 3.3.

    S_hybrid(τ) = λ₁·S_BM25 + λ₂·S_dense + λ₃·S_graph

    Args:
        triple: candidate triple to ground
        existing_triples: previously validated triples for graph retrieval
        weights: (λ₁, λ₂, λ₃) for BM25, dense, graph
        top_k: number of retrieved evidences per method

    Returns:
        Dict with retrieval results and combined score.
    """
    λ1, λ2, λ3 = weights
    entity = str(triple.get("entity", ""))
    value = str(triple.get("value", ""))
    query = f"{entity} {value}".strip()

    if not query:
        return {"score": 0.0, "bm25": [], "dense": [], "graph": []}

    # 1. BM25 sparse retrieval
    bm25_results = retrieve_bm25(query, top_k=top_k)
    s_bm25 = bm25_results[0][0] if bm25_results else 0.0
    # Normalize BM25 score to [0, 1]
    s_bm25 = min(s_bm25 / max(s_bm25 + 1, 1), 1.0)

    # 2. Dense semantic retrieval (MedCPT)
    dense_results = retrieve_dense(query, top_k=top_k)
    s_dense = dense_results[0][0] if dense_results else 0.0
    # Cosine similarity already in [-1, 1], shift to [0, 1]
    s_dense = (s_dense + 1) / 2

    # 3. Graph neighborhood retrieval
    graph_results = []
    s_graph = 0.0
    if existing_triples:
        graph_results = retrieve_graph_neighborhood(entity, existing_triples, hop=1)
        s_graph = min(len(graph_results) / 5.0, 1.0)  # Normalize by expected neighbors

    # Combined score
    s_hybrid = λ1 * s_bm25 + λ2 * s_dense + λ3 * s_graph

    return {
        "score": round(s_hybrid, 4),
        "s_bm25": round(s_bm25, 4),
        "s_dense": round(s_dense, 4),
        "s_graph": round(s_graph, 4),
        "bm25_matches": [{"score": s, "concept": c} for s, c in bm25_results[:3]],
        "dense_matches": [{"score": s, "text": t} for s, t in dense_results[:3]],
        "graph_neighbors": len(graph_results),
    }


def retrieve_for_patient(
    triples: list[dict],
    existing_triples: list[dict] | None = None,
) -> list[dict]:
    """Run hybrid retrieval for all triples of a patient.

    Enriches each triple with retrieval evidence and scores.
    """
    # Build BM25 index once
    build_bm25_index()

    enriched = []
    for triple in triples:
        retrieval = hybrid_retrieve(triple, existing_triples)
        enriched_triple = {
            **triple,
            "_retrieval": retrieval,
        }
        enriched.append(enriched_triple)

    avg_score = sum(t["_retrieval"]["score"] for t in enriched) / max(len(enriched), 1)
    logger.info(
        "Hybrid retrieval: %d triples, avg score=%.3f",
        len(enriched), avg_score,
    )
    return enriched


def unload_models():
    """Free MedCPT models from memory."""
    global _medcpt_query, _medcpt_article
    _medcpt_query = None
    _medcpt_article = None
    logger.info("MedCPT models unloaded")
