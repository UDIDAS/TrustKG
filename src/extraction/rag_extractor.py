"""Retrieval-Augmented EAV extraction (TRUST-KG pipeline).

Architecture:
  Step 1: scispacy NER → entity candidates (CPU parallel)
  Step 2: Candidate formatting + section-aware grouping
  Step 3: LLM extraction with candidates as grounding context
  Step 4: Merge NER-grounded triples with any new LLM-discovered triples

This replaces bare LLM extraction with retrieval-augmented extraction,
feeding the LLM a structured candidate list so it covers more entities.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.config import EXTRACTION_DIR, CHUNK_MAX_CHARS, MAX_WORKERS
from src.data.reader import ClinicalDocument
from src.extraction.ner import (
    extract_entities,
    extract_entities_batch,
    format_candidate_list,
    EntityMention,
)
from src.extraction.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_EXTRACTION_PROMPT,
    RAG_CHUNKED_PROMPT,
)

logger = logging.getLogger(__name__)


def _chunk_with_candidates(
    text: str,
    mentions: list[EntityMention],
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[tuple[str, str]]:
    """Split text into chunks, each with its corresponding NER candidate list.

    Returns list of (chunk_text, candidate_list_str) tuples.
    """
    if len(text) <= max_chars:
        cand_str = format_candidate_list(mentions)
        return [(text, cand_str)]

    chunks: list[tuple[str, str]] = []
    paragraphs = text.split("\n")
    current_lines: list[str] = []
    current_len = 0
    chunk_start = 0

    for para in paragraphs:
        para_len = len(para) + 1
        if current_len + para_len > max_chars and current_lines:
            chunk_text = "\n".join(current_lines)
            chunk_end = chunk_start + len(chunk_text)
            # Filter mentions that fall within this chunk's span
            chunk_mentions = [
                m for m in mentions
                if m.start >= chunk_start and m.end <= chunk_end + 50
            ]
            cand_str = format_candidate_list(chunk_mentions)
            chunks.append((chunk_text, cand_str))
            chunk_start = chunk_end + 1
            current_lines = [para]
            current_len = para_len
        else:
            current_lines.append(para)
            current_len += para_len

    if current_lines:
        chunk_text = "\n".join(current_lines)
        chunk_end = chunk_start + len(chunk_text)
        chunk_mentions = [
            m for m in mentions
            if m.start >= chunk_start and m.end <= chunk_end + 50
        ]
        cand_str = format_candidate_list(chunk_mentions)
        chunks.append((chunk_text, cand_str))

    return chunks


def _retrieve_ontology_context(
    chunk_mentions: list[EntityMention],
    existing_triples: list[dict],
    top_k: int = 5,
) -> str:
    """Run BM25 + MedCPT retrieval on NER candidates to get ontology grounding.

    This is the core of Draft Section 3.3: retrieval DURING extraction.
    Returns formatted ontology context to inject into the extraction prompt.
    """
    from src.extraction.hybrid_retrieval import retrieve_bm25, retrieve_dense, retrieve_graph_neighborhood

    seen = set()
    ontology_lines = ["\n## Ontology-Grounded Concepts (from BM25 + MedCPT retrieval):"]

    for mention in chunk_mentions[:30]:  # Cap to avoid slow retrieval
        query = mention.text
        norm = query.lower().strip()
        if norm in seen or len(norm) < 3:
            continue
        seen.add(norm)

        # BM25 sparse retrieval against ontology corpus
        bm25_hits = retrieve_bm25(query, top_k=2)
        for score, concept in bm25_hits:
            if score > 0.5:
                ontology_lines.append(
                    f'  - "{concept["text"]}" [{concept["system"]}:{concept["code"]}] '
                    f"(matched: \"{query}\")"
                )

        # Dense semantic retrieval (MedCPT)
        dense_hits = retrieve_dense(query, top_k=1)
        for score, matched_text in dense_hits:
            if score > 20 and matched_text.lower() not in seen:
                ontology_lines.append(
                    f'  - "{matched_text}" [MedCPT sim={score:.0f}] (matched: "{query}")'
                )

    # Graph neighborhood: evidence from already-extracted triples
    if existing_triples:
        graph_lines = []
        for mention in chunk_mentions[:10]:
            neighbors = retrieve_graph_neighborhood(mention.text, existing_triples, hop=1)
            for n in neighbors[:2]:
                ent = n.get("entity", "")
                val = n.get("value", "")
                if ent and val:
                    graph_lines.append(f'  - Prior: ({ent}, {n.get("attribute","")}, {str(val)[:40]})')
        if graph_lines:
            ontology_lines.append("\n## Prior Graph Evidence:")
            ontology_lines.extend(graph_lines[:10])

    if len(ontology_lines) <= 1:
        return ""  # No retrieval results
    return "\n".join(ontology_lines)


def _extract_document_rag(
    doc: ClinicalDocument,
    model_name: str,
    gpu_id: int,
    mentions: list[EntityMention],
    seed_triples: list[dict] | None = None,
) -> dict[str, Any]:
    """Run retrieval-augmented extraction for one document on one model.

    Key difference from bare LLM: BM25 + MedCPT retrieval runs DURING
    extraction (not post-hoc), providing ontology-grounded concepts
    in the prompt. This matches Draft Section 3.3.

    seed_triples: Optional pre-existing KG triples (e.g. from train split)
        to use as graph neighborhood evidence during extraction.
    """
    from src.extraction.local_llm import extract_json

    chunks = _chunk_with_candidates(doc.text, mentions)
    # Seed with train-KG triples for graph neighborhood retrieval
    all_triples: list[dict[str, Any]] = list(seed_triples) if seed_triples else []
    n_seed = len(all_triples)

    for i, (chunk_text, cand_str) in enumerate(chunks):
        # Retrieve ontology context for this chunk's candidates
        chunk_mentions = [m for m in mentions
                          if any(m.text.lower() in cand_str.lower() for _ in [1])]
        ontology_context = _retrieve_ontology_context(
            chunk_mentions[:30], all_triples,  # pass already-extracted as graph evidence
        )

        # Build prompt with NER candidates + ontology retrieval
        full_candidates = cand_str
        if ontology_context:
            full_candidates = cand_str + "\n" + ontology_context

        if i == 0:
            prompt = RAG_EXTRACTION_PROMPT.format(
                candidates=full_candidates, narrative=chunk_text,
            )
        else:
            prompt = RAG_CHUNKED_PROMPT.format(
                candidates=full_candidates, narrative=chunk_text,
            )

        triples = extract_json(model_name, RAG_SYSTEM_PROMPT, prompt, gpu_id)
        all_triples.extend(triples)
        logger.info(
            "  [RAG/%s/GPU%d] %s chunk %d/%d: %d triples (%d candidates + %d ontology)",
            model_name, gpu_id, doc.patient_id,
            i + 1, len(chunks), len(triples),
            cand_str.count('- "'), ontology_context.count('- "'),
        )

    # Return only newly extracted triples (not seed)
    new_triples = all_triples[n_seed:]
    return {
        "model": model_name,
        "mode": "rag",
        "patient_id": doc.patient_id,
        "num_triples": len(new_triples),
        "num_ner_candidates": len(mentions),
        "triples": new_triples,
        "seed_triples_used": n_seed,
    }


class RAGExtractor:
    """Retrieval-Augmented EAV extraction using NER + LLM."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or EXTRACTION_DIR / "rag"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_batch(
        self,
        docs: list[ClinicalDocument],
        model_name: str = "gemma4-4b",
        gpu_id: int = 0,
        seed_triples: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract from all documents using RAG pipeline.

        Step 1: NER on all docs (CPU parallel)
        Step 2: LLM extraction with candidates (GPU sequential per model)

        seed_triples: Pre-existing KG triples (e.g. from train split) to seed
            graph neighborhood retrieval during extraction.
        """
        import os
        os.environ.setdefault("HF_TOKEN", "")

        # Step 1: NER (CPU parallel via ThreadPool)
        logger.info("Step 1: Running NER on %d documents (CPU parallel)...", len(docs))
        t0 = time.time()
        all_mentions = extract_entities_batch([d.text for d in docs])
        ner_time = time.time() - t0
        total_mentions = sum(len(m) for m in all_mentions)
        logger.info(
            "NER complete: %d mentions from %d docs in %.1fs",
            total_mentions, len(docs), ner_time,
        )

        # Step 2: RAG extraction (GPU)
        logger.info("Step 2: RAG extraction with %s on GPU %d...", model_name, gpu_id)
        results: list[dict[str, Any]] = []

        for doc, mentions in zip(docs, all_mentions):
            result = _extract_document_rag(doc, model_name, gpu_id, mentions, seed_triples=seed_triples)
            result["cohort"] = doc.cohort
            result["text_length"] = len(doc.text)

            # Write to disk immediately
            model_dir = self.output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            out_path = model_dir / f"{doc.patient_id}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            results.append(result)
            logger.info(
                "RAG Completed %s: %d triples (from %d NER candidates)",
                doc.patient_id, result["num_triples"], len(mentions),
            )

        return results

    def extract_all_models(
        self,
        docs: list[ClinicalDocument],
        model_pairs: list[tuple[str, int]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run RAG extraction with multiple models.

        Pre-computes NER once, then feeds candidates to each model.
        Uses 2 GPUs in parallel per pass.
        """
        import os
        os.environ.setdefault("HF_TOKEN", "")

        if model_pairs is None:
            model_pairs = [
                ("qwen3-8b", 0),
                ("gemma4-4b", 1),
                ("llama32-3b", 0),
            ]

        # Step 1: NER once (CPU parallel)
        logger.info("Step 1: Running NER on %d documents...", len(docs))
        t0 = time.time()
        all_mentions = extract_entities_batch([d.text for d in docs])
        logger.info("NER complete: %d mentions in %.1fs",
                     sum(len(m) for m in all_mentions), time.time() - t0)

        # Step 2: RAG extraction per model
        from src.extraction.local_llm import load_model, unload_all
        all_results: dict[str, list[dict[str, Any]]] = {}

        # Group models by GPU pass
        gpu0_models = [m for m, g in model_pairs if g == 0]
        gpu1_models = [m for m, g in model_pairs if g == 1]
        passes = list(zip(
            gpu0_models + [None] * max(0, len(gpu1_models) - len(gpu0_models)),
            gpu1_models + [None] * max(0, len(gpu0_models) - len(gpu1_models)),
        ))

        for pass_idx, (m0, m1) in enumerate(passes):
            logger.info("=== RAG PASS %d/%d ===", pass_idx + 1, len(passes))

            # Pre-load models
            if m0:
                try:
                    load_model(m0, gpu_id=0)
                except Exception as e:
                    logger.error("Failed to load %s: %s", m0, e)
                    m0 = None
            if m1:
                try:
                    load_model(m1, gpu_id=1)
                except Exception as e:
                    logger.error("Failed to load %s: %s", m1, e)
                    m1 = None

            # Run in parallel
            def _run(model_name, gpu_id):
                results = []
                for doc, mentions in zip(docs, all_mentions):
                    r = _extract_document_rag(doc, model_name, gpu_id, mentions)
                    r["cohort"] = doc.cohort
                    r["text_length"] = len(doc.text)
                    # Stream to disk
                    model_dir = self.output_dir / model_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    with open(model_dir / f"{doc.patient_id}.json", "w") as f:
                        json.dump(r, f, indent=2, default=str)
                    results.append(r)
                return results

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {}
                if m0:
                    futures[pool.submit(_run, m0, 0)] = m0
                if m1:
                    futures[pool.submit(_run, m1, 1)] = m1

                for future in futures:
                    model_name = futures[future]
                    try:
                        results = future.result()
                        all_results[model_name] = results
                        total = sum(r["num_triples"] for r in results)
                        logger.info("RAG %s: %d total triples", model_name, total)
                    except Exception as e:
                        logger.error("RAG FAILED %s: %s", model_name, e)

            unload_all()

        return all_results
