"""Graph-grounded QA: score answer options against KG with positive/negative edges.

For each test question:
  1. For EACH answer option, retrieve KG triples about that option
  2. Count positive support (+) and negative evidence (-)
  3. Score = positive_support - negative_penalty
  4. Pick highest-scoring option

No LLM at inference — KG quality directly determines accuracy.

Works for:
  - MedQA: options are medical concepts (drugs, diagnoses, procedures)
  - PubMedQA: options are yes/no/maybe, scored against evidence polarity
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OptionKGScorer:
    """Score each answer option by retrieving KG triples ABOUT that option."""

    def __init__(self, kg_triples: list[dict], device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        cache = "/tmp/ud3d4_hf_cache"
        self.kg = kg_triples
        self.encoder = SentenceTransformer(
            "ncbi/MedCPT-Article-Encoder", cache_folder=cache, device=device,
        )

        # Separate by polarity
        self.pos_triples = [t for t in kg_triples if t.get("polarity", "+") == "+"]
        self.neg_triples = [t for t in kg_triples if t.get("polarity") == "-"]

        # Encode triple texts
        def _text(t):
            if "head" in t:
                return f"{t['head']} {t['relation']} {t['tail']}"
            return f"{t.get('entity', '')} {t.get('attribute', '')} {t.get('value', '')}"

        logger.info("Encoding KG: %d positive, %d negative triples...",
                     len(self.pos_triples), len(self.neg_triples))

        self.pos_texts = [_text(t) for t in self.pos_triples]
        self.neg_texts = [_text(t) for t in self.neg_triples]

        self.pos_emb = self.encoder.encode(
            self.pos_texts, batch_size=512, normalize_embeddings=True, show_progress_bar=True,
        ) if self.pos_texts else np.zeros((0, 768))

        self.neg_emb = self.encoder.encode(
            self.neg_texts, batch_size=512, normalize_embeddings=True, show_progress_bar=True,
        ) if self.neg_texts else np.zeros((0, 768))

        logger.info("KG encoded: %d pos, %d neg", len(self.pos_emb), len(self.neg_emb))

    def score_option(self, option_text: str, question_context: str = "", top_k: int = 10) -> dict:
        """Score one option by retrieving KG triples about it."""
        # Encode the option (what we're looking up in the KG)
        query = f"{option_text} {question_context[:200]}" if question_context else option_text
        q_emb = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)

        # Find positive triples supporting this option
        pos_score = 0.0
        pos_evidence = []
        if len(self.pos_emb) > 0:
            sims = np.dot(q_emb, self.pos_emb.T).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            pos_score = float(np.mean(sims[top_idx]))
            pos_evidence = [self.pos_texts[i] for i in top_idx[:3]]

        # Find negative triples against this option
        neg_score = 0.0
        neg_evidence = []
        if len(self.neg_emb) > 0:
            sims = np.dot(q_emb, self.neg_emb.T).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            neg_score = float(np.mean(sims[top_idx]))
            neg_evidence = [self.neg_texts[i] for i in top_idx[:3]]

        return {
            "total": pos_score - neg_score,
            "pos_score": pos_score,
            "neg_score": neg_score,
            "pos_evidence": pos_evidence,
            "neg_evidence": neg_evidence,
        }


def build_medqa_kg(
    train_examples: list[dict],
    model_name: str = "gemma4-4b",
    gpu_id: int = 0,
    max_extract: int = 500,
) -> list[dict]:
    """Build MedQA KG with (head, relation, tail, polarity) using LLM."""
    from src.extraction.local_llm import extract_json

    PROMPT = """Extract medical knowledge triples from this clinical case.

Clinical case:
{narrative}

Correct answer: {correct}
Incorrect options: {incorrect}

Output JSON array: {{"head": "...", "relation": "...", "tail": "...", "polarity": "+/-"}}

Rules:
- head and tail = single medical concepts (not sentences)
- polarity "+" = supports the correct answer
- polarity "-" = rules out an incorrect option
- Use predicates: treats, causes, indicates, presents_with, diagnosed_by, ruled_out_by, contraindicated_in, associated_with, side_effect_of, prevents

Output ONLY valid JSON array."""

    SYSTEM = "You are a medical KG builder. Output ONLY a JSON array of triples."

    kg = []
    for i, ex in enumerate(train_examples[:max_extract]):
        options = ex.get("options", {})
        answer_idx = ex.get("answer_idx", "")
        question = ex.get("question", "")
        if not isinstance(options, dict) or not answer_idx or len(question) < 50:
            continue

        correct = options.get(answer_idx, "")
        incorrect = [v for k, v in options.items() if k != answer_idx]

        prompt = PROMPT.format(
            narrative=question[:2500],
            correct=correct,
            incorrect=", ".join(incorrect),
        )
        triples = extract_json(model_name, SYSTEM, prompt, gpu_id)

        for t in triples:
            if "head" in t and "relation" in t and "tail" in t:
                t.setdefault("polarity", "+")
                kg.append(t)

        if (i + 1) % 50 == 0:
            logger.info("MedQA KG: %d/%d, %d triples", i + 1, max_extract, len(kg))

    logger.info("MedQA KG: %d triples from %d examples", len(kg), min(len(train_examples), max_extract))
    return kg


def build_pubmedqa_kg(
    train_examples: list[dict],
    model_name: str = "gemma4-4b",
    gpu_id: int = 0,
    max_extract: int = 200,
) -> list[dict]:
    """Build PubMedQA KG with polarity."""
    from src.extraction.local_llm import extract_json

    PROMPT = """Extract biomedical knowledge triples from this research.

Context: {context}
Question: {question}
Conclusion: {answer}

Output JSON array: {{"head": "...", "relation": "...", "tail": "...", "polarity": "+/-"}}

- "+" = supports the conclusion
- "-" = contradicts or limits the conclusion
- Use: increases, decreases, causes, prevents, correlates_with, no_effect_on, inhibits, treats

Output ONLY valid JSON array."""

    SYSTEM = "You are a biomedical KG builder. Output ONLY JSON array."

    kg = []
    for i, ex in enumerate(train_examples[:max_extract]):
        context = ex.get("context", "")
        if len(context) < 50:
            continue
        prompt = PROMPT.format(
            context=context[:2500],
            question=ex.get("question", ""),
            answer=ex.get("final_decision", ""),
        )
        triples = extract_json(model_name, SYSTEM, prompt, gpu_id)
        for t in triples:
            if "head" in t and "relation" in t and "tail" in t:
                t.setdefault("polarity", "+")
                kg.append(t)
        if (i + 1) % 50 == 0:
            logger.info("PubMedQA KG: %d/%d, %d triples", i + 1, max_extract, len(kg))

    logger.info("PubMedQA KG: %d triples from %d examples", len(kg), min(len(train_examples), max_extract))
    return kg


def run_medqa_option_scoring(
    kg_triples: list[dict],
    max_test: int = 200,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run MedQA with option-based KG scoring."""
    from datasets import load_dataset

    os.environ.setdefault("HF_TOKEN", "")
    cache = "/tmp/ud3d4_hf_cache"

    scorer = OptionKGScorer(kg_triples, device="cpu")

    medqa_test = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", cache_dir=cache)
    test_subset = [medqa_test[i] for i in range(min(max_test, len(medqa_test)))]

    correct = 0
    total = len(test_subset)

    for i, row in enumerate(test_subset):
        options = row["options"]
        if not isinstance(options, dict):
            continue

        question = row["question"]
        # Extract key clinical context from question (last 200 chars often have the actual question)
        q_context = question[-200:]

        # Score each option against the KG
        option_scores = {}
        for key, text in options.items():
            score = scorer.score_option(text, q_context)
            option_scores[key] = score["total"]

        # Pick highest-scoring option
        pred = max(option_scores, key=option_scores.get)

        if pred == row["answer_idx"]:
            correct += 1

        if (i + 1) % 50 == 0:
            logger.info("MedQA [option_scoring] %d/%d: acc=%.1f%%",
                        i + 1, total, correct / (i + 1) * 100)

    acc = round(correct / max(total, 1), 4)
    logger.info("MedQA [option_scoring]: %.1f%% (%d/%d)", acc * 100, correct, total)

    result = {"accuracy": acc, "correct": correct, "total": total, "kg_triples": len(kg_triples)}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "medqa_option_scoring.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


def run_pubmedqa_option_scoring(
    kg_triples: list[dict],
    max_test: int = 200,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run PubMedQA with option-based KG scoring.

    For PubMedQA, the 'options' are yes/no/maybe.
    We score the QUESTION against positive vs negative KG triples.
    More positive support → yes, more negative → no, balanced → maybe.
    """
    from datasets import load_dataset

    os.environ.setdefault("HF_TOKEN", "")
    cache = "/tmp/ud3d4_hf_cache"

    scorer = OptionKGScorer(kg_triples, device="cpu")

    pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", cache_dir=cache)
    test_data = [pqa[i] for i in range(800, min(1000, 800 + max_test))]

    correct = 0
    total = len(test_data)

    for i, row in enumerate(test_data):
        ctx = " ".join(row["context"]["contexts"]) if isinstance(row["context"], dict) else str(row["context"])
        question = row["question"]
        answer = row["final_decision"].lower()

        # Score the question+context against KG
        query = f"{question} {ctx[:500]}"
        score = scorer.score_option(query)

        # Decision based on positive vs negative balance
        diff = score["pos_score"] - score["neg_score"]
        if diff > 0.02:
            pred = "yes"
        elif diff < -0.02:
            pred = "no"
        else:
            pred = "maybe"

        if pred == answer:
            correct += 1

        if (i + 1) % 50 == 0:
            logger.info("PubMedQA [option_scoring] %d/%d: acc=%.1f%%",
                        i + 1, total, correct / (i + 1) * 100)

    acc = round(correct / max(total, 1), 4)
    logger.info("PubMedQA [option_scoring]: %.1f%% (%d/%d)", acc * 100, correct, total)

    result = {"accuracy": acc, "correct": correct, "total": total}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "pubmedqa_option_scoring.json", "w") as f:
            json.dump(result, f, indent=2)

    return result
