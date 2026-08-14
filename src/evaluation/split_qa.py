"""Split-aware QA evaluation: each dataset uses its OWN train-split KG.

Design:
  PubMedQA (1000 labeled) → 800 train / 200 test
    Train KG: context sentences + long_answer conclusions from train split
    Test: retrieve from train KG to ground answers

  MedQA-USMLE (10178 train / 1273 test)
    Train KG: question+correct_answer knowledge from train split
    Test: retrieve from train KG to ground answers

  CORAL (24 train / 8 test patients)
    Train KG: extracted triples from train patients
    Test: retrieve from train KG to ground extraction

Key optimizations:
  - Pre-encode all train KG texts with MedCPT Article Encoder (once)
  - Per question: encode query with MedCPT Query Encoder, dot product → top-k
  - Models loaded once, reused across all questions
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

# ── Robust answer parsers ──────────────────────────────────────


def _parse_mcq_answer(response: str) -> str:
    """Robustly parse MCQ answer letter (A/B/C/D) from model response.

    Prioritizes explicit 'Answer: X' from CoT, then falls back to heuristics.
    """
    resp = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if "</think>" in resp:
        resp = resp.split("</think>")[-1].strip()
    resp = re.sub(r"\*+", "", resp)
    upper = resp.upper().strip()

    # 1) Explicit "Answer: X" (CoT format — highest priority, last occurrence)
    matches = list(re.finditer(r"ANSWER\s*:\s*([A-D])\b", upper))
    if matches:
        return matches[-1].group(1)

    # 2) Response IS just the letter
    if upper in ("A", "B", "C", "D"):
        return upper

    # 3) Starts with letter
    m = re.match(r"^([A-D])\s*[.):,\s]", upper)
    if m:
        return m.group(1)

    # 4) "correct answer is X" / "option X"
    m = re.search(r"(?:CORRECT|OPTION)\s*(?:IS|:)\s*[:\s]*\(?([A-D])\b", upper)
    if m:
        return m.group(1)

    # 5) "(X)" anywhere
    m = re.search(r"\(([A-D])\)", upper)
    if m:
        return m.group(1)

    # 6) Last standalone letter (CoT reasons then concludes)
    matches = list(re.finditer(r"\b([A-D])\b", upper))
    if matches:
        return matches[-1].group(1)

    return "A"


def _parse_ynm_answer(response: str) -> str:
    """Robustly parse yes/no/maybe answer from model response.

    Prioritizes explicit 'Answer: X' from CoT, then falls back to heuristics.
    """
    resp = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if "</think>" in resp:
        resp = resp.split("</think>")[-1].strip()
    lower = resp.lower().strip()

    # 1) Explicit "Answer: X" (from CoT format — highest priority)
    m = re.search(r"answer\s*:\s*(yes|no|maybe)", lower)
    if m:
        return m.group(1)

    # 2) Starts with the answer word
    m = re.match(r"^(yes|no|maybe)\b", lower)
    if m:
        return m.group(1)

    # 3) "answer is X" / "decision: X"
    m = re.search(r"(?:decision|verdict|conclusion)\s*(?:is|:)\s*(yes|no|maybe)", lower)
    if m:
        return m.group(1)

    # 4) Last occurrence of yes/no/maybe (CoT often reasons then concludes)
    last_match = None
    for ans in ["yes", "no", "maybe"]:
        for m in re.finditer(r"\b" + ans + r"\b", lower):
            if last_match is None or m.start() > last_match[1]:
                last_match = (ans, m.start())
    if last_match:
        return last_match[0]

    return "maybe"


# ── KG construction from train splits ──────────────────────────


_REASONING_KG_PROMPT = """Extract a medical knowledge subgraph from this clinical case.

Clinical narrative:
{narrative}

Correct answer: {correct}
Incorrect options: {incorrect}

Output a JSON array of triples with explicit polarity:
{{"head": "...", "relation": "...", "tail": "...", "polarity": "+/-"}}

Rules:
- head and tail = single medical concepts (nodes)
- relation = semantically rich predicate
- polarity "+" = supports/confirms the correct answer
- polarity "-" = rules out/excludes an incorrect option

Example:
[
  {{"head": "low ferritin", "relation": "indicates", "tail": "iron deficiency anemia", "polarity": "+"}},
  {{"head": "microcytic MCV", "relation": "confirms", "tail": "iron deficiency anemia", "polarity": "+"}},
  {{"head": "microcytic MCV", "relation": "rules_out", "tail": "B12 deficiency", "polarity": "-"}},
  {{"head": "B12 deficiency", "relation": "requires", "tail": "macrocytic MCV", "polarity": "-"}},
  {{"head": "thalassemia", "relation": "excluded_by_absence_of", "tail": "target cells", "polarity": "-"}},
  {{"head": "iron deficiency anemia", "relation": "treated_by", "tail": "oral iron", "polarity": "+"}}
]

Extract clinical findings, connect them to the correct answer (+), and show why each incorrect option is excluded (-)."""

_REASONING_KG_SYSTEM = "You are a medical knowledge graph builder. Output ONLY a JSON array of (head, relation, tail, polarity) triples. Every head and tail must be a single medical concept."

_PUBMEDQA_KG_PROMPT = """Extract a biomedical knowledge subgraph from this research evidence.

Context:
{context}

Research question: {question}
Conclusion: {answer}

Output JSON array with polarity:
{{"head": "...", "relation": "...", "tail": "...", "polarity": "+/-"}}

- polarity "+" = finding supports the conclusion
- polarity "-" = finding contradicts or is irrelevant

Use relations: increases, decreases, causes, prevents, correlates_with, no_effect_on, inhibits, activates, biomarker_for, risk_factor_for, treats, mechanism_of

Example:
[
  {{"head": "aspirin", "relation": "inhibits", "tail": "COX-2", "polarity": "+"}},
  {{"head": "COX-2 inhibition", "relation": "reduces", "tail": "inflammation", "polarity": "+"}},
  {{"head": "sample size", "relation": "limits", "tail": "generalizability", "polarity": "-"}}
]

Output ONLY valid JSON array."""


def build_pubmedqa_train_kg(
    train_examples: list[dict],
    model_name: str = "gemma4-4b",
    gpu_id: int = 0,
    max_extract: int = 200,
) -> list[dict]:
    """Build conceptual KG from PubMedQA — (head, relation, tail) triples.

    Each training example → connected subgraph of biomedical relationships.
    Same LLM as CORAL extraction. KG quality → retrieval quality → QA.
    """
    from src.extraction.local_llm import extract_json

    kg = []
    n_extracted = 0

    for ex in train_examples[:max_extract]:
        context = ex.get("context", "")
        question = ex.get("question", "")
        answer = ex.get("final_decision", "")
        if len(context) < 50:
            continue

        prompt = _PUBMEDQA_KG_PROMPT.format(
            context=context[:2500],
            question=question,
            answer=answer,
        )
        triples = extract_json(model_name, _REASONING_KG_SYSTEM, prompt, gpu_id)

        for t in triples:
            if "head" in t and "relation" in t and "tail" in t:
                t.setdefault("polarity", "+")
                kg.append(t)
            elif "entity" in t and "attribute" in t and "value" in t:
                kg.append({"head": t["entity"], "relation": t["attribute"],
                           "tail": t["value"], "polarity": "+"})

        n_extracted += 1
        if n_extracted % 50 == 0:
            logger.info("PubMedQA KG: %d/%d, %d triples", n_extracted, max_extract, len(kg))

    logger.info("PubMedQA train KG: %d triples from %d subgraphs", len(kg), n_extracted)
    return kg


def build_medqa_train_kg(
    train_examples: list[dict],
    model_name: str = "gemma4-4b",
    gpu_id: int = 0,
    max_extract: int = 500,
) -> list[dict]:
    """Build conceptual KG from MedQA — (head, relation, tail) reasoning triples.

    Each question → connected subgraph:
      - (finding, indicates, correct_diagnosis)
      - (finding, rules_out, incorrect_option)
      - (condition, treated_by, treatment)

    Same LLM as CORAL. Proper KG triples, not EAV.
    """
    from src.extraction.local_llm import extract_json

    kg = []
    n_extracted = 0

    for ex in train_examples[:max_extract]:
        question = ex.get("question", "")
        answer_idx = ex.get("answer_idx", "")
        options = ex.get("options", {})
        if len(question) < 50 or not isinstance(options, dict):
            continue

        correct_text = options.get(answer_idx, "")
        incorrect_texts = [v for k, v in options.items() if k != answer_idx]

        prompt = _REASONING_KG_PROMPT.format(
            narrative=question[:2500],
            correct=correct_text,
            incorrect=", ".join(incorrect_texts),
        )
        triples = extract_json(model_name, _REASONING_KG_SYSTEM, prompt, gpu_id)

        for t in triples:
            if "head" in t and "relation" in t and "tail" in t:
                t.setdefault("polarity", "+")
                kg.append(t)
            elif "entity" in t and "attribute" in t and "value" in t:
                kg.append({"head": t["entity"], "relation": t["attribute"],
                           "tail": t["value"], "polarity": "+"})

        n_extracted += 1
        if n_extracted % 50 == 0:
            logger.info("MedQA KG: %d/%d, %d triples", n_extracted, max_extract, len(kg))

    logger.info("MedQA train KG: %d triples from %d subgraphs", len(kg), n_extracted)
    return kg


# ── Pre-encoded retrieval ──────────────────────────────────────


class PreEncodedRetriever:
    """MedCPT retriever with pre-encoded KG embeddings for efficiency."""

    def __init__(self, kg_triples: list[dict], device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.kg_triples = kg_triples
        self.triple_texts = []
        for t in kg_triples:
            # Support both (head, relation, tail) and legacy (entity, attribute, value)
            if "head" in t:
                text = f"{t['head']} {t['relation']} {t['tail']}"
            else:
                text = f"{t.get('entity', '')} {t.get('attribute', '')} {t.get('value', '')}"
            self.triple_texts.append(text.strip()[:512])

        cache = "/tmp/ud3d4_hf_cache"
        logger.info("Loading MedCPT encoders...")
        self.query_model = SentenceTransformer(
            "ncbi/MedCPT-Query-Encoder", cache_folder=cache, device=device,
        )
        article_model = SentenceTransformer(
            "ncbi/MedCPT-Article-Encoder", cache_folder=cache, device=device,
        )

        # Pre-encode all KG texts (once)
        logger.info("Pre-encoding %d KG triples...", len(self.triple_texts))
        self.kg_embeddings = article_model.encode(
            self.triple_texts, batch_size=512, show_progress_bar=True,
            normalize_embeddings=True,
        )
        logger.info("KG embeddings shape: %s", self.kg_embeddings.shape)

        # Free article encoder after pre-encoding
        del article_model

    def retrieve(self, question: str, top_k: int = 10) -> str:
        """Retrieve top-k KG triples with polarity markers."""
        q_emb = self.query_model.encode(
            [question], show_progress_bar=False, normalize_embeddings=True,
        )
        scores = np.dot(q_emb, self.kg_embeddings.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        lines = ["Medical knowledge graph:"]
        for idx in top_indices:
            t = self.kg_triples[idx]
            if "head" in t:
                pol = t.get("polarity", "+")
                lines.append(f"[{pol}] ({t['head']}, {t['relation']}, {t['tail']})")
            else:
                lines.append(f"[+] ({t.get('entity','')}, {t.get('attribute','')}, {t.get('value','')})")

        return "\n".join(lines)


# ── Main evaluation ────────────────────────────────────────────


def run_split_qa(
    model_name: str = "gemma4-4b",
    gpu_id: int = 0,
    max_test: int = 200,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run split-aware QA: train KG → test evaluation for each dataset."""
    import torch
    from datasets import load_dataset
    from src.extraction.local_llm import generate

    os.environ.setdefault("HF_TOKEN", "")
    cache = "/tmp/ud3d4_hf_cache"
    results = {}

    # ═══════════════════════════════════════════════════════════
    # PubMedQA: 800 train / 200 test
    # ═══════════════════════════════════════════════════════════
    logger.info("=== PubMedQA (800 train / 200 test) ===")
    pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", cache_dir=cache)

    # Split: first 800 train, last 200 test
    pqa_train = [pqa[i] for i in range(800)]
    pqa_test = [pqa[i] for i in range(800, min(1000, 800 + max_test))]

    # Build train KG using TRUST-KG extraction pipeline
    pqa_train_kg = build_pubmedqa_train_kg(
        [
            {
                "context": " ".join(row["context"]["contexts"]) if isinstance(row["context"], dict) else str(row["context"]),
                "long_answer": row.get("long_answer", ""),
                "question": row.get("question", ""),
            }
            for row in pqa_train
        ],
        model_name=model_name,
        gpu_id=gpu_id,
        max_extract=200,
    )

    # Pre-encode
    retriever = PreEncodedRetriever(pqa_train_kg, device="cpu")

    # Evaluate: vanilla vs KG-grounded (zero-shot — best for 4B model)
    for mode in ["vanilla", "kg_grounded"]:
        correct = 0
        total = len(pqa_test)
        for i, row in enumerate(pqa_test):
            context = " ".join(row["context"]["contexts"]) if isinstance(row["context"], dict) else str(row["context"])
            question = row["question"]
            answer = row["final_decision"].lower()

            if mode == "kg_grounded":
                kg_context = retriever.retrieve(question, top_k=8)
                prompt = (
                    f"Context: {context[:2000]}\n\n{kg_context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer with exactly one word: yes, no, or maybe."
                )
            else:
                prompt = (
                    f"Context: {context[:2000]}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer with exactly one word: yes, no, or maybe."
                )

            system = "You are a biomedical expert. Answer concisely with yes, no, or maybe."
            response = generate(model_name, system, prompt, gpu_id, max_new_tokens=32, temperature=0.01)
            pred = _parse_ynm_answer(response)

            if pred == answer:
                correct += 1

            if (i + 1) % 50 == 0:
                logger.info("PubMedQA [%s/%s] %d/%d (acc: %.1f%%)",
                            mode, model_name, i + 1, total, correct / (i + 1) * 100)

        acc = round(correct / max(total, 1), 4)
        results[f"pubmedqa_{mode}"] = {
            "benchmark": "PubMedQA", "mode": mode, "model": model_name,
            "accuracy": acc, "total": total, "correct": correct,
        }
        logger.info("PubMedQA [%s/%s]: %.1f%% (%d/%d)",
                     mode, model_name, acc * 100, correct, total)

    del retriever

    # ═══════════════════════════════════════════════════════════
    # MedQA: train split → test split
    # ═══════════════════════════════════════════════════════════
    logger.info("=== MedQA-USMLE (10178 train / %d test) ===", max_test)
    medqa_train = load_dataset("GBaker/MedQA-USMLE-4-options", split="train", cache_dir=cache)
    medqa_test = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", cache_dir=cache)

    # Build train KG using TRUST-KG extraction pipeline
    medqa_train_kg = build_medqa_train_kg(
        [
            {
                "question": row["question"],
                "answer": row["answer"],
                "answer_idx": row["answer_idx"],
                "options": row["options"],
            }
            for row in medqa_train
        ],
        model_name=model_name,
        gpu_id=gpu_id,
        max_extract=500,
    )

    # Pre-encode
    retriever = PreEncodedRetriever(medqa_train_kg, device="cpu")

    # Evaluate (zero-shot — best for 4B model)
    test_subset = [medqa_test[i] for i in range(min(max_test, len(medqa_test)))]
    for mode in ["vanilla", "kg_grounded"]:
        correct = 0
        total = len(test_subset)
        for i, row in enumerate(test_subset):
            options = row["options"]
            if isinstance(options, dict):
                opt_text = "\n".join(f"{k}. {v}" for k, v in options.items())
            else:
                opt_text = str(options)

            if mode == "kg_grounded":
                kg_context = retriever.retrieve(row["question"], top_k=8)
                prompt = (
                    f"{kg_context}\n\n"
                    f"Question: {row['question']}\n\n"
                    f"Options:\n{opt_text}\n\n"
                    f"Answer with ONLY the letter (A, B, C, or D)."
                )
            else:
                prompt = (
                    f"Question: {row['question']}\n\n"
                    f"Options:\n{opt_text}\n\n"
                    f"Answer with ONLY the letter (A, B, C, or D)."
                )

            system = "You are a medical expert taking the USMLE exam. Answer with just the letter."
            response = generate(model_name, system, prompt, gpu_id, max_new_tokens=32, temperature=0.01)
            pred = _parse_mcq_answer(response)

            if pred == row["answer_idx"]:
                correct += 1

            if (i + 1) % 50 == 0:
                logger.info("MedQA [%s/%s] %d/%d (acc: %.1f%%)",
                            mode, model_name, i + 1, total, correct / (i + 1) * 100)

        acc = round(correct / max(total, 1), 4)
        results[f"medqa_{mode}"] = {
            "benchmark": "MedQA-USMLE", "mode": mode, "model": model_name,
            "accuracy": acc, "total": total, "correct": correct,
        }
        logger.info("MedQA [%s/%s]: %.1f%% (%d/%d)",
                     mode, model_name, acc * 100, correct, total)

    del retriever

    # ═══════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"split_qa_{model_name}.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved to %s", output_dir / f"split_qa_{model_name}.json")

    return results
