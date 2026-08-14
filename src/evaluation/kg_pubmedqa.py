"""PubMedQA: Fine-tuned encoder with KG-augmented input.

Approach:
  1. Build KG from train split using TRUST-KG pipeline
  2. For each example, augment context with retrieved KG triples
  3. Fine-tune PubmedBERT on: [context + KG_triples] [SEP] [question] → yes/no/maybe
  4. Evaluate on test split

Demonstrates: KG construction quality directly improves classification accuracy.
Compare: with KG vs without KG (context-only) to show KG contribution.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class PubMedQADataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def _augment_with_kg(context: str, question: str, kg_triples: list[dict],
                      kg_embs: np.ndarray, query_enc, top_k: int = 5) -> str:
    """Augment context with relevant KG triples for this question."""
    if len(kg_embs) == 0:
        return f"{context} Question: {question}"

    q_emb = query_enc.encode([question], normalize_embeddings=True, show_progress_bar=False)
    scores = np.dot(q_emb, kg_embs.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]

    kg_text = " ".join(
        f"{t.get('head', t.get('entity', ''))} {t.get('relation', t.get('attribute', ''))} {t.get('tail', t.get('value', ''))}"
        for t in [kg_triples[i] for i in top_idx]
    )

    return f"{context} Medical knowledge: {kg_text} Question: {question}"


def run_pubmedqa_finetuned(
    kg_triples: list[dict],
    max_test: int = 200,
    epochs: int = 10,
    lr: float = 2e-5,
    batch_size: int = 16,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Fine-tune PubmedBERT on PubMedQA with KG-augmented input."""
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.environ.setdefault("HF_TOKEN", "")
    cache = "/tmp/ud3d4_hf_cache"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load data
    pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", cache_dir=cache)
    train_data = [pqa[i] for i in range(800)]
    test_data = [pqa[i] for i in range(800, min(1000, 800 + max_test))]

    label_map = {"yes": 0, "no": 1, "maybe": 2}

    # Pre-encode KG for retrieval
    logger.info("Encoding %d KG triples for retrieval...", len(kg_triples))
    query_enc = SentenceTransformer("ncbi/MedCPT-Query-Encoder", cache_folder=cache)
    article_enc = SentenceTransformer("ncbi/MedCPT-Article-Encoder", cache_folder=cache)

    kg_texts = []
    for t in kg_triples:
        if "head" in t:
            kg_texts.append(f"{t['head']} {t['relation']} {t['tail']}")
        else:
            kg_texts.append(f"{t.get('entity', '')} {t.get('attribute', '')} {t.get('value', '')}")

    kg_embs = article_enc.encode(
        kg_texts, batch_size=512, show_progress_bar=True, normalize_embeddings=True,
    ) if kg_texts else np.zeros((0, 768))

    del article_enc  # free memory

    # Build augmented texts
    def _make_texts(examples, use_kg=True):
        texts = []
        for row in examples:
            ctx = " ".join(row["context"]["contexts"]) if isinstance(row["context"], dict) else str(row["context"])
            q = row["question"]
            if use_kg and len(kg_embs) > 0:
                texts.append(_augment_with_kg(ctx[:800], q, kg_triples, kg_embs, query_enc, top_k=5))
            else:
                texts.append(f"{ctx[:900]} Question: {q}")
        return texts

    # Load encoder for classification (safetensors compatible)
    model_name = "microsoft/deberta-v3-base"
    logger.info("Loading %s...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)

    results = {}

    # Train and evaluate for both: with KG and without KG
    for mode in ["kg_augmented", "context_only"]:
        use_kg = mode == "kg_augmented"
        logger.info("=== Training: %s ===", mode)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3, cache_dir=cache, revision="refs/pr/14",
        ).to(device)

        # Build datasets
        train_texts = _make_texts(train_data, use_kg=use_kg)
        test_texts = _make_texts(test_data, use_kg=use_kg)
        train_labels = [label_map[row["final_decision"]] for row in train_data]
        test_labels = [label_map[row["final_decision"]] for row in test_data]

        train_ds = PubMedQADataset(train_texts, train_labels, tokenizer)
        test_ds = PubMedQADataset(test_texts, test_labels, tokenizer)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        # Train
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            if (epoch + 1) % 2 == 0:
                logger.info("  Epoch %d/%d, loss: %.4f", epoch + 1, epochs, total_loss / len(train_loader))

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                batch_device = {k: v.to(device) for k, v in batch.items()}
                outputs = model(input_ids=batch_device["input_ids"],
                               attention_mask=batch_device["attention_mask"])
                preds = outputs.logits.argmax(dim=-1).cpu()
                correct += (preds == batch["labels"]).sum().item()
                total += len(preds)

        acc = correct / total
        logger.info("%s accuracy: %.1f%% (%d/%d)", mode, acc * 100, correct, total)

        results[mode] = {
            "accuracy": round(acc, 4),
            "correct": correct,
            "total": total,
        }

        del model
        torch.cuda.empty_cache()

    # KG improvement
    delta = results["kg_augmented"]["accuracy"] - results["context_only"]["accuracy"]
    logger.info("KG improvement: %.1f%% (%.1f%% → %.1f%%)",
                delta * 100,
                results["context_only"]["accuracy"] * 100,
                results["kg_augmented"]["accuracy"] * 100)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "pubmedqa_finetuned.json", "w") as f:
            json.dump(results, f, indent=2)

    return results
