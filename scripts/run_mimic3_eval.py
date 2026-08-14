"""MIMIC-III unified pipeline: KG construction + downstream prediction.

A. KG Construction (Tables 3, 10):
   - Extract from discharge notes using TRUST-KG pipeline
   - Evaluate against ALL structured GT (diagnoses + procedures + meds + labs)
   - Measure: P/R/F1, hallucination, ontology compliance

B. Downstream Prediction (Table 8):
   - Build train KG from train notes
   - For test patients: retrieve from train KG → predict diagnoses/procedures
   - Evaluate prediction accuracy against structured GT

Train/test split: 15 + 15 oncology patients.
Notes capped at 4K chars (1 chunk, ~2 min/note).
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HF_TOKEN", "")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

NOTE_MAX_CHARS = 4000  # 1 chunk per note


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(re.sub(r"[^\w\s]", " ", a.lower()).split())
    tokens_b = set(re.sub(r"[^\w\s]", " ", b.lower()).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def load_structured_gt(hadm_id, diagnoses, procedures, prescriptions, labs,
                        diag_desc, proc_desc, lab_desc):
    """Load all structured facts for one admission."""
    gt = []

    # Diagnoses
    for _, row in diagnoses[diagnoses["hadm_id"] == hadm_id].iterrows():
        desc = diag_desc.get(str(row["icd9_code"]), "")
        if desc:
            gt.append({"type": "diagnosis", "concept": desc.lower()})

    # Procedures
    for _, row in procedures[procedures["hadm_id"] == hadm_id].iterrows():
        desc = proc_desc.get(str(row["icd9_code"]), "")
        if desc:
            gt.append({"type": "procedure", "concept": desc.lower()})

    # Medications (top 20)
    for _, row in prescriptions[prescriptions["hadm_id"] == hadm_id].head(20).iterrows():
        drug = str(row.get("drug", "")).strip().lower()
        if drug and len(drug) > 2:
            gt.append({"type": "medication", "concept": drug})

    # Labs (top 20 unique)
    seen_labs = set()
    for _, row in labs[labs["hadm_id"] == hadm_id].head(50).iterrows():
        label = lab_desc.get(row.get("itemid"), "")
        if label and label not in seen_labs:
            seen_labs.add(label)
            gt.append({"type": "lab", "concept": label.lower()})
            if len(seen_labs) >= 20:
                break

    return gt


_STOPWORDS = {"unspecified", "other", "nos", "without", "mention", "not", "elsewhere", "classified"}


def _normalize(text: str) -> str:
    """Normalize medical concept for matching."""
    t = re.sub(r"[^\w\s]", " ", text.lower().strip())
    tokens = [w for w in t.split() if w not in _STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def _concept_match(a: str, b: str) -> bool:
    """Check if two medical concepts match using normalized string matching."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    # Exact normalized match
    if na == nb:
        return True
    # Substring (if shorter is >=4 chars)
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 4 and shorter in longer:
        return True
    # Token overlap (any shared medical token counts)
    if _token_overlap(na, nb) > 0.3:
        return True
    return False


def evaluate_extraction(triples, gt_facts, source_text):
    """Category-aware evaluation: match extracted triples against structured GT."""
    # Categorize extracted concepts by FHIR type
    fhir_to_gt_category = {
        "Condition": "diagnosis", "condition": "diagnosis",
        "MedicationStatement": "medication", "medicationstatement": "medication",
        "Observation": "lab", "observation": "lab",
        "Procedure": "procedure", "procedure": "procedure",
    }

    extracted_by_cat = {"diagnosis": set(), "procedure": set(), "medication": set(), "lab": set(), "other": set()}
    for t in triples:
        fhir = str(t.get("fhir_type", "")).strip()
        cat = fhir_to_gt_category.get(fhir, "other")
        for f in ["entity", "value"]:
            v = str(t.get(f, "")).lower().strip()
            if v and len(v) > 2:
                extracted_by_cat[cat].add(v)
                extracted_by_cat["other"].add(v)  # also add to catch-all

    gt_by_cat = {"diagnosis": set(), "procedure": set(), "medication": set(), "lab": set()}
    for fact in gt_facts:
        gt_by_cat.setdefault(fact["type"], set()).add(fact["concept"])

    # Match within categories + cross-category fallback
    total_gt = 0
    total_matched_gt = 0
    total_ext = set()
    total_matched_ext = 0

    for cat in ["diagnosis", "procedure", "medication", "lab"]:
        gt_set = gt_by_cat.get(cat, set())
        ext_set = extracted_by_cat.get(cat, set()) | extracted_by_cat.get("other", set())
        total_gt += len(gt_set)

        for g in gt_set:
            if any(_concept_match(g, e) for e in ext_set):
                total_matched_gt += 1

        for e in extracted_by_cat.get(cat, set()):
            total_ext.add(e)

    # Also count all extracted for precision
    all_extracted = set()
    for cat_set in extracted_by_cat.values():
        all_extracted.update(cat_set)
    all_extracted.discard("other")

    # Precision: extracted concepts matching any GT
    all_gt = set()
    for cat_set in gt_by_cat.values():
        all_gt.update(cat_set)

    matched_ext = sum(1 for e in all_extracted if any(_concept_match(e, g) for g in all_gt))
    precision = matched_ext / max(len(all_extracted), 1)
    recall = total_matched_gt / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Hallucination: extracted not in source
    source_lower = source_text.lower()
    halluc = sum(1 for e in all_extracted
                 if e not in source_lower and not any(w in source_lower for w in e.split()[:2] if len(w) > 3))
    halluc_rate = halluc / max(len(all_extracted), 1)

    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "hallucination": round(halluc_rate, 3),
            "n_extracted": len(all_extracted), "n_gt": total_gt, "n_matched": total_matched_gt}


def evaluate_downstream_prediction(test_notes, train_kg, test_gt_map):
    """Downstream: use train KG to predict test patient diagnoses/procedures."""
    from sentence_transformers import SentenceTransformer

    cache = "/tmp/ud3d4_hf_cache"
    query_enc = SentenceTransformer("ncbi/MedCPT-Query-Encoder", cache_folder=cache)
    article_enc = SentenceTransformer("ncbi/MedCPT-Article-Encoder", cache_folder=cache)

    # Encode train KG
    kg_texts = []
    for t in train_kg:
        text = f"{t.get('entity', '')} {t.get('attribute', '')} {t.get('value', '')}"
        kg_texts.append(text.strip()[:200])

    kg_embs = article_enc.encode(kg_texts, batch_size=256, normalize_embeddings=True,
                                  show_progress_bar=False)
    del article_enc

    # For each test patient, retrieve from train KG and check if predictions match GT
    results = []
    for note_text, hadm_id in test_notes:
        gt_facts = test_gt_map.get(hadm_id, [])
        if not gt_facts:
            continue

        gt_concepts = set(f["concept"] for f in gt_facts)

        # Encode test note → retrieve from KG
        q_emb = query_enc.encode([note_text[:2000]], normalize_embeddings=True, show_progress_bar=False)
        scores = np.dot(q_emb, kg_embs.T).flatten()
        top_idx = np.argsort(scores)[::-1][:20]

        # Retrieved KG concepts as predictions
        predicted = set()
        for idx in top_idx:
            t = train_kg[idx]
            for f in ["entity", "value"]:
                v = str(t.get(f, "")).lower().strip()
                if v and len(v) > 2:
                    predicted.add(v)

        # Semantic matching for prediction evaluation
        matcher = _get_matcher()
        pred_list = list(predicted)
        gt_list_ds = list(gt_concepts)
        if pred_list and gt_list_ds:
            pred_embs = matcher.encode(pred_list)
            gt_embs_ds = matcher.encode(gt_list_ds)
            hits = sum(1 for j, g in enumerate(gt_list_ds)
                       if any(matcher.match(g, p, gt_embs_ds[j], pred_embs[i])
                              for i, p in enumerate(pred_list)))
            recall = hits / len(gt_list_ds)
            prec_hits = sum(1 for i, p in enumerate(pred_list)
                            if any(matcher.match(p, g, pred_embs[i], gt_embs_ds[j])
                                   for j, g in enumerate(gt_list_ds)))
            precision = prec_hits / len(pred_list)
        else:
            recall = 0
            precision = 0
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        results.append({"precision": precision, "recall": recall, "f1": f1})

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default="gemma4-4b")
    parser.add_argument("--n-train", type=int, default=15)
    parser.add_argument("--n-test", type=int, default=15)
    args = parser.parse_args()

    from src.data.mimic3_reader import read_table
    from src.data.reader import ClinicalDocument
    from src.extraction.rag_extractor import RAGExtractor
    from src.extraction.fhir_normalizer import normalize_patient_triples

    # Load tables
    logger.info("Loading MIMIC-III tables...")
    notes = read_table("NOTEEVENTS")
    diagnoses = read_table("DIAGNOSES_ICD")
    procedures = read_table("PROCEDURES_ICD")
    prescriptions = read_table("PRESCRIPTIONS")
    labs = read_table("LABEVENTS")

    d_diag = read_table("D_ICD_DIAGNOSES")
    diag_desc = dict(zip(d_diag["icd9_code"].astype(str), d_diag["long_title"].str.lower()))
    d_proc = read_table("D_ICD_PROCEDURES")
    proc_desc = dict(zip(d_proc["icd9_code"].astype(str), d_proc["long_title"].str.lower()))
    d_lab = read_table("D_LABITEMS")
    lab_desc = dict(zip(d_lab["itemid"], d_lab["label"].str.lower()))

    # Select oncology patients with discharge summaries
    onc_pats = set(diagnoses[diagnoses["icd9_code"].astype(str).str.match(r"^(1[4-9]|2[0-3])")]["subject_id"].unique())
    discharge = notes[(notes["category"] == "Discharge summary") & (notes["subject_id"].isin(onc_pats))]
    discharge = discharge.groupby("subject_id").first().reset_index().sort_values("subject_id")

    n_total = args.n_train + args.n_test
    selected = discharge.head(n_total)
    train_df = selected.head(args.n_train)
    test_df = selected.tail(args.n_test)
    logger.info("Selected %d train + %d test patients", len(train_df), len(test_df))

    # Convert to docs (capped at NOTE_MAX_CHARS)
    def to_docs(df):
        docs = []
        for _, row in df.iterrows():
            text = str(row["text"]).strip()[:NOTE_MAX_CHARS]
            if len(text) < 200:
                continue
            docs.append(ClinicalDocument(
                patient_id=f"mimic3_{int(row['subject_id'])}",
                cohort="mimic3", source="discharge_summary", text=text,
                metadata={"hadm_id": int(row["hadm_id"]), "subject_id": int(row["subject_id"])},
            ))
        return docs

    train_docs = to_docs(train_df)
    test_docs = to_docs(test_df)

    # ═══ A. KG Construction ═══
    out_dir = Path("results/extraction/mimic3")
    extractor = RAGExtractor(output_dir=out_dir)

    logger.info("=== A. EXTRACTING TRAIN (%d notes, capped %d chars) ===", len(train_docs), NOTE_MAX_CHARS)
    train_results = extractor.extract_batch(train_docs, model_name=args.model, gpu_id=args.gpu)

    train_kg = []
    for r in train_results:
        normalize_patient_triples(r["triples"])
        train_kg.extend(r["triples"])
    logger.info("Train KG: %d triples from %d notes", len(train_kg), len(train_results))

    logger.info("=== A. EXTRACTING TEST (%d notes, with %d seed triples) ===", len(test_docs), len(train_kg))
    test_results = extractor.extract_batch(test_docs, model_name=args.model, gpu_id=args.gpu, seed_triples=train_kg)

    # Evaluate extraction quality
    logger.info("=== A. EVALUATING EXTRACTION QUALITY ===")
    all_metrics = {"train": [], "test": []}

    for split, results_list, docs in [("train", train_results, train_docs), ("test", test_results, test_docs)]:
        for r, doc in zip(results_list, docs):
            hadm_id = doc.metadata["hadm_id"]
            gt = load_structured_gt(hadm_id, diagnoses, procedures, prescriptions, labs,
                                     diag_desc, proc_desc, lab_desc)
            normalize_patient_triples(r["triples"])
            m = evaluate_extraction(r["triples"], gt, doc.text)
            all_metrics[split].append(m)

    for split in ["train", "test"]:
        if all_metrics[split]:
            p = [m["precision"] for m in all_metrics[split]]
            r = [m["recall"] for m in all_metrics[split]]
            f1 = [m["f1"] for m in all_metrics[split]]
            h = [m["hallucination"] for m in all_metrics[split]]
            logger.info("%s (n=%d): P=%.3f±%.3f R=%.3f±%.3f F1=%.3f±%.3f Halluc=%.3f",
                        split.upper(), len(p), np.mean(p), np.std(p),
                        np.mean(r), np.std(r), np.mean(f1), np.std(f1), np.mean(h))

    # ═══ B. Downstream Prediction ═══
    logger.info("=== B. DOWNSTREAM PREDICTION (train KG → predict test diagnoses) ===")
    test_notes_for_pred = [(doc.text, doc.metadata["hadm_id"]) for doc in test_docs]
    test_gt_map = {}
    for doc in test_docs:
        hadm_id = doc.metadata["hadm_id"]
        test_gt_map[hadm_id] = load_structured_gt(hadm_id, diagnoses, procedures, prescriptions, labs,
                                                    diag_desc, proc_desc, lab_desc)

    pred_results = evaluate_downstream_prediction(test_notes_for_pred, train_kg, test_gt_map)

    if pred_results:
        p = [m["precision"] for m in pred_results]
        r = [m["recall"] for m in pred_results]
        f1 = [m["f1"] for m in pred_results]
        logger.info("DOWNSTREAM PREDICTION (n=%d): P=%.3f±%.3f R=%.3f±%.3f F1=%.3f±%.3f",
                    len(pred_results), np.mean(p), np.std(p),
                    np.mean(r), np.std(r), np.mean(f1), np.std(f1))

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "extraction": {"train": all_metrics["train"], "test": all_metrics["test"]},
        "downstream": pred_results,
        "train_kg_size": len(train_kg),
    }
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Train KG: {len(train_kg)} triples")
    for split in ["train", "test"]:
        if all_metrics[split]:
            f1 = [m["f1"] for m in all_metrics[split]]
            print(f"Extraction {split.upper()}: F1={np.mean(f1):.3f}")
    if pred_results:
        f1 = [m["f1"] for m in pred_results]
        print(f"Downstream prediction: F1={np.mean(f1):.3f}")


if __name__ == "__main__":
    main()
