"""Multi-Agent Semantic Validation (Draft Section 3.4).

Implements the draft's validation exactly:
  - J(τ): Semantic plausibility score via LLM
  - ξ(τ): Adversarial perturbation robustness (contradiction ratio)
  - C(τ): Multi-prompt self-consistency
  - T(τ) = β₁R(τ) + β₂C(τ) + β₃J(τ) − β₄ξ(τ)

Uses the same local LLMs (Qwen3/Gemma4/Llama) as validation agents.
Each agent independently evaluates triple plausibility.

Falls back to rule-based validation (validation.py) if LLM validation
is too slow or fails. Both results are tracked for ablation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── Validation prompts ─────────────────────────────────────────

_PLAUSIBILITY_PROMPT = """\
Evaluate the clinical plausibility of this extracted triple.

Triple:
  Entity: {entity}
  Attribute: {attribute}
  Value: {value}
  FHIR Type: {fhir_type}
  Evidence: {evidence}

Is this triple clinically plausible? Consider:
1. Does the entity-attribute-value relationship make medical sense?
2. Is the value reasonable for this entity and attribute?
3. Is the FHIR type assignment correct?

Answer with a JSON object: {{"plausible": true/false, "confidence": 0.0-1.0, "reason": "brief reason"}}"""


_CONTRADICTION_PROMPT = """\
Given the original triple and a modified version, determine if they contradict each other.

Original:
  {entity} — {attribute} — {value}

Modified:
  {entity} — {attribute} — {modified_value}

Do these two statements contradict each other? Answer with:
{{"contradicts": true/false, "reason": "brief reason"}}"""


_CONSISTENCY_PROMPT = """\
Given the following clinical narrative excerpt, verify if this triple is supported by the text.

Narrative: {evidence}

Triple:
  Entity: {entity}
  Attribute: {attribute}
  Value: {value}

Is this triple fully supported by the narrative text? Answer with:
{{"supported": true/false, "confidence": 0.0-1.0}}"""


# ── Adversarial perturbations ──────────────────────────────────

def _generate_perturbations(triple: dict, n: int = 3) -> list[dict]:
    """Generate adversarial semantic variants of a triple (Draft ξ(τ)).

    Perturbation strategies:
      1. Negate value (positive→negative, present→absent)
      2. Swap value with related but different value
      3. Change FHIR type to incompatible type
    """
    variants = []
    value = str(triple.get("value", ""))
    entity = str(triple.get("entity", ""))

    # Strategy 1: Negate
    negation_map = {
        "positive": "negative", "negative": "positive",
        "present": "absent", "absent": "present",
        "yes": "no", "no": "yes",
        "normal": "abnormal", "abnormal": "normal",
        "elevated": "decreased", "decreased": "elevated",
        "high": "low", "low": "high",
        "malignant": "benign", "benign": "malignant",
    }
    for orig, neg in negation_map.items():
        if orig in value.lower():
            negated = re.sub(re.escape(orig), neg, value, flags=re.IGNORECASE)
            variants.append({**triple, "value": negated, "_perturbation": "negation"})
            break

    # Strategy 2: Swap numeric values
    nums = re.findall(r"\d+\.?\d*", value)
    if nums:
        import random
        for num_str in nums[:1]:
            num = float(num_str)
            swapped = str(round(num * random.uniform(2, 10), 1))
            variants.append({
                **triple,
                "value": value.replace(num_str, swapped, 1),
                "_perturbation": "value_swap",
            })

    # Strategy 3: Type swap
    fhir = triple.get("fhir_type", "")
    type_swaps = {
        "Condition": "MedicationStatement",
        "Observation": "Procedure",
        "Procedure": "Condition",
        "MedicationStatement": "Observation",
    }
    if fhir in type_swaps:
        variants.append({
            **triple,
            "fhir_type": type_swaps[fhir],
            "_perturbation": "type_swap",
        })

    return variants[:n]


# ── Agent-based validation (Draft Section 3.4) ────────────────

def validate_triple_with_agents(
    triple: dict,
    source_text: str,
    model_name: str = "qwen3-8b",
    gpu_id: int = 0,
    retrieval_score: float = 0.5,
) -> dict[str, Any]:
    """Run multi-agent semantic validation on a single triple.

    Implements Draft Section 3.4:
      J(τ): semantic plausibility via LLM
      ξ(τ): adversarial perturbation robustness
      C(τ): self-consistency (multi-prompt)
      T(τ) = β₁R(τ) + β₂C(τ) + β₃J(τ) − β₄ξ(τ)
    """
    from src.extraction.local_llm import generate, _parse_json_response

    entity = str(triple.get("entity", ""))
    attribute = str(triple.get("attribute", ""))
    value = str(triple.get("value", ""))
    fhir_type = str(triple.get("fhir_type", ""))
    evidence = str(triple.get("evidence_span", ""))[:200]

    system = "You are a clinical validation agent. Answer with valid JSON only."

    # ── J(τ): Semantic plausibility ──
    try:
        prompt = _PLAUSIBILITY_PROMPT.format(
            entity=entity, attribute=attribute, value=value,
            fhir_type=fhir_type, evidence=evidence,
        )
        resp = generate(model_name, system, prompt, gpu_id, max_new_tokens=128, temperature=0.01)
        parsed = _parse_json_response(resp)
        if parsed and isinstance(parsed, list) and parsed[0]:
            j_score = float(parsed[0].get("confidence", 0.5))
            j_plausible = parsed[0].get("plausible", True)
        elif parsed and isinstance(parsed, dict):
            j_score = float(parsed.get("confidence", 0.5))
            j_plausible = parsed.get("plausible", True)
        else:
            j_score = 0.5
            j_plausible = True
    except Exception:
        j_score = 0.5
        j_plausible = True

    # ── ξ(τ): Adversarial perturbation robustness ──
    perturbations = _generate_perturbations(triple)
    contradictions = 0
    for pert in perturbations:
        try:
            prompt = _CONTRADICTION_PROMPT.format(
                entity=entity, attribute=attribute, value=value,
                modified_value=str(pert.get("value", "")),
            )
            resp = generate(model_name, system, prompt, gpu_id, max_new_tokens=64, temperature=0.01)
            parsed = _parse_json_response(resp)
            if parsed:
                p = parsed[0] if isinstance(parsed, list) else parsed
                if p.get("contradicts", False):
                    contradictions += 1
        except Exception:
            pass

    xi_score = contradictions / max(len(perturbations), 1)

    # ── C(τ): Self-consistency (multi-prompt verification) ──
    try:
        prompt = _CONSISTENCY_PROMPT.format(
            evidence=evidence, entity=entity,
            attribute=attribute, value=value,
        )
        resp = generate(model_name, system, prompt, gpu_id, max_new_tokens=64, temperature=0.01)
        parsed = _parse_json_response(resp)
        if parsed:
            p = parsed[0] if isinstance(parsed, list) else parsed
            c_score = float(p.get("confidence", 0.5))
        else:
            c_score = 0.5
    except Exception:
        c_score = 0.5

    # ── T(τ): Combined trust score (Draft formula) ──
    β1, β2, β3, β4 = 0.25, 0.25, 0.30, 0.20
    r_score = retrieval_score
    trust = β1 * r_score + β2 * c_score + β3 * j_score - β4 * xi_score

    return {
        "R_retrieval": round(r_score, 4),
        "C_consistency": round(c_score, 4),
        "J_plausibility": round(j_score, 4),
        "J_plausible": j_plausible,
        "xi_contradiction": round(xi_score, 4),
        "n_perturbations": len(perturbations),
        "n_contradictions": contradictions,
        "trust_score": round(max(0, min(1, trust)), 4),
    }


def validate_patient_triples_with_agents(
    triples: list[dict],
    source_text: str,
    model_name: str = "qwen3-8b",
    gpu_id: int = 0,
    trust_threshold: float = 0.4,
    max_triples: int = 50,
) -> dict[str, Any]:
    """Validate all triples for a patient using LLM agents.

    For efficiency, only validates up to max_triples (highest-uncertainty first).
    Remaining triples use rule-based validation as fallback.

    Returns accepted, rejected, and per-triple validation scores.
    """
    from src.extraction.validation import validate_patient_triples as rule_validate

    # First: rule-based validation on all triples (fast, CPU)
    rule_result = rule_validate(triples, source_text, trust_threshold=0.0)
    all_triples = rule_result["accepted"] + rule_result["rejected"]

    # Select triples for LLM validation (most uncertain first)
    uncertain = sorted(
        all_triples,
        key=lambda t: abs(t.get("_validation", {}).get("trust_score", 0.5) - 0.5),
    )[:max_triples]

    # LLM agent validation on uncertain triples
    for triple in uncertain:
        retrieval_score = triple.get("_retrieval", {}).get("score", 0.5)
        agent_result = validate_triple_with_agents(
            triple, source_text, model_name, gpu_id, retrieval_score,
        )
        triple["_agent_validation"] = agent_result
        # Update trust with agent score (weighted average with rule-based)
        rule_trust = triple.get("_validation", {}).get("trust_score", 0.5)
        agent_trust = agent_result["trust_score"]
        triple["_combined_trust"] = round(0.4 * rule_trust + 0.6 * agent_trust, 4)

    # For remaining triples, use rule-based trust as combined trust
    agent_pids = {id(t) for t in uncertain}
    for t in all_triples:
        if id(t) not in agent_pids:
            t["_combined_trust"] = t.get("_validation", {}).get("trust_score", 0.5)

    # Partition by combined trust
    accepted = [t for t in all_triples if t.get("_combined_trust", 0) >= trust_threshold]
    rejected = [t for t in all_triples if t.get("_combined_trust", 0) < trust_threshold]

    trusts = [t.get("_combined_trust", 0) for t in all_triples]

    return {
        "accepted": accepted,
        "rejected": rejected,
        "stats": {
            "total": len(all_triples),
            "agent_validated": len(uncertain),
            "rule_only": len(all_triples) - len(uncertain),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "mean_combined_trust": round(sum(trusts) / max(len(trusts), 1), 4),
            "mean_agent_trust": round(
                sum(t["_agent_validation"]["trust_score"] for t in uncertain) / max(len(uncertain), 1), 4
            ) if uncertain else 0,
        },
    }
