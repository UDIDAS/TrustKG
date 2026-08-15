"""Multi-model multi-GPU LLM inference for EAV extraction.

Supports loading different models on different GPUs simultaneously.
Designed for 2x A6000 (48 GB each) — runs 2 models in parallel.

Memory constraints:
  - One model per GPU, bf16 precision
  - KV cache freed after each generation
  - Models unloaded explicitly between passes
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# HF cache on /tmp to avoid home quota
_TMP_CACHE = "/tmp/ud3d4_hf_cache"

# Registry of supported models with their HF IDs
MODEL_REGISTRY = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "gemma3-4b": "google/gemma-3-4b-it",   # paper's "Gemma 3 4B" (official; replaces broken google/gemma-4-E4B-it)
    "llama32-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-moe": "microsoft/Phi-mini-MoE-instruct",   # small MoE (BROKEN on transformers 5.8: remote code needs is_torch_fx_available)
    "olmoe-1b7b": "allenai/OLMoE-1B-7B-0924-Instruct",   # small MoE (7B total / 1B active), native transformers, no remote code / flash_attn
}

# Track loaded models per GPU: {gpu_id: (model_name, model, tokenizer)}
_loaded: dict[int, tuple[str, Any, Any]] = {}


def _resolve_cache_dir(model_id: str) -> str | None:
    """Return /tmp cache if model is stored there, else default HF cache."""
    tmp_path = os.path.join(_TMP_CACHE, "models--" + model_id.replace("/", "--"))
    if os.path.exists(tmp_path):
        return _TMP_CACHE
    return None


def load_model(model_name: str, gpu_id: int = 0) -> tuple[Any, Any]:
    """Load a model onto a specific GPU. Unloads any existing model on that GPU first.

    Args:
        model_name: key from MODEL_REGISTRY (e.g. "qwen3-8b")
        gpu_id: CUDA device index (0 or 1)

    Returns:
        (model, tokenizer) tuple
    """
    if gpu_id in _loaded and _loaded[gpu_id][0] == model_name:
        logger.info("Model %s already loaded on GPU %d", model_name, gpu_id)
        return _loaded[gpu_id][1], _loaded[gpu_id][2]

    # Unload existing model on this GPU
    unload_model(gpu_id)

    model_id = MODEL_REGISTRY[model_name]
    cache_dir = _resolve_cache_dir(model_id)
    device = f"cuda:{gpu_id}"

    logger.info("Loading %s (%s) onto %s ...", model_name, model_id, device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
    )

    # Use eager attention as fallback when flash_attn is unavailable
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN"),
        attn_implementation="eager",
    )
    model.eval()

    _loaded[gpu_id] = (model_name, model, tokenizer)
    vram = torch.cuda.memory_allocated(gpu_id) / 1e9
    logger.info("Loaded %s on %s (%.1f GB VRAM)", model_name, device, vram)
    return model, tokenizer


def unload_model(gpu_id: int = 0) -> None:
    """Free a model from a specific GPU."""
    if gpu_id in _loaded:
        name = _loaded[gpu_id][0]
        del _loaded[gpu_id]
        torch.cuda.empty_cache()
        logger.info("Unloaded %s from GPU %d", name, gpu_id)


def unload_all() -> None:
    """Free all models from all GPUs."""
    for gpu_id in list(_loaded.keys()):
        unload_model(gpu_id)


def generate(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    gpu_id: int = 0,
    max_new_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """Generate text from a specific model on a specific GPU.

    Thread-safe per GPU — each GPU runs one model sequentially.
    Two GPUs can run in parallel via ThreadPoolExecutor.
    """
    model, tokenizer = load_model(model_name, gpu_id)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Qwen3 models default to "thinking" mode — disable it for structured
    # extraction by adding enable_thinking=False to chat template
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Models that don't support enable_thinking kwarg
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}],
                tokenize=False, add_generation_prompt=True,
            )

    device = f"cuda:{gpu_id}"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Free KV cache immediately
    del inputs, outputs
    torch.cuda.empty_cache()

    return result


def generate_batch(
    model_name: str,
    prompts: list[tuple[str, str]],
    gpu_id: int = 0,
    max_new_tokens: int | None = None,
    temperature: float = 0.1,
    batch_size: int = 6,
) -> list[str]:
    """Throughput-oriented batched generation.

    prompts: list of (system_prompt, user_prompt). The model is loaded ONCE and
    kept resident; prompts are processed in padded batches (left-padded for decoder
    generation) so the GPU stays saturated instead of one-prompt-at-a-time.
    Returns raw output strings aligned to `prompts`.
    """
    if max_new_tokens is None:
        from src.config import EXTRACT_MAX_NEW_TOKENS
        max_new_tokens = EXTRACT_MAX_NEW_TOKENS
    model, tokenizer = load_model(model_name, gpu_id)
    device = f"cuda:{gpu_id}"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"   # required so generated tokens align across the batch

    def _render(sp: str, up: str) -> str:
        msgs = [{"role": "system", "content": sp}, {"role": "user", "content": up}]
        for kw in ({"enable_thinking": False}, {}):
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            except TypeError:
                continue
            except Exception:
                break
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": sp + "\n\n" + up}], tokenize=False, add_generation_prompt=True)

    outputs: list[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            texts = [_render(sp, up) for sp, up in prompts[i:i + batch_size]]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            in_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0, temperature=max(temperature, 0.01),
                    top_p=0.95 if temperature > 0 else 1.0, pad_token_id=tokenizer.eos_token_id,
                )
            for row in out[:, in_len:]:
                outputs.append(tokenizer.decode(row, skip_special_tokens=True))
            del inputs, out
            torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = prev_side
    return outputs


def generate_with_entropy(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    gpu_id: int = 0,
    max_new_tokens: int = 4096,
    temperature: float = 0.1,
) -> tuple[str, float]:
    """Generate text and compute token-level entropy (Draft Section 3.2).

    H(v) = -Σ p_t * log(p_t)

    Returns (generated_text, mean_entropy). High entropy = uncertain output.
    """
    model, tokenizer = load_model(model_name, gpu_id)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}],
                tokenize=False, add_generation_prompt=True,
            )

    device = f"cuda:{gpu_id}"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0][input_len:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute token-level entropy: H = -Σ p * log(p)
    entropies = []
    if hasattr(outputs, "scores") and outputs.scores:
        for score in outputs.scores:
            probs = torch.softmax(score[0], dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum().item()
            entropies.append(entropy)

    mean_entropy = sum(entropies) / max(len(entropies), 1) if entropies else 0.0

    del inputs, outputs
    torch.cuda.empty_cache()

    return result, mean_entropy


def extract_json(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    gpu_id: int = 0,
) -> list[dict[str, Any]]:
    """Generate and parse JSON array from a specific model."""
    from src.config import EXTRACT_MAX_NEW_TOKENS
    raw = generate(model_name, system_prompt, user_prompt, gpu_id,
                   max_new_tokens=EXTRACT_MAX_NEW_TOKENS)
    return _parse_json_response(raw)


def _parse_json_response(raw: str) -> list[dict[str, Any]]:
    """Robustly parse JSON array from LLM output, including truncated responses."""
    raw = raw.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    # Strip thinking tags (Qwen3 sometimes leaks these)
    if "<think>" in raw:
        think_end = raw.find("</think>")
        if think_end != -1:
            raw = raw[think_end + len("</think>"):].strip()
        else:
            # Thinking never closed — take everything after <think>...</s>
            raw = raw[raw.find("<think>") + len("<think>"):].strip()

    # Attempt 1: direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for key in ("triples", "data", "results"):
                if key in result and isinstance(result[key], list):
                    return result[key]
            return [result]
        return []
    except json.JSONDecodeError:
        pass

    # Attempt 2: find JSON array in response
    start = raw.find("[")
    if start == -1:
        logger.warning("No JSON array found: %s...", raw[:200])
        return []

    end = raw.rfind("]")
    if end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Attempt 3: recover truncated JSON array
    # Find the last complete JSON object (ends with "}")
    fragment = raw[start:]
    last_complete = fragment.rfind("}")
    if last_complete > 0:
        truncated = fragment[: last_complete + 1] + "]"
        try:
            result = json.loads(truncated)
            if isinstance(result, list):
                logger.info("Recovered %d triples from truncated JSON", len(result))
                return result
        except json.JSONDecodeError:
            pass

    # Attempt 4: parse individual objects with regex
    import re
    objects = []
    for match in re.finditer(r'\{[^{}]*\}', raw):
        try:
            obj = json.loads(match.group())
            if "entity" in obj or "value" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    if objects:
        logger.info("Recovered %d triples via regex fallback", len(objects))
        return objects

    logger.warning("JSON parse failed completely: %s...", raw[:300])
    return []
