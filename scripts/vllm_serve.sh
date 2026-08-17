#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for ONE model, on one GPU.
#   Usage: scripts/vllm_serve.sh <hf_id> [port] [gpu] [max_model_len]
# The pipeline (in env llmft) then sets VLLM_URL=http://localhost:<port> and
# generate_batch() POSTs prompts to it — same weights, ~40x throughput.
set -euo pipefail
HF_ID="${1:?need HF model id, e.g. google/gemma-4-E4B-it}"
PORT="${2:-8000}"; GPU="${3:-0}"; MAXLEN="${4:-16384}"
cd /home/ud3d4/Desktop/TrustKG
source scripts/vllm_env.sh
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[vllm_serve] $HF_ID  port=$PORT  gpu=$GPU  max_len=$MAXLEN"
exec vllm serve "$HF_ID" \
    --port "$PORT" \
    --served-model-name "$HF_ID" \
    --gpu-memory-utilization 0.88 \
    --max-model-len "$MAXLEN"
