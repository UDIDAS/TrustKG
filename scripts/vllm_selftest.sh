#!/usr/bin/env bash
# One-shot end-to-end validation of the vLLM integration:
#   start gemma-4 server -> wait /health -> extract N notes through the llmft
#   pipeline (VLLM_URL) -> tear down.  Self-contained so it needs no cross-call
#   background persistence.  Usage: bash scripts/vllm_selftest.sh [dataset] [limit]
set -uo pipefail
cd /home/ud3d4/Desktop/TrustKG
DS="${1:-coral}"; LIM="${2:-3}"; PORT=8000; LOG=results/vllm_server_g4.log

pkill -9 -f "vllm serve" 2>/dev/null || true
( source scripts/vllm_env.sh; export CUDA_VISIBLE_DEVICES=0
  exec vllm serve google/gemma-4-E4B-it --port $PORT \
       --served-model-name google/gemma-4-E4B-it \
       --gpu-memory-utilization 0.88 --max-model-len 16384 ) > "$LOG" 2>&1 &
SRV=$!
echo "[selftest] server pid=$SRV; waiting for /health (max ~280s)..."

READY=0
for i in $(seq 1 70); do
  if curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null | grep -q 200; then
    READY=1; echo "[selftest] READY after ~$((i*4))s"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[selftest] SERVER DIED during boot:"; tail -20 "$LOG"; exit 2; fi
  sleep 4
done
[ "$READY" = 1 ] || { echo "[selftest] TIMEOUT"; tail -20 "$LOG"; kill $SRV 2>/dev/null; exit 3; }

export VLLM_URL="http://localhost:$PORT"
echo "[selftest] running pipeline: --dataset $DS --limit $LIM via vLLM"
/home/ud3d4/.conda/envs/llmft/bin/python scripts/run_ensemble_fast.py \
    --dataset "$DS" --gpu 0 --models gemma4-e4b --twopass gemma4-e4b \
    --limit "$LIM" --extract-only --tag vllm_test 2>&1 | tail -30
RC=${PIPESTATUS[0]}

kill $SRV 2>/dev/null; sleep 2; pkill -9 -f "vllm serve" 2>/dev/null || true
echo "[selftest] DONE pipeline_rc=$RC"
