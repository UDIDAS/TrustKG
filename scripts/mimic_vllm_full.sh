#!/usr/bin/env bash
# Full MIMIC ensemble extraction via vLLM, model-by-model, both cohorts.
#   For each model: start its vLLM server (GPU0), run all notes of both MIMIC
#   cohorts through the llmft pipeline (MedCPT/NER on GPU1, generation offloaded
#   to the server via VLLM_URL), stop server.  Phase A is checkpointed per-note,
#   so this is fully resumable.  Runs detached (setsid); tail results/mimic_vllm_full.log.
# Order puts gemma4-2pass/mimiciii first so the first checkpoint reveals the real rate.
set -uo pipefail
cd /home/ud3d4/Desktop/TrustKG
PORT=8000; SLOG=results/vllm_server.log
PY=/home/ud3d4/.conda/envs/llmft/bin/python
ts () { date +%H:%M:%S; }

start_server () {  # $1 = hf_id
  pkill -9 -f "vllm serve" 2>/dev/null || true; sleep 4
  ( source scripts/vllm_env.sh; export CUDA_VISIBLE_DEVICES=0
    exec vllm serve "$1" --port $PORT --served-model-name "$1" \
         --gpu-memory-utilization 0.88 --max-model-len 16384 ) > "$SLOG" 2>&1 &
  for i in $(seq 1 90); do
    if curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null | grep -q 200; then
      echo "[$(ts)] server UP: $1 (after ~$((i*4))s)"; return 0; fi
    sleep 4
  done
  echo "[$(ts)] [FATAL] server failed to boot: $1"; tail -20 "$SLOG"; return 1
}

# model_tag  hf_id  twopass(1|0)
MODELS=(
  "gemma4-e4b|google/gemma-4-E4B-it|1"
  "llama32-3b|meta-llama/Llama-3.2-3B-Instruct|0"
  "qwen3-4b|Qwen/Qwen3-4B|0"
  "medgemma-4b|google/medgemma-4b-it|0"
)

echo "[$(ts)] === MIMIC vLLM full ensemble run START ==="
for spec in "${MODELS[@]}"; do
  IFS='|' read -r mtag hfid tp <<< "$spec"
  start_server "$hfid" || exit 1
  export VLLM_URL="http://localhost:$PORT"
  tpflag=""; [ "$tp" = "1" ] && tpflag="--twopass $mtag"
  for ds in mimiciii mimiciv; do
    echo "[$(ts)] --- extract $mtag on $ds (twopass=$tp) ---"
    $PY scripts/run_ensemble_fast.py --dataset "$ds" --gpu 1 \
        --models "$mtag" $tpflag --extract-only --tag "mimic_${ds}" \
        2>&1 | grep -vE "HTTP Request|resolve-cache|Temporary Redirect" || true
  done
  unset VLLM_URL
done
pkill -9 -f "vllm serve" 2>/dev/null || true

echo "[$(ts)] === all extraction cached; Phase B union+validate per cohort ==="
for ds in mimiciii mimiciv; do
  echo "[$(ts)] --- union $ds ---"
  $PY scripts/run_ensemble_fast.py --dataset "$ds" --gpu 1 \
      --models gemma4-e4b llama32-3b qwen3-4b medgemma-4b --twopass gemma4-e4b \
      --tag "mimic_${ds}" 2>&1 | tail -25 || true
done
echo "[$(ts)] === MIMIC vLLM FULL RUN COMPLETE ==="
