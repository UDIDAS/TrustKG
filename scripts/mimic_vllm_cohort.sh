#!/usr/bin/env bash
# One MIMIC cohort, one GPU, one dedicated vLLM server (server + MedCPT/NER share the GPU).
# Run two instances in parallel (mimiciii/GPU0/8000 + mimiciv/GPU1/8001) to use both GPUs.
#   Usage: mimic_vllm_cohort.sh <mimiciii|mimiciv> <gpu> <port>
# Per-note checkpointed (resumable); tail /dev/shm/mimic_<cohort>.log
set -uo pipefail
cd /home/ud3d4/Desktop/TrustKG
DS="${1:?cohort}"; GPU="${2:?gpu}"; PORT="${3:?port}"
SLOG="results/vllm_server_${DS}.log"
PY=/home/ud3d4/.conda/envs/llmft/bin/python
ts () { date +%H:%M:%S; }

start_server () {  # $1 = hf_id ; kills only THIS port's server (not the other cohort's)
  # free GPU $GPU FULLY: EngineCore worker subprocs don't match "vllm serve", so
  # kill everything nvidia-smi reports on this GPU or the next server OOMs on their leftover VRAM
  for pid in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
  pkill -9 -f "vllm serve .*--port $PORT" 2>/dev/null || true; sleep 6
  ( source scripts/vllm_env.sh; export CUDA_VISIBLE_DEVICES=$GPU
    exec vllm serve "$1" --port "$PORT" --served-model-name "$1" \
         --gpu-memory-utilization 0.85 --max-model-len 16384 ) > "$SLOG" 2>&1 &
  for i in $(seq 1 100); do
    if curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null | grep -q 200; then
      echo "[$(ts)][$DS] server UP: $1 (~$((i*4))s)"; return 0; fi
    sleep 4
  done
  echo "[$(ts)][$DS] FATAL: server boot failed: $1"; tail -15 "$SLOG"; return 1
}

MODELS=(
  "gemma4-e4b|google/gemma-4-E4B-it|1"
  "llama32-3b|meta-llama/Llama-3.2-3B-Instruct|0"
  "qwen3-4b|Qwen/Qwen3-4B|0"
  "medgemma-4b|google/medgemma-4b-it|0"
)

echo "[$(ts)][$DS] === START on GPU$GPU port$PORT ==="
for spec in "${MODELS[@]}"; do
  IFS='|' read -r mtag hfid tp <<< "$spec"
  if [ "$(ls results/extraction/mimic_${DS}/bymodel/$mtag/*.json 2>/dev/null | wc -l)" -ge 400 ]; then
    echo "[$(ts)][$DS] $mtag already complete (400 cached), skip"; continue; fi
  start_server "$hfid" || exit 1
  export VLLM_URL="http://localhost:$PORT"
  tpflag=""; [ "$tp" = "1" ] && tpflag="--twopass $mtag"
  echo "[$(ts)][$DS] --- extract $mtag (twopass=$tp) ---"
  $PY scripts/run_ensemble_fast.py --dataset "$DS" --gpu "$GPU" --models "$mtag" $tpflag \
      --extract-only --tag "mimic_${DS}" --note-batch 16 \
      2>&1 | grep --line-buffered -vE "HTTP Request|resolve-cache|Temporary Redirect" || true
  unset VLLM_URL
done
for pid in $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
pkill -9 -f "vllm serve .*--port $PORT" 2>/dev/null || true

echo "[$(ts)][$DS] === extraction cached; Phase B union+validate ==="
$PY scripts/run_ensemble_fast.py --dataset "$DS" --gpu "$GPU" \
    --models gemma4-e4b llama32-3b qwen3-4b medgemma-4b --twopass gemma4-e4b \
    --tag "mimic_${DS}" 2>&1 | tail -30 || true
echo "[$(ts)][$DS] === COHORT COMPLETE ==="
