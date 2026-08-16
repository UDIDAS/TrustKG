#!/bin/bash
# Resume the CORAL extractor-comparison combo sweep (run_ensemble_fast, tag=combo_coral)
# after a cluster restart. IDEMPOTENT: relaunches only models with <40 cached patients,
# and only if nothing is already running. Per-model caches + NER live in results/
# (persistent /home NFS) and survive restart; HF weights in /scratch re-download on first
# load (HF_TOKEN from .env covers the gated Gemma/Llama/MedGemma). Safe to run anytime.
# Pool: gemma3-4b, medgemma-4b (GPU0) | llama32-3b, qwen3-4b (GPU1).  (phi4-mini dropped:
# broken on transformers 5.8 -> ImportError LossKwargs.)
set -u
cd /home/ud3d4/Desktop/TrustKG || exit 1
PY=/home/ud3d4/.conda/envs/llmft/bin/python
export TRUSTKG_ROOT=/home/ud3d4/Desktop/TrustKG
export HF_HOME=/scratch/ud3d4/hf_cache HF_HUB_CACHE=/scratch/ud3d4/hf_cache/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(grep -h '^HF_TOKEN' /home/ud3d4/Desktop/.env 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' "')
unset GOOGLE_APPLICATION_CREDENTIALS
mkdir -p /scratch/ud3d4/hf_cache/hub
LOG=results/resume_combo.log
B=results/extraction/combo_coral/bymodel
ccnt(){ ls "$B/$1"/*.json 2>/dev/null | wc -l; }
run(){ "$PY" scripts/run_ensemble_fast.py --dataset coral --gpu "$2" --models "$1" \
       --twopass none --tag combo_coral --extract-only >> "results/combo_$1.log" 2>&1; }

if pgrep -f "run_ensemble_fast.py --dataset coral" >/dev/null 2>&1; then
  echo "$(date) resume_combo: extraction already running, skip" >> "$LOG"; exit 0
fi

echo "$(date) resume_combo: start (gemma=$(ccnt gemma3-4b) medgemma=$(ccnt medgemma-4b) llama=$(ccnt llama32-3b) qwen=$(ccnt qwen3-4b) of 40)" >> "$LOG"
# GPU0 chain: gemma, medgemma | GPU1 chain: llama, qwen  (incomplete models only)
( for m in gemma3-4b medgemma-4b; do [ "$(ccnt "$m")" -lt 40 ] && run "$m" 0; done ) &
( for m in llama32-3b qwen3-4b;   do [ "$(ccnt "$m")" -lt 40 ] && run "$m" 1; done ) &
wait
echo "$(date) resume_combo: chains exited (gemma=$(ccnt gemma3-4b) medgemma=$(ccnt medgemma-4b) llama=$(ccnt llama32-3b) qwen=$(ccnt qwen3-4b))" >> "$LOG"
