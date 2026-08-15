#!/bin/bash
# Auto-start MIMIC oncology extraction once the CORAL ensemble run finishes and
# frees the GPUs. LOCAL ONLY — reads data/mimic_oncology/*.jsonl; never BigQuery.
set -u
cd /home/ud3d4/Desktop/TrustKG
PY=/home/ud3d4/.conda/envs/llmft/bin/python
export TRUSTKG_ROOT=/home/ud3d4/Desktop/TrustKG
export HF_HOME=/scratch/ud3d4/hf_cache HF_HUB_CACHE=/scratch/ud3d4/hf_cache/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset GOOGLE_APPLICATION_CREDENTIALS   # belt: extraction must never reach BigQuery

count() { [ -f "$1" ] && "$PY" -c "import json;print(len(json.load(open('$1'))))" 2>/dev/null || echo 0; }

echo "$(date) watcher up: waiting for CORAL ensemble (both GPUs -> 20 patients)"
until [ "$(count results/ens3_metrics_gpu0.json)" -ge 20 ] && [ "$(count results/ens3_metrics_gpu1.json)" -ge 20 ]; do
  sleep 600
done
echo "$(date) CORAL ensemble complete -> launching FAST MIMIC extraction (mimiciii on GPU0, mimiciv on GPU1)"
# Optimized runner: resident model + batched chunks (run_mimic_fast.py).
nohup "$PY" scripts/run_mimic_fast.py --source mimiciii --gpu 0 > results/mimic_ex_gpu0.log 2>&1 &
nohup "$PY" scripts/run_mimic_fast.py --source mimiciv  --gpu 1 > results/mimic_ex_gpu1.log 2>&1 &
wait
echo "$(date) MIMIC extraction finished"
