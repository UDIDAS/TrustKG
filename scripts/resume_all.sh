#!/bin/bash
# Auto-resume TRUST-KG runs after a cluster restart. IDEMPOTENT: relaunches only
# what's incomplete and not already running. Results are checkpointed in results/
# (persistent /home NFS); the runners skip already-done patients/notes, so a
# restart loses at most the one in-progress item. Safe to run anytime.
set -u
cd /home/ud3d4/Desktop/TrustKG || exit 1
PY=/home/ud3d4/.conda/envs/llmft/bin/python
export TRUSTKG_ROOT=/home/ud3d4/Desktop/TrustKG
export HF_HOME=/scratch/ud3d4/hf_cache HF_HUB_CACHE=/scratch/ud3d4/hf_cache/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(grep -h '^HF_TOKEN' /home/ud3d4/Desktop/.env 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' "')
unset GOOGLE_APPLICATION_CREDENTIALS         # extraction is local; never BigQuery
mkdir -p /scratch/ud3d4/hf_cache/hub         # scratch wiped on restart; models re-download here
LOG=results/resume_all.log
count(){ [ -f "$1" ] && "$PY" -c "import json;print(len(json.load(open('$1'))))" 2>/dev/null || echo 0; }
running(){ pgrep -f "$1" >/dev/null 2>&1; }

echo "$(date) resume_all: start" >> "$LOG"

# ── CORAL ensemble: resume each cohort if <20 done and not already running ──
c0=$(count results/ens3_metrics_gpu0.json); c1=$(count results/ens3_metrics_gpu1.json)
if [ "$c0" -lt 20 ] && ! running "run_coral_ensemble.py --gpu 0"; then
  PDAC=$(for i in $(seq 0 19); do echo -n "pdac_$i "; done)
  nohup "$PY" scripts/run_coral_ensemble.py --gpu 0 --models gemma3-4b qwen3-8b llama32-3b \
      --twopass gemma3-4b --tag ens3 --patients $PDAC >> results/ens_gpu0.log 2>&1 &
  echo "$(date) resume_all: relaunched CORAL gpu0 (was $c0/20)" >> "$LOG"
fi
if [ "$c1" -lt 20 ] && ! running "run_coral_ensemble.py --gpu 1"; then
  BRCA=$(for i in $(seq 20 39); do echo -n "brca_$i "; done)
  nohup "$PY" scripts/run_coral_ensemble.py --gpu 1 --models gemma3-4b qwen3-8b llama32-3b \
      --twopass gemma3-4b --tag ens3 --patients $BRCA >> results/ens_gpu1.log 2>&1 &
  echo "$(date) resume_all: relaunched CORAL gpu1 (was $c1/20)" >> "$LOG"
fi

# ── MIMIC: HELD for the optimized runner (scripts/run_mimic_fast.py). We do NOT
#    auto-launch it on resume — it's started deliberately after CORAL finishes and the
#    fast (batched) runner is benchmarked/validated. run_mimic_fast is resumable
#    (results/mimic_*_fast_metrics.json), so it's safe to (re)start when we do.
m3=$(count results/mimic_mimiciii_fast_metrics.json); m4=$(count results/mimic_mimiciv_fast_metrics.json)

echo "$(date) resume_all: done (coral $c0/$c1 of 20; mimic-fast $m3/$m4 of 400; MIMIC auto-start held)" >> "$LOG"
