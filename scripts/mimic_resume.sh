#!/usr/bin/env bash
# Idempotent self-heal for the MIMIC vLLM ensemble run. Safe to call repeatedly
# (cron @reboot + periodic). For each cohort, relaunch its driver ONLY IF it is
# not already running AND extraction is incomplete. Resumes from the per-note cache.
# Guards: flock (single instance) + pgrep (never duplicates a running driver).
exec 9>/tmp/mimic_resume.lock 2>/dev/null || exit 0
flock -n 9 || exit 0
cd /home/ud3d4/Desktop/TrustKG || exit 0

launch () {  # $1=cohort  $2=gpu  $3=port
  local ds="$1" gpu="$2" port="$3"
  pgrep -f "mimic_vllm_cohort.sh $ds" >/dev/null 2>&1 && return   # already running
  # 4 models x 400 notes = 1600 cached files => extraction complete, don't relaunch
  local n; n=$(ls results/extraction/mimic_${ds}/bymodel/*/*.json 2>/dev/null | wc -l)
  [ "${n:-0}" -ge 1600 ] && return
  echo "[$(date '+%F %T')] resume: relaunching $ds (gpu$gpu port$port), $n/1600 cached" >> results/mimic_resume.log
  setsid bash scripts/mimic_vllm_cohort.sh "$ds" "$gpu" "$port" </dev/null >>/dev/shm/mimic_${ds}.log 2>&1 &
}

launch mimiciii 0 8000
launch mimiciv  1 8001
