# Source this before any vLLM command:  source scripts/vllm_env.sh
# Captures the env fixes needed to run vLLM nightly + Gemma-4-E4B on this HPC box.
#   1. env's newer libstdc++ on LD_LIBRARY_PATH (CXXABI_1.3.15)
#   2. spack CUDA 12.9.1 toolkit for nvcc (matches the cu129 nightly wheels)
#   3. vllm env bin on PATH (for the `ninja` JIT build tool)
#   4. flashinfer sampler off (avoids a JIT compile path)
#   5. all caches -> /scratch (protect the 50GB home quota)
export CUDA_HOME=/opt/ohpc/pub/spack/apps/linux-zen2/cuda-12.9.1-ok6xerx7xeslx4x46rgcgmfnxp53idxb
export PATH=/home/ud3d4/.conda/envs/vllm/bin:$CUDA_HOME/bin:${PATH:-}
export LD_LIBRARY_PATH=/home/ud3d4/.conda/envs/vllm/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export HF_HOME=/scratch/ud3d4/hf_cache HF_HUB_CACHE=/scratch/ud3d4/hf_cache/hub
export HF_TOKEN=$(grep -h '^HF_TOKEN' /home/ud3d4/Desktop/.env 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' "')
export VLLM_LOGGING_LEVEL=WARNING VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT=/scratch/ud3d4/vllm_cache XDG_CACHE_HOME=/scratch/ud3d4/xdg_cache
mkdir -p /scratch/ud3d4/vllm_cache /scratch/ud3d4/xdg_cache
# vLLM python: /home/ud3d4/.conda/envs/vllm/bin/python  (vLLM 0.27.2rc1 nightly / torch 2.13+cu129)
# Gemma-4-E4B benchmark: 1206 notes/hr 1-pass on one A6000 (vs ~28/hr HF).
