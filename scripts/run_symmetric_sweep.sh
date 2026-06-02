#!/usr/bin/env bash
# NCCL >= 2.27 symmetric memory vs the measured latency floors (roadmap Phase 4).
#
# The repo's committed floors (results/, NCCL 2.18.3): 23.1 us eager / 13.7 us CUDA Graph.
# NCCL 2.27 introduced user-buffer / symmetric (window) registration with published
# small-message latency gains of up to 9x. This script measures, on the SAME 4-GPU slice
# (host GPUs 2,3,4,5 = PCI 41/44/86/87) and inside a container shipping NCCL 2.29.2:
#
#   arm A  no registration            -> what a newer NCCL gives for free vs 2.18.3
#   arm B  -R 2 symmetric registration -> what symmetric memory adds on top
#   each both eager and CUDA-Graph (-G 1), small-message range 8B..16MB
#
# Run on the GPU box, on a QUIET box (latency floors are host-jitter-sensitive - the
# repo's own Phase 2 finding):  bash run_symmetric_sweep.sh [output_dir]
set -euo pipefail

IMG=nvcr.io/nvidia/pytorch:26.02-py3
GPUS='"device=2,3,4,5"'
OUT="${1:-/home/user/sa-portfolio/nccl-symmetric/results}"
mkdir -p "$OUT"

# box/clock state alongside the measurement
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,memory.used --format=csv \
  > "$OUT/boxstate.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
  > "$OUT/other_tenants.csv" || true

docker run --rm --gpus "$GPUS" --shm-size=8g -v "$OUT:/out" "$IMG" bash -c '
set -e
# nccl-tests from source so the -R (registration) flag is available; links the container NCCL.
cd /tmp
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
make -C nccl-tests -j 16 > /out/nccl_tests_build.log 2>&1
NCCL_VER=$(python3 -c "import torch; print(\".\".join(map(str, torch.cuda.nccl.version())))")
echo "NCCL version: $NCCL_VER" | tee /out/nccl_version.txt

# small-message focus: 8B .. 16MB (the latency-floor regime), 4 GPUs, 50 iters like the
# committed sweeps. -R 2 = symmetric (window) registration.
COMMON="-b 8 -e 16M -f 2 -g 4 -w 5 -n 50"
cd /tmp/nccl-tests/build
for reg in 0 2; do
  for graph in 0 1; do
    tag="reg${reg}_graph${graph}"
    echo "== all_reduce $tag =="
    ./all_reduce_perf $COMMON -R $reg -G $graph | tee "/out/all_reduce_${tag}.txt"
  done
done
chown -R 1000:1000 /out
'
echo "DONE -> $OUT"
