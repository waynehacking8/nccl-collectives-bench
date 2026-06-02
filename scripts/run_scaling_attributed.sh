#!/usr/bin/env bash
# Scaling-study algorithm attribution (roadmap item; background in results/scaling_report.md).
#
# The committed 2/4/6-GPU scaling runs (results/scaling.txt, NCCL 2.18.3) used NCCL's automatic
# algorithm selection without NCCL_DEBUG capture, so the 6-GPU 443 GB/s busbw cannot be
# attributed: Ring would mean 93% physical link utilization, NVLS would mean ~56% (busbw
# inflated by a ring factor that does not describe in-switch-reduction traffic).
#
# This script re-runs the large-message scaling sweep with the tuner's decision captured:
#
#   3 GPU counts (-g 2/4/6)  x  3 algorithm arms (auto / NCCL_ALGO=Ring / NCCL_ALGO=NVLS)
#   = 9 arms, each with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING and stderr captured
#     in the same per-arm log, so the tuner's actual algorithm choice is on record.
#
# GPU 0 runs a production service and is never used. The container sees host GPUs 2..7
# (all idle at run time; see boxstate.csv / other_tenants.csv saved next to the results),
# so container device 0..5 = host GPU 2..7 and -g N uses host GPUs 2..(N+1).
#
# Run on the GPU box:  bash run_scaling_attributed.sh [output_dir]
set -euo pipefail

IMG=nvcr.io/nvidia/pytorch:26.02-py3
GPUS='"device=2,3,4,5,6,7"'
OUT="${1:-/home/user/sa-portfolio/nccl-scaling-attributed/results}"
mkdir -p "$OUT"

# box/clock state alongside the measurement
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,memory.used --format=csv \
  > "$OUT/boxstate.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
  > "$OUT/other_tenants.csv" || true

docker run --rm --gpus "$GPUS" --shm-size=8g -v "$OUT:/out" "$IMG" bash -c '
set -e
# nccl-tests from source, linked against the container NCCL.
cd /tmp
git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
make -C nccl-tests -j 16 > /out/nccl_tests_build.log 2>&1
NCCL_VER=$(python3 -c "import torch; print(\".\".join(map(str, torch.cuda.nccl.version())))")
echo "NCCL version: $NCCL_VER" | tee /out/nccl_version.txt

# Same large-message range as the committed scaling study: 256 MB .. 8 GB, x2 steps.
COMMON="-b 256M -e 8G -f 2 -w 5 -n 50"
cd /tmp/nccl-tests/build
for g in 2 4 6; do
  for algo in auto Ring NVLS; do
    tag="g${g}_${algo}"
    echo "== all_reduce $tag =="
    ALGO_ENV=""
    if [ "$algo" != "auto" ]; then ALGO_ENV="NCCL_ALGO=$algo"; fi
    # NCCL debug output and test stderr both land in the per-arm log (2>&1) — the debug
    # lines are the attribution evidence this experiment exists to capture.
    env NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING $ALGO_ENV \
      ./all_reduce_perf $COMMON -g $g > "/out/${tag}.log" 2>&1 \
      || echo "ARM FAILED rc=$?" >> "/out/${tag}.log"
  done
done
chown -R 1000:1000 /out
'
echo "DONE -> $OUT"
