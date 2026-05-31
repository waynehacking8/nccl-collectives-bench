#!/usr/bin/env bash
# Sweep the key collectives across message sizes on $GPUS GPUs.
set -euo pipefail
GPUS="${1:-4}"
mkdir -p results
COMMON="-b 8 -e 8G -f 2 -g $GPUS -w 5 -n 50"   # min 8B, max 8GB, x2 step, 50 iters
for op in all_reduce all_gather reduce_scatter; do
  echo ">> $op"
  ./nccl-tests/build/${op}_perf $COMMON | tee "results/${op}.txt"
done
