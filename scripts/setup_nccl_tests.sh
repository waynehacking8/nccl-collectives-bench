#!/usr/bin/env bash
set -euo pipefail
[ -d nccl-tests ] || git clone https://github.com/NVIDIA/nccl-tests.git
make -C nccl-tests -j
echo ">> topology:"; nvidia-smi topo -m || true
