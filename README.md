# NCCL Collectives Benchmark — 4× H100 NVLink

Micro-benchmarks of the NCCL collective operations that bound distributed LLM training
and tensor-parallel inference — **all-reduce, all-gather, reduce-scatter** — measured on
**4× H100 over NVLink**, with bus-bandwidth analysis against the theoretical link budget.

Built to make multi-GPU communication concrete: not "I've heard of NCCL" but "I measured
all-reduce bus bandwidth across message sizes, saw where it saturates NVLink, and can
explain why tensor parallelism is communication-bound at small batch sizes."

## What this is
- A thin, reproducible wrapper over NVIDIA `nccl-tests` (the canonical tool) plus a parser
  that turns raw output into tidy CSV/JSON.
- A bandwidth sweep across message sizes (8 B → 8 GB) for all-reduce / all-gather / reduce-scatter.
- Analysis: measured **bus bandwidth** vs NVLink theoretical, the small-message latency floor,
  and what it implies for TP=4 LLM inference.

## What this is NOT
- Not a reimplementation of NCCL — it drives the official `nccl-tests` and adds analysis.
- Not multi-node (yet) — single 4×H100 box, NVLink. The same harness extends to InfiniBand
  multi-node (roadmap) by changing the launcher.

## Hardware
- 4× NVIDIA H100 80GB, NVLink (intra-node). `nccl-tests` + CUDA toolkit.

## Layout
```
scripts/setup_nccl_tests.sh   # clone + build nvidia/nccl-tests
scripts/run_sweep.sh          # all_reduce/all_gather/reduce_scatter across sizes -> results/*.txt
analysis/parse.py             # raw nccl-tests output -> results/*.csv
analysis/plot.py              # bandwidth vs size + busbw/algbw curves -> results/*.png
analysis/theoretical.py       # NVLink budget + % of peak achieved
docs/design-decisions.md      # busbw vs algbw, why all-reduce is the one to watch
docs/roadmap.md
results/                      # outputs (populated on the 4xH100 box)
```

## Quick start (run on the 4×H100 box)
```bash
make setup            # build nccl-tests
make sweep            # run the collective sweeps -> results/
make analyze          # parse + plot + compute % of NVLink peak -> results/report.md
```

## Results
Populated after running on the 4×H100 box — see `results/`. **(in progress)**
