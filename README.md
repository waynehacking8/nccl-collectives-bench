# NCCL Collectives Benchmark — H100 NVSwitch

Micro-benchmarks of the NCCL collective operations that bound distributed LLM training
and tensor-parallel inference — **all-reduce, all-gather, reduce-scatter** — measured on a
**4-GPU slice of an 8× H100 NVSwitch host** (scaling study uses 2/4/6 GPUs), with
bus-bandwidth analysis against the theoretical link budget.

Makes multi-GPU communication concrete: all-reduce bus bandwidth measured across message
sizes, where it saturates NVLink, and why tensor parallelism is communication-bound at
small batch sizes.

## What this is
- A thin, reproducible wrapper over NVIDIA `nccl-tests` (the canonical tool) plus a parser
  that turns raw output into tidy CSV/JSON.
- A bandwidth sweep across message sizes (8 B → 8 GB) for all-reduce / all-gather / reduce-scatter.
- Analysis: measured **bus bandwidth** vs NVLink theoretical, the small-message latency floor,
  and what it implies for TP=4 LLM inference.

## What this is NOT
- Not a reimplementation of NCCL — it drives the official `nccl-tests` and adds analysis.
- Not multi-node (yet) — single 8× H100 NVSwitch box, NVLink. The same harness extends to
  InfiniBand multi-node (roadmap) by changing the launcher.

## Hardware
- 8× NVIDIA H100 80GB SXM5 on an NVSwitch fabric (all pairs NV18); runs use a 4-GPU slice
  (the scaling study sweeps 2/4/6 GPUs). `nccl-tests` + CUDA toolkit.

## Layout
```
scripts/setup_nccl_tests.sh   # clone + build nvidia/nccl-tests
scripts/run_sweep.sh          # all_reduce/all_gather/reduce_scatter across sizes -> results/*.txt
analysis/parse.py             # raw nccl-tests output -> results/*.csv
analysis/plot.py              # bandwidth vs size + busbw/algbw curves -> results/*.png
analysis/theoretical.py       # NVLink budget + % of peak achieved
docs/design-decisions.md      # busbw vs algbw, why all-reduce is the one to watch
docs/roadmap.md
results/                      # outputs (populated on the H100 NVSwitch box)
```

## Quick start (run on the H100 NVSwitch box)
```bash
make setup            # build nccl-tests
make sweep            # run the collective sweeps -> results/
make analyze          # parse + plot + compute % of NVLink peak -> results/report.md
```

## Results — measured on a 4-GPU slice of an 8× H100 80GB SXM5 NVSwitch host (NCCL 2.18.3)

Full writeup: [`results/report.md`](results/report.md). Bandwidth curves: `results/busbw.png`.

NVLink budget (measured via `nvidia-smi nvlink --status`): 18 links × 26.562 GB/s = **478 GB/s** per-GPU unidirectional.

| collective | peak busbw | % of NVLink uni | small-msg latency floor |
|---|---|---|---|
| all_reduce | 366 GB/s | 77% | 22.7 µs |
| all_gather | 344 GB/s | 72% | 16.8 µs |
| reduce_scatter | 350 GB/s | 73% | 21.4 µs |

**Algorithm study (all_reduce busbw):** NVLS (NVLink SHARP, in-network reduction on NVSwitch)
beats Ring at every size — 376 vs 366 GB/s @8GB, 359 vs 340 @256MB — and Tree (259 GB/s,
multi-node-oriented) trails both. **Protocol study @256MB:** Simple 340 / LL128 313 / LL 147 GB/s.

**Scaling with GPU count** (all_reduce peak busbw, `analysis/scaling.py`): 2→347, 4→365,
**6→443 GB/s**. Busbw rises with N because higher GPU counts utilize the NVSwitch fabric
more fully — more concurrent NVLink paths and better NVLS (in-switch reduction) efficiency
keep the links saturated. (The ring factor 2(N−1)/N is *divided out* of algbw to define
busbw precisely so it compares across N; it is the divisor that produces busbw, not a
mechanism that pushes it up.) So the 4-GPU 366 GB/s above is a mid-scale operating point,
not the ceiling. The 6-GPU 443 GB/s is **93% of the 478 GB/s figure**, but read that as an
optimistic upper-bound framing, not "near line-rate": the 478 GB/s is the *unidirectional*
per-GPU link budget, whereas steady-state all-reduce traffic is simultaneously bidirectional,
and the literature commonly reports ~75–85% of the unidirectional budget.

> Note: `make sweep` defaults to 4 GPUs. On a shared box, pin to free GPUs and never touch
> a busy one: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4,5 make sweep`
> (or `--gpus '"device=2,3,4,5"'` under Docker / NGC `nvcr.io/nvidia/pytorch`).

### Next: the TP-inference latency wall *(frontier extension, in progress)*
The sweep above is steady-state bandwidth. LLM tensor-parallel decode lives in the *opposite*
regime — tiny (≤64 KB) all-reduces, twice per layer, latency-bound on the ~22 µs floor. See
[`tp_latency/`](tp_latency/): CUDA-Graph capture vs eager, custom one-shot all-reduce vs NCCL,
and an analytical comms-roofline for TP=N decode validated against measurement.

---

## References
- [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests) — the canonical benchmark this harness drives.
- [NVIDIA/nccl](https://github.com/NVIDIA/nccl) — the collective communication library under test.

## Disclaimer
Personal project for learning and benchmarking. Views and results are my own and do not represent any employer.

---

_Part of my portfolio — [waynehacking8.github.io](https://waynehacking8.github.io/). Writeup: [Where tensor-parallel inference hits the NVLink wall](https://waynehacking8.github.io/blog/nccl-nvlink-bandwidth/)._
