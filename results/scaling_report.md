## All-reduce bandwidth scaling with GPU count

NVSwitch fabric, all_reduce, peak bandwidth vs GPU count, NCCL automatic algorithm selection (NVLink budget 478 GB/s/GPU unidirectional). The "tuner algo" column is measured by the attribution run (`results/scaling_attributed/report.md`); physical per-link traffic is busbw for Ring and ~algbw for NVLS:

| GPUs | peak busbw | peak algbw | ring factor 2(N-1)/N | tuner algo (measured) | physical per-link | % of NVLink budget |
|---|---|---|---|---|---|---|
| 2 | 347 GB/s | 347 GB/s | 1.00 | Ring | ~347 GB/s | 73% |
| 4 | 365 GB/s | 243 GB/s | 1.50 | Ring | ~365 GB/s | 76% |
| 6 | 443 GB/s | 266 GB/s | 1.67 | NVLS | ~266 GB/s | 56% |

**How to read this (and how not to).** busbw is nccl-tests' *ring-equivalent* normalization: busbw = algbw x 2(N-1)/N, the per-link traffic a ring algorithm would generate at the measured algbw. It equals physical per-link traffic only when the algorithm actually is Ring. These runs use NCCL's automatic algorithm selection, and the attribution run (`results/scaling_attributed/report.md`, `analysis/scaling_attribution.py`: same sweep with `NCCL_DEBUG=INFO,TUNING` captured and `NCCL_ALGO` pinned per arm) **measured** what the tuner picks at each N:

- **2 GPUs** — tuner picks **Ring** (debug-log evidence; pinned NVLS collapses to 207 GB/s here). 347 GB/s is a real per-link rate (73% of budget).
- **4 GPUs** — tuner picks **Ring** (auto 365 == pinned Ring 365; pinned NVLS would be 376, consistent with `results/algo_sweep.txt`). busbw here is a real per-link rate (76% of budget).
- **6 GPUs** — tuner switches to **NVLS** (auto 443 == pinned NVLS 443; pinned Ring manages only 367). NVLS busbw is inflated by a ring factor that does not describe its traffic — with in-switch reduction each GPU ships its data once, moving only ~algbw = 266 GB/s, **56% of budget**. Most of the 365 -> 443 rise is the normalization formula plus the algorithm switch, not extra bytes on the links. Decomposition: busbw ratio 443/376 = 1.18 = factor ratio (1.67/1.50 = 1.11) x physical algbw gain (266/251 = 1.06).

The attribution run reproduces these peaks within 0.5% under NCCL 2.29.2, so its tuner choices (Ring/Ring/NVLS) attribute the numbers in this table. The defensible statements:

1. End-to-end all-reduce **algbw** — what a training step actually sees per byte of gradient — drops from 347 GB/s (2 GPUs) to 243/266 GB/s (4/6 GPUs), because each GPU must move 2(N-1)/N x more data as N grows. There is no free lunch in scaling out the ring.
2. The 4-GPU ~366 GB/s busbw quoted elsewhere in this repo **is** a physical per-link rate, confirmed by the pinned-Ring sweep and now by the tuner's logged choice.
3. The 6-GPU "93% of budget" figure is **not** physical link utilization — the measured algorithm is NVLS and the links carry ~56% of budget. Ring's physical link efficiency is flat across N (73% -> 76% -> 77%); the apparent jump was an artifact of the normalization.

