## All-reduce bandwidth scaling with GPU count

NVSwitch fabric, all_reduce, peak bandwidth vs GPU count, NCCL automatic algorithm selection (NVLink budget 478 GB/s/GPU unidirectional):

| GPUs | peak busbw | peak algbw | ring factor 2(N-1)/N | busbw as % of NVLink budget |
|---|---|---|---|---|
| 2 | 347 GB/s | 347 GB/s | 1.00 | 73% |
| 4 | 365 GB/s | 243 GB/s | 1.50 | 76% |
| 6 | 443 GB/s | 266 GB/s | 1.67 | 93% |

**How to read this (and how not to).** busbw is nccl-tests' *ring-equivalent* normalization: busbw = algbw x 2(N-1)/N, the per-link traffic a ring algorithm would generate at the measured algbw. It equals physical per-link traffic only when the algorithm actually is Ring. These runs use NCCL's automatic algorithm selection (no `NCCL_ALGO` pin; `NCCL_DEBUG` output was not captured), which changes what the busbw column means at each N:

- **2 GPUs** — Ring and NVLS traffic patterns coincide (each GPU sends and receives the full buffer once), so 347 GB/s is a real per-link rate (73% of budget).
- **4 GPUs** — the auto-selected result (365) matches the pinned-**Ring** row of the algorithm sweep (366, `results/algo_sweep.txt`), not pinned NVLS (376). busbw here is a real per-link rate (76% of budget).
- **6 GPUs** — the 443 busbw is **ambiguous**. If the tuner stayed on Ring, it is a physical 93% of the link budget — a 17-point jump in link efficiency over the 4-GPU point with no mechanism to explain it. If the tuner switched to NVLS at the higher rank count (the parsimonious reading: pinned NVLS already beats Ring at 4 GPUs, and NVLS busbw is inflated by a ring factor that does not describe its traffic — with in-switch reduction each GPU ships its data once, moving only ~algbw = 266 GB/s, 56% of budget), then most of the 365 -> 443 rise is the normalization formula, not extra bytes on the links. Decomposition under the NVLS reading: busbw ratio 443/376 = 1.18 = factor ratio (1.67/1.50 = 1.11) x physical algbw gain (266/251 = 1.06).

The committed logs cannot distinguish the two readings (no `NCCL_DEBUG` output); resolving this is a roadmap item (re-run with `NCCL_DEBUG=INFO,TUNING` and `NCCL_ALGO` pinned to Ring and NVLS separately). Until then, the defensible statements are:

1. End-to-end all-reduce **algbw** — what a training step actually sees per byte of gradient — drops from 347 GB/s (2 GPUs) to 243/266 GB/s (4/6 GPUs), because each GPU must move 2(N-1)/N x more data as N grows. There is no free lunch in scaling out the ring.
2. The 4-GPU ~366 GB/s busbw quoted elsewhere in this repo **is** a physical per-link rate, confirmed independently by the pinned-Ring sweep.
3. The 6-GPU "93% of budget" figure should **not** be quoted as physical link utilization — under the more likely NVLS reading the links carry ~56% of budget.

