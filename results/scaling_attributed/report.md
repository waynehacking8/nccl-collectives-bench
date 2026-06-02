# Scaling-study algorithm attribution

**Question (from `results/scaling_report.md`):** the committed 2/4/6-GPU scaling runs (`results/scaling.txt`, NCCL 2.18.3, automatic algorithm selection, no `NCCL_DEBUG` capture) report a 6-GPU peak busbw of 443 GB/s. Is that Ring traffic — a physical 93% of the 478 GB/s per-GPU NVLink budget — or NVLS traffic, where busbw is inflated by a ring-factor normalization that does not describe in-switch-reduction traffic?

**Setup:** all_reduce_perf, 256 MB..8 GB, `-f 2 -w 5 -n 50`, NCCL 2.29.2 (pytorch:26.02 container), host GPUs 2..7 (GPU 0 = production, never used; box state in `boxstate.csv` / `other_tenants.csv`). 9 arms: `-g 2/4/6` x {auto, `NCCL_ALGO=Ring`, `NCCL_ALGO=NVLS`}, each with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING` captured in the same per-arm log (`g{N}_{arm}.log`).

## 1. What the tuner actually picks (from the NCCL debug log)

| GPUs | auto selection (all 6 sizes) | channels | evidence |
|---|---|---|---|
| 2 | **RING** (proto SIMPLE) | 0..23 | `g2_auto.log`: `AllReduce: ... Bytes -> Algo RING proto SIMPLE` |
| 4 | **RING** (proto SIMPLE) | 0..23 | `g4_auto.log`: `AllReduce: ... Bytes -> Algo RING proto SIMPLE` |
| 6 | **NVLS** (proto SIMPLE) | 0..15 | `g6_auto.log`: `AllReduce: ... Bytes -> Algo NVLS proto SIMPLE` |

The tuner picks **Ring at 2 and 4 GPUs** and switches to **NVLS at 6 GPUs**. The selection is uniform across all message sizes (256 MB..8 GB) in every arm.

## 2. All nine arms: reported vs physical bandwidth

busbw is nccl-tests' ring-equivalent normalization (algbw x 2(N-1)/N). Physical per-link traffic: Ring -> busbw (the normalization is exact for ring traffic); NVLS -> ~algbw (in-switch reduction ships each GPU's data once).

| GPUs | arm | algo used | peak busbw | peak algbw | physical per-link | % of 478 GB/s budget |
|---|---|---|---|---|---|---|
| 2 | auto | RING | 347 GB/s | 347 GB/s | ~347 GB/s | 73% |
| 2 | `NCCL_ALGO=Ring` | RING | 348 GB/s | 348 GB/s | ~348 GB/s | 73% |
| 2 | `NCCL_ALGO=NVLS` | NVLS | 207 GB/s | 207 GB/s | ~207 GB/s | 43% |
| 4 | auto | RING | 365 GB/s | 243 GB/s | ~365 GB/s | 76% |
| 4 | `NCCL_ALGO=Ring` | RING | 365 GB/s | 243 GB/s | ~365 GB/s | 76% |
| 4 | `NCCL_ALGO=NVLS` | NVLS | 376 GB/s | 251 GB/s | ~251 GB/s | 52% |
| 6 | auto | NVLS | 443 GB/s | 266 GB/s | ~266 GB/s | 56% |
| 6 | `NCCL_ALGO=Ring` | RING | 367 GB/s | 220 GB/s | ~367 GB/s | 77% |
| 6 | `NCCL_ALGO=NVLS` | NVLS | 443 GB/s | 266 GB/s | ~266 GB/s | 56% |

## 3. Findings

1. **The 6-GPU 443 GB/s is NVLS, and it is not physical link traffic.** The tuner selects NVLS at 6 GPUs (auto 443 == pinned NVLS 443 GB/s; pinned Ring manages only 367). With in-switch reduction the links physically carry ~algbw = 266 GB/s = **56% of budget**, not 93%. Most of the 4->6 GPU busbw rise (365 -> 443) is the ring factor (1.50 -> 1.67) applied to non-ring traffic, plus the algorithm switch.

2. **Ring's physical link efficiency is flat across N — there never was a jump to explain.** Pinned Ring busbw (= physical): 348 (73%) -> 365 (76%) -> 367 GB/s (77%) at 2/4/6 GPUs. The "93% link utilization" reading of the committed 443 was an artifact of applying the ring formula to NVLS traffic.

3. **The attribution transfers to the committed scaling study.** This run reproduces the committed peaks (347/365/443 GB/s, NCCL 2.18.3 bare-metal) at 347/365/443 GB/s under NCCL 2.29.2 in a container — within 0.5% at every point. The tuner's choices measured here (Ring/Ring/NVLS) are therefore the choices behind the committed numbers.

4. **What NVLS actually buys at 6 GPUs is +21% end-to-end bandwidth, not 93% link utilization.** algbw — what a training step sees per byte of gradient — is 220 GB/s under Ring vs 266 GB/s under NVLS. The win comes from *removing* traffic from the links (in-switch reduction), not from saturating them.

5. **Tuner non-optimality, both directions.** At 4 GPUs the tuner picks Ring (365) although pinned NVLS is 3% faster (376) — consistent with the repo's 4-GPU algorithm sweep (`results/algo_sweep.txt`). At 2 GPUs pinning NVLS is a disaster: 207 vs 348 GB/s (40% slower) — the multicast path has per-message overhead that a 2-rank ring does not.

## 4. Cross-run caveats

- The committed scaling study ran NCCL 2.18.3 bare-metal on an unrecorded idle GPU slice; this run is NCCL 2.29.2 in the pytorch:26.02 container on host GPUs 2..7. Peaks agree within 0.5%, so the version/container difference does not matter in this large-message regime (it does matter for small-message latency — see `results/symmetric/report.md`).
- GPU 7 held an idle ollama allocation (522 MiB, 0% util) during the run (`other_tenants.csv`); GPU 0's production service never participates.
- Physical per-link traffic for NVLS is approximated as algbw. The exact multicast fan-out traffic is switch-internal and not observable from the host; ~algbw is the per-GPU NVLink port traffic implied by ship-once/receive-once.

## 5. Raw data

- `g{2,4,6}_{auto,Ring,NVLS}.log` — full nccl-tests output with interleaved `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING` lines (the attribution evidence).
- `boxstate.csv`, `other_tenants.csv`, `nccl_version.txt`, `nccl_tests_build.log`.
- Sweep script: `scripts/run_scaling_attributed.sh`; this report + chart: `analysis/scaling_attribution.py` (re-run -> zero diff).
