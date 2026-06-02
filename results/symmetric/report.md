# NCCL 2.29 symmetric memory vs the committed latency floors

Same 4-GPU NVSwitch slice (host GPUs 2,3,4,5) as every committed measurement; all_reduce_perf, 50 iters, out-of-place times. Committed floors (NCCL 2.18.3): **23.1 us eager / 13.7 us CUDA Graph**. Published claim (NCCL 2.27 release notes): up to **9x** lower small-message latency from symmetric (window) registration.

> Box state during this run: other tenants were active on GPUs outside the measurement slice
> (`other_tenants.csv` — recorded for transparency). All five configurations in this report were
> measured back-to-back in the same session, so the *within-report* comparisons (2.29 plain vs
> 2.29 symmetric, eager vs CUDA Graph) are same-environment and unaffected. The *cross-session*
> comparison against the committed 2.18.3 reference (the small 23.3-vs-25.1 µs delta) spans two
> different box states — read that row as indicative, not as a controlled version regression test.

## Latency floors (median over sizes <= 2 KB)

| configuration | floor (us) | vs 2.18.3 eager (23.1 us) |
|---|---|---|
| NCCL 2.18.3 eager (committed reference) | **23.3** | 0.99x |
| NCCL 2.29.2 eager | **25.1** | 0.92x |
| NCCL 2.29.2 eager + symmetric (-R 2) | **23.6** | 0.98x |
| NCCL 2.29.2 CUDA Graph | **16.3** | 1.41x |
| NCCL 2.29.2 CUDA Graph + symmetric | **19.7** | 1.17x |

## Per-size latency (us, out-of-place)

| size | 2.18 eager | 2.29 eager | 2.29 eager + symmetric (-R 2) | 2.29 CUDA Graph | 2.29 CUDA Graph + symmetric |
|---|---|---|---|---|---|
| 8 B | 23.3 | 23.5 | 23.2 | 16.1 | 19.3 |
| 64 B | 23.3 | 25.1 | 23.6 | 16.4 | 19.8 |
| 512 B | 23.3 | 25.1 | 23.7 | 16.8 | 20.5 |
| 2 KB | 23.2 | 24.7 | 23.4 | 15.2 | 17.6 |
| 8 KB | 23.8 | 25.9 | 24.2 | 15.5 | 18.6 |
| 16 KB | 24.3 | 25.9 | 24.8 | 16.2 | 19.7 |
| 64 KB | 25.5 | 27.1 | 24.2 | 16.8 | 16.5 |
| 256 KB | 27.2 | 28.3 | 24.3 | 17.4 | 17.5 |
| 1 MB | 26.8 | 27.8 | 32.6 | 24.0 | 20.9 |
| 4 MB | 43.4 | 43.4 | 25.6 | 47.8 | 34.8 |
| 16 MB | 101.9 | 101.3 | 76.5 | 105.8 | 84.0 |

## Large-message busbw (GB/s)

| configuration | busbw @ 16 MB |
|---|---|
| NCCL 2.18.3 eager (committed reference) | 247.1 |
| NCCL 2.29.2 eager | 248.3 |
| NCCL 2.29.2 eager + symmetric (-R 2) | 329.1 |
| NCCL 2.29.2 CUDA Graph | 237.8 |
| NCCL 2.29.2 CUDA Graph + symmetric | 299.8 |

## Findings

1. **The published 'up to 9x lower small-message latency' does NOT reproduce on a single-node NVSwitch slice.** Symmetric registration is latency-*neutral* below ~256 KB (23.6 vs 25.1 us eager floor). The 9x claim targets the paths where registration removes proxy/copy work - multi-node networking and NVLS trees - not a 4-GPU NVSwitch all_reduce whose small-message time is launch-bound, not copy-bound.
2. **The measured win is large-message bandwidth**: +33% busbw at 16 MB (247 -> 329 GB/s) and 1.7x lower latency at 4 MB (43 -> 26 us). Symmetric (window) registration lets NCCL use the zero-copy NVLink path instead of staging through intermediate buffers - a bandwidth optimization, not a latency one.
3. **Newer NCCL is not automatically faster**: 2.29.2's eager floor (25.1 us) is ~2 us *worse* than 2.18.3's (23.3 us) on identical hardware - version upgrades need re-measurement, not assumption.
4. **The ~23 us eager launch floor survives everything** - version upgrades and buffer registration alike. This strengthens the repo's central finding: the floor is host-side launch overhead, and only CUDA-Graph capture (which removes the launches) breaks it. The TP-decode comms ceiling (271/456 tok/s for Llama-70B TP=4) therefore stands unchanged.

**Methodology note:** the committed CUDA-Graph floor (13.7 us) was measured by this repo's own `tp_latency` harness (graph capture around the collective); the new graph-mode numbers here use `nccl-tests -G 1` (a different graph-capture implementation). The eager-vs-eager comparison is tool-identical; the graph-vs-graph one is indicative only.

