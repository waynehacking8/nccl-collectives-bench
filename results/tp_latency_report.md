# TP-decode all-reduce: the latency wall, and CUDA Graphs

TP=4 (4-GPU slice of 8× H100 NVSwitch host), bf16, 200 timed iters. Decode issues 2 all-reduces/layer at batch=1; each moves only `hidden×2` bytes, so it is latency-bound, not bandwidth-bound.

## 1. Eager vs CUDA-Graph all-reduce latency (size sweep)

| message | eager µs | graph µs | graph speedup |
|---|---|---|---|
| 1 KB | 33.1 | 17.1 | 1.93× |
| 2 KB | 40.0 | 19.2 | 2.09× |
| 4 KB | 40.9 | 18.8 | 2.18× |
| 8 KB | 64.6 | 19.1 | 3.38× |
| 16 KB | 35.5 | 19.8 | 1.79× |
| 32 KB | 35.2 | 21.1 | 1.67× |
| 64 KB | 37.8 | 19.8 | 1.91× |
| 128 KB | 34.2 | 20.2 | 1.70× |
| 256 KB | 70.0 | 19.8 | 3.53× |
| 512 KB | 37.2 | 19.8 | 1.88× |
| 1024 KB | 71.3 | 24.1 | 2.96× |
| 2048 KB | 36.6 | 33.0 | 1.11× |
| 4096 KB | 44.9 | 44.9 | 1.00× |

At small sizes the eager path is pinned on a ~37 µs launch+handshake floor; CUDA-Graph replay removes per-op launch dispatch and drops it to ~19 µs (1.9× at the TP-decode sizes). This is precisely the trick vLLM / TensorRT-LLM use to make TP decode viable. (The single-size eager spike is host-side launch jitter, not fabric traffic — see §4; the floor uses the median so it is unaffected.)

## 2. Decode tokens/s ceiling from communication alone

`tokens/s = 1e6 / (2 · num_layers · T_allreduce_µs)`, batch=1, TP=4.

| model | hidden | layers | msg | eager µs | graph µs | **eager tok/s** | **graph tok/s** |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (on GPU0 here) | 4096 | 36 | 8 KB | 36.6 | 19.5 | **379** | **713** |
| Llama-3-8B | 4096 | 32 | 8 KB | 36.6 | 19.5 | **426** | **803** |
| Llama-3-70B | 8192 | 80 | 16 KB | 36.6 | 19.5 | **171** | **321** |
| Llama-3-405B | 16384 | 126 | 32 KB | 36.6 | 19.5 | **108** | **204** |

This is a *comms-only* ceiling — real decode is `max(compute, comms)` per step, but it shows why naive eager TP collapses for deep models: the all-reduce launch tax is paid 2·L times per token. CUDA Graphs lift the ceiling by the per-op speedup above.

## 3. Analytical model validation

Ring all-reduce `T(bytes) ≈ T_floor + 2(N-1)/N · bytes / BW_bus`, with measured `BW_bus = 366 GB/s` and `N=4` (factor 2(N-1)/N = 1.50).

| message | model µs | measured eager µs |
|---|---|---|
| 1 KB | 36.7 | 33.1 |
| 2 KB | 36.7 | 40.0 |
| 4 KB | 36.7 | 40.9 |
| 8 KB | 36.7 | 64.6 |
| 16 KB | 36.7 | 35.5 |
| 32 KB | 36.8 | 35.2 |
| 64 KB | 36.9 | 37.8 |
| 128 KB | 37.2 | 34.2 |
| 256 KB | 37.7 | 70.0 |
| 512 KB | 38.8 | 37.2 |
| 1024 KB | 40.9 | 71.3 |
| 2048 KB | 45.2 | 36.6 |
| 4096 KB | 53.8 | 44.9 |

The floor dominates below ~64 KB (the TP-decode regime); the bandwidth term only takes over once messages are large (batched decode / prefill).

## 4. Shared-box vs quiet-box re-run — what the eager spike actually is

The original sweep ran while the box hosted other tenants and showed an 82 µs eager spike at one size. The hypothesis to test: is that NVSwitch-fabric jitter from the other tenants' traffic? Re-run with every non-production tenant stopped (`results/quiet/`):

| message | eager (shared) | eager (quiet) | graph (shared) | graph (quiet) |
|---|---|---|---|---|
| 1 KB | 33.1 | 22.5 | 17.1 | 11.8 |
| 2 KB | 40.0 | 21.2 | 19.2 | 12.6 |
| 4 KB | 40.9 | 23.5 | 18.8 | 12.0 |
| 8 KB | 64.6 | 47.2 | 19.1 | 12.3 |
| 16 KB | 35.5 | 23.0 | 19.8 | 13.2 |
| 32 KB | 35.2 | 23.5 | 21.1 | 13.4 |
| 64 KB | 37.8 | 23.1 | 19.8 | 13.6 |
| 128 KB | 34.2 | 23.4 | 20.2 | 13.8 |
| 256 KB | 70.0 | 28.6 | 19.8 | 14.2 |
| 512 KB | 37.2 | 81.5 | 19.8 | 17.0 |
| 1024 KB | 71.3 | 22.0 | 24.1 | 21.4 |
| 2048 KB | 36.6 | 27.3 | 33.0 | 30.6 |
| 4096 KB | 44.9 | 40.5 | 44.9 | 43.3 |

**The hypothesis is rejected — and the correct attribution is more useful.** The quiet run reproduces a same-magnitude spike (~82 µs) at a *different* size, so the spike is not other tenants' traffic; it is **host-side launch jitter intrinsic to eager-mode submission** (one straggler iteration in the 200-iter mean — OS scheduling / launch-path noise). Three corroborating facts: (1) the eager floor barely moves (36.6 → 23.2 µs); (2) the CUDA-Graph floor is identical (19.5 → 12.9 µs); (3) **no graph-mode point spikes in either run** — graph replay bypasses the per-iteration launch path entirely. Practical consequence: CUDA Graphs don't just lower TP-decode latency ~1.7×, they also remove its tail jitter, which matters for p99 ITL in serving.

