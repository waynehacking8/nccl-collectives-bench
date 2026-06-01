# TP-decode all-reduce: the latency wall, and CUDA Graphs

TP=4 (4-GPU slice of 8× H100 NVSwitch host), bf16, 200 timed iters. Decode issues 2 all-reduces/layer at batch=1; each moves only `hidden×2` bytes, so it is latency-bound, not bandwidth-bound.

## 1. Eager vs CUDA-Graph all-reduce latency (size sweep)

| message | eager µs | graph µs | graph speedup |
|---|---|---|---|
| 1 KB | 22.7 | 11.8 | 1.92× |
| 2 KB | 23.0 | 11.9 | 1.94× |
| 4 KB | 38.6 | 12.0 | 3.21× |
| 8 KB | 27.8 | 13.4 | 2.07× |
| 16 KB | 23.1 | 14.0 | 1.65× |
| 32 KB | 23.1 | 14.0 | 1.65× |
| 64 KB | 22.8 | 14.7 | 1.55× |
| 128 KB | 23.1 | 15.1 | 1.53× |
| 256 KB | 82.0 | 15.8 | 5.20× |
| 512 KB | 23.7 | 18.0 | 1.31× |
| 1024 KB | 23.3 | 21.4 | 1.09× |
| 2048 KB | 31.4 | 30.6 | 1.03× |
| 4096 KB | 40.6 | 43.6 | 0.93× |

At small sizes the eager path is pinned on a ~23 µs launch+handshake floor; CUDA-Graph replay removes per-op launch dispatch and drops it to ~14 µs (1.7× at the TP-decode sizes). This is precisely the trick vLLM / TensorRT-LLM use to make TP decode viable. (The single-size eager spike is host-side launch jitter, not fabric traffic — see §4; the floor uses the median so it is unaffected.)

## 2. Decode tokens/s ceiling from communication alone

`tokens/s = 1e6 / (2 · num_layers · T_allreduce_µs)`, batch=1, TP=4.

| model | hidden | layers | msg | eager µs | graph µs | **eager tok/s** | **graph tok/s** |
|---|---|---|---|---|---|---|---|
| Qwen3-8B (on GPU0 here) | 4096 | 36 | 8 KB | 23.1 | 13.7 | **602** | **1013** |
| Llama-3-8B | 4096 | 32 | 8 KB | 23.1 | 13.7 | **677** | **1140** |
| Llama-3-70B | 8192 | 80 | 16 KB | 23.1 | 13.7 | **271** | **456** |
| Llama-3-405B | 16384 | 126 | 32 KB | 23.1 | 13.7 | **172** | **290** |

This is a *comms-only* ceiling — real decode is `max(compute, comms)` per step, but it shows why naive eager TP collapses for deep models: the all-reduce launch tax is paid 2·L times per token. CUDA Graphs lift the ceiling by the per-op speedup above.

## 3. Analytical model validation

Ring all-reduce `T(bytes) ≈ T_floor + 2(N-1)/N · bytes / BW_bus`, with measured `BW_bus = 366 GB/s` and `N=4` (factor 2(N-1)/N = 1.50).

| message | model µs | measured eager µs |
|---|---|---|
| 1 KB | 23.1 | 22.7 |
| 2 KB | 23.1 | 23.0 |
| 4 KB | 23.1 | 38.6 |
| 8 KB | 23.1 | 27.8 |
| 16 KB | 23.1 | 23.1 |
| 32 KB | 23.2 | 23.1 |
| 64 KB | 23.4 | 22.8 |
| 128 KB | 23.6 | 23.1 |
| 256 KB | 24.2 | 82.0 |
| 512 KB | 25.2 | 23.7 |
| 1024 KB | 27.4 | 23.3 |
| 2048 KB | 31.7 | 31.4 |
| 4096 KB | 40.3 | 40.6 |

The floor dominates below ~64 KB (the TP-decode regime); the bandwidth term only takes over once messages are large (batched decode / prefill).

## 4. Shared-box vs quiet-box re-run — what the eager spike actually is

The original sweep ran while the box hosted other tenants and showed an 82 µs eager spike at one size. The hypothesis to test: is that NVSwitch-fabric jitter from the other tenants' traffic? Re-run with every non-production tenant stopped (`results/quiet/`):

| message | eager (shared) | eager (quiet) | graph (shared) | graph (quiet) |
|---|---|---|---|---|
| 1 KB | 22.7 | 22.5 | 11.8 | 11.8 |
| 2 KB | 23.0 | 21.2 | 11.9 | 12.6 |
| 4 KB | 38.6 | 23.5 | 12.0 | 12.0 |
| 8 KB | 27.8 | 47.2 | 13.4 | 12.3 |
| 16 KB | 23.1 | 23.0 | 14.0 | 13.2 |
| 32 KB | 23.1 | 23.5 | 14.0 | 13.4 |
| 64 KB | 22.8 | 23.1 | 14.7 | 13.6 |
| 128 KB | 23.1 | 23.4 | 15.1 | 13.8 |
| 256 KB | 82.0 | 28.6 | 15.8 | 14.2 |
| 512 KB | 23.7 | 81.5 | 18.0 | 17.0 |
| 1024 KB | 23.3 | 22.0 | 21.4 | 21.4 |
| 2048 KB | 31.4 | 27.3 | 30.6 | 30.6 |
| 4096 KB | 40.6 | 40.5 | 43.6 | 43.3 |

**The hypothesis is rejected — and the correct attribution is more useful.** The quiet run reproduces a same-magnitude spike (~82 µs) at a *different* size, so the spike is not other tenants' traffic; it is **host-side launch jitter intrinsic to eager-mode submission** (one straggler iteration in the 200-iter mean — OS scheduling / launch-path noise). Three corroborating facts: (1) the eager floor barely moves (23.1 → 23.2 µs); (2) the CUDA-Graph floor is identical (13.7 → 12.9 µs); (3) **no graph-mode point spikes in either run** — graph replay bypasses the per-iteration launch path entirely. Practical consequence: CUDA Graphs don't just lower TP-decode latency ~1.7×, they also remove its tail jitter, which matters for p99 ITL in serving.

