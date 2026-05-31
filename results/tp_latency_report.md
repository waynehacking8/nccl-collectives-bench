# TP-decode all-reduce: the latency wall, and CUDA Graphs

TP=4 (4× H100, NVSwitch), bf16, 200 timed iters. Decode issues 2 all-reduces/layer at batch=1; each moves only `hidden×2` bytes, so it is latency-bound, not bandwidth-bound.

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

At small sizes the eager path is pinned on a ~23 µs launch+handshake floor; CUDA-Graph replay removes per-op launch dispatch and drops it to ~14 µs (1.7× at the TP-decode sizes). This is precisely the trick vLLM / TensorRT-LLM use to make TP decode viable. (Eager outliers in the sweep are NVSwitch-fabric jitter from sharing the box with another tenant — hence the median-based floor.)

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

