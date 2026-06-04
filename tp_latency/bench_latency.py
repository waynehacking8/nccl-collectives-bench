#!/usr/bin/env python3
"""TP-decode all-reduce latency: eager vs CUDA-Graph, at real LLM message sizes.

Why this exists
---------------
The busbw sweep in this repo measures *steady-state bandwidth* (8 MB+ messages).
LLM tensor-parallel **decode** lives in the opposite regime: each transformer layer
issues two all-reduces (post-attention, post-MLP), and at batch=1 each one moves only
`hidden_size * batch * 2` bytes — e.g. 8192*1*2 = 16 KB. At 16 KB an NCCL all-reduce is
nowhere near bandwidth-bound; it sits on the ~23 µs kernel-launch + handshake floor (measured).

A 70B-class model (80 layers) therefore pays 2*80 = 160 all-reduces per token. At ~23 µs
each that is ~3.7 ms/token of pure comms launch overhead -> a hard ceiling of ~271 tok/s (measured)
from communication alone, before any compute. The production fix is **CUDA Graphs**:
capture the whole decode step once and replay it, collapsing per-op launch latency. This
script measures exactly that gap, then the companion roofline.py turns it into a
tokens/s ceiling and validates an analytical model.

Run (4 ranks, pinned GPUs chosen by the launcher / docker --gpus):
    torchrun --nproc_per_node=4 tp_latency/bench_latency.py
"""
import json, math, os
import torch
import torch.distributed as dist

# (hidden_size, batch) decode points for common TP'd models, plus a pure size sweep.
LLM_POINTS = [
    ("h4096_b1", 4096, 1),     # 7-13B class, decode batch 1
    ("h4096_b8", 4096, 8),
    ("h5120_b1", 5120, 1),     # 13B
    ("h8192_b1", 8192, 1),     # 70B class, decode batch 1  -> 16 KB
    ("h8192_b16", 8192, 16),   # 70B, batched decode        -> 256 KB
    ("h16384_b1", 16384, 1),   # 400B+ class
    ("h16384_b32", 16384, 32),
]
SIZE_SWEEP_BYTES = [1 << k for k in range(10, 23)]  # 1 KB .. 4 MB
DTYPE = torch.bfloat16
ITERS = 200
WARMUP = 20


def time_eager(buf, iters):
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        dist.all_reduce(buf)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # us
    mean_t = sum(times) / len(times)
    std_t = math.sqrt(sum((t - mean_t) ** 2 for t in times) / len(times))
    return mean_t, std_t


def time_graph(buf, iters):
    # warm NCCL on a side stream before capture (required for graph-safe collectives)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            dist.all_reduce(buf)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        dist.all_reduce(buf)

    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3)  # us
    mean_t = sum(times) / len(times)
    std_t = math.sqrt(sum((t - mean_t) ** 2 for t in times) / len(times))
    return mean_t, std_t


def bench_one(nbytes, rank):
    n = max(1, nbytes // DTYPE.itemsize)
    buf = torch.ones(n, dtype=DTYPE, device="cuda")
    dist.barrier()
    eager, eager_std = time_eager(buf, ITERS)
    dist.barrier()
    graph, graph_std = time_graph(buf, ITERS)
    dist.barrier()
    del buf
    torch.cuda.empty_cache()
    return {"bytes": n * DTYPE.itemsize, "eager_us": eager, "eager_std_us": eager_std,
            "graph_us": graph, "graph_std_us": graph_std,
            "speedup": eager / graph if graph else 0.0}


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if rank == 0:
        print(f"world_size={world} dtype={DTYPE} iters={ITERS}", flush=True)

    results = {"world_size": world, "dtype": str(DTYPE), "iters": ITERS,
               "size_sweep": [], "llm_points": []}

    for nbytes in SIZE_SWEEP_BYTES:
        r = bench_one(nbytes, rank)
        if rank == 0:
            results["size_sweep"].append(r)
            print(f"  sweep {r['bytes']:>9d} B  eager {r['eager_us']:7.1f}±{r['eager_std_us']:.1f} us  "
                  f"graph {r['graph_us']:7.1f}±{r['graph_std_us']:.1f} us  x{r['speedup']:.2f}", flush=True)

    for name, hidden, batch in LLM_POINTS:
        nbytes = hidden * batch * DTYPE.itemsize
        r = bench_one(nbytes, rank)
        r.update({"name": name, "hidden": hidden, "batch": batch})
        if rank == 0:
            results["llm_points"].append(r)
            print(f"  {name:12s} {hidden}x{batch} {r['bytes']:>8d} B  "
                  f"eager {r['eager_us']:7.1f}±{r['eager_std_us']:.1f} us  "
                  f"graph {r['graph_us']:7.1f}±{r['graph_std_us']:.1f} us  "
                  f"x{r['speedup']:.2f}", flush=True)

    if rank == 0:
        os.makedirs("results", exist_ok=True)
        with open("results/tp_latency.json", "w") as f:
            json.dump(results, f, indent=2)
        print("wrote results/tp_latency.json", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
