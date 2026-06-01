# Roadmap

## Phase 1 — Intra-node NVLink
- [ ] Build nccl-tests; verify topology with `nvidia-smi topo -m`.
- [ ] all_reduce / all_gather / reduce_scatter sweep, 8B..8GB, 4 GPUs.
- [ ] Parse -> CSV; plot busbw/algbw vs size; % of NVLink peak.

## Phase 2 — Interpretation
- [ ] Latency floor (small-size busbw) and steady-state bandwidth; mark the crossover.
- [ ] Tie to TP=4 LLM inference: estimate comms time per decode step at batch 1 vs 64.
- [x] Re-run the small-message latency sweep on a quiet box (every non-production tenant stopped).
  - **Question:** the ~23 µs eager floor and the 82 µs outlier @256KB were measured on a shared
    box (fabric jitter from other tenants' traffic). Is the floor lower / does the outlier
    disappear with exclusive access?
  - **Method:** all benchmark tenants stopped (only the box's production single-GPU service, which
    generates no NVLink traffic, remained); sweep + TP latency bench re-run → `results/quiet/`.
  - **Result (hypothesis rejected — better attribution found):** the outlier did NOT disappear; a
    same-magnitude spike (~82 µs) appeared at a *different* size. The eager floor is unchanged
    (23.1 → 23.2 µs median), the graph floor unchanged (13.7 → 12.9 µs), and the bandwidth sweep
    matches within ±4%. So the spike is **host-side launch jitter intrinsic to eager mode**, not
    fabric traffic — and it never appears in CUDA-Graph mode in either run. The TP-decode comms
    ceiling (271/456 tok/s for Llama-70B TP=4) stands as published. See
    `results/tp_latency_report.md` §4.

## Phase 3 — Multi-node (when a second node + IB is available)
- [ ] Same sweep over InfiniBand; intra- vs inter-node bandwidth gap; SHARP if available.

## Out of scope
- Reimplementing collectives. (A hand-written ring all-reduce demo is a maybe, clearly labeled.)
