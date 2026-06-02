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

## Phase 4 — Literature-ceiling reproductions on this box (specified)

Goal: turn published NCCL/NVSwitch reference numbers into measured rows from this exact host.

- [ ] **Full 8-GPU all-reduce: Ring vs NVLS (reference: ~370 vs ~480 GB/s busbw, nccl-tests #312).**
  - **Question:** this repo's measurements stop at 6 GPUs (443 GB/s). On the full 8-GPU host,
    does NVLS reach the ~480 GB/s reference — and how should the fact that 480 exceeds the
    450 GB/s per-GPU line rate be explained (in-switch reduction makes the busbw formula
    over-credit NVLS)?
  - **Method:** requires all 8 GPUs idle (same quiet-window arrangement as the Phase 2 re-run):
    `all_reduce_perf -b 8 -e 8G -f 2 -g 8` with `NCCL_NVLS_ENABLE=1` vs `=0`.
  - **Read-out:** Ring vs NVLS busbw curves at 8 GPUs; report both as "% of 478 GB/s
    unidirectional budget" and "% of the 900 GB/s bidirectional / 3.6 TB/s bisection
    architecture ceiling"; explain the NVLS over-credit explicitly. Completes the repo's
    2/4/6/8 scaling story.

- [ ] **NCCL ≥2.27 symmetric memory vs the measured latency floors (reference: up to 9× lower
  small-message latency).**
  - **Question:** the repo's floors are 23.1 µs (eager) / 13.7 µs (CUDA Graph). NCCL 2.27's
    user-buffer registration / symmetric memory claims large small-message gains — how much of
    the floor is recoverable on this box without root?
  - **Method:** build NCCL ≥2.27 + matching nccl-tests in user space; register buffers
    (`ncclMemAlloc` + `ncclCommRegister`, `all_reduce_perf -R 2`); re-run the small-message
    sweep on the 4-GPU slice and (when idle) the full 8 GPUs.
  - **Read-out:** symmetric vs non-symmetric latency/busbw-vs-size overlay; the new floor
    re-prices the TP-decode ceiling (271/456 tok/s for Llama-70B TP=4) — if the floor halves,
    the comms-bound ceiling roughly doubles.

- [ ] **Candidates (spec on demand):** MSCCL++ vs NCCL (reference 3.8× small / 2.2× large on
  8×H100, arXiv:2504.09014); SM cost of collectives (NCCL 2.27 SHARP offload: 16→6 SMs) —
  measures the compute *stolen* by communication, a dimension this repo has not touched.
