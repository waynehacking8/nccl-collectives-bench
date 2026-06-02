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
- **Full-node (8-GPU) NVLS reference reproduction (~480 GB/s, nccl-tests #312).** GPU 0 on this
  host is permanently reserved for a production service, so the full 8-GPU set is never
  available; a 7-GPU partial run would not match the published reference (different busbw
  factor, different NVLS tree) and was judged not worth the ambiguity. All measurements in
  this repo cap at 6 GPUs, stated explicitly.

## Phase 4 — Literature-ceiling reproductions on this box (specified)

Goal: turn published NCCL/NVSwitch reference numbers into measured rows from this exact host.

- [x] **NCCL ≥2.27 symmetric memory vs the measured latency floors (reference: up to 9× lower
  small-message latency).**
  **DONE (published claim does NOT reproduce here — better attribution found) — README
  "symmetric memory" section / `results/symmetric/report.md` / `results/symmetric_latency.png`.**
  Measured with NCCL 2.29.2 (pytorch:26.02 container), same 4-GPU slice, eager + CUDA Graph,
  `-R 2` vs unregistered. Result: small-message latency is **neutral** (floor 23.6 vs 25.1 µs;
  the 9× claim targets multi-node/NVLS proxy paths, not launch-bound NVSwitch all_reduce);
  the real measured gain is **large-message bandwidth: +33% busbw** (247→329 GB/s @ 16 MB,
  1.7× lower latency @ 4 MB) via the zero-copy NVLink path. Bonus finding: NCCL 2.29's eager
  floor is ~2 µs worse than 2.18.3's. The ~23 µs launch floor survives everything except CUDA
  Graphs → the TP-decode ceiling (271/456 tok/s, Llama-70B TP=4) stands unchanged.
  Sweep script: `scripts/run_symmetric_sweep.sh`; analysis: `analysis/symmetric_compare.py`.
  - **Question:** the repo's floors are 23.1 µs (eager) / 13.7 µs (CUDA Graph). NCCL 2.27's
    user-buffer registration / symmetric memory claims large small-message gains — how much of
    the floor is recoverable on this box without root?
  - **Method:** build NCCL ≥2.27 + matching nccl-tests in user space; register buffers
    (`ncclMemAlloc` + `ncclCommRegister`, `all_reduce_perf -R 2`); re-run the small-message
    sweep on the same 4-GPU slice as the published floors (GPU 0 is reserved for production;
    all measurements in this repo cap at 6 GPUs).
  - **Read-out:** symmetric vs non-symmetric latency/busbw-vs-size overlay; the new floor
    re-prices the TP-decode ceiling (271/456 tok/s for Llama-70B TP=4) — if the floor halves,
    the comms-bound ceiling roughly doubles.

- [x] **Scaling-study algorithm attribution (busbw vs physical link traffic).** The 2/4/6-GPU
  scaling runs used NCCL's automatic algorithm selection without `NCCL_DEBUG` capture, so the
  6-GPU 443 GB/s busbw cannot be attributed: Ring would mean 93% physical link utilization (an
  unexplained jump from the 4-GPU 76%), NVLS would mean ~56% (busbw inflated by a ring factor
  that does not describe in-switch-reduction traffic). Re-run `-g 2/4/6` with
  `NCCL_DEBUG=INFO,TUNING` and `NCCL_ALGO` pinned to Ring and NVLS separately; report physical
  per-link traffic per algorithm. See `results/scaling_report.md` for the full decomposition.
  **DONE (measured: Ring at 2/4 GPUs, NVLS at 6 — the 443 is not physical link traffic) —
  `results/scaling_attributed/report.md` / `results/scaling_attributed/attribution.png`.**
  - **Result:** 9 arms (`-g 2/4/6` × {auto, `NCCL_ALGO=Ring`, `NCCL_ALGO=NVLS`}, NCCL 2.29.2,
    pytorch:26.02 container, host GPUs 2..7, `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING`
    captured in every per-arm log). The tuner picks **Ring at 2 and 4 GPUs** and **NVLS at
    6 GPUs** (`g6_auto.log`: `AllReduce: ... Bytes -> Algo NVLS proto SIMPLE`), uniformly
    across all sizes 256 MB..8 GB. The committed 6-GPU 443 GB/s is therefore NVLS traffic: the
    links physically carry ~algbw = 266 GB/s = **56% of the 478 GB/s budget**, not 93%. Ring's
    physical link efficiency is flat across N (348 = 73% → 365 = 76% → 367 = 77%) — the
    apparent jump was the ring normalization applied to non-ring traffic. This run reproduces
    the committed peaks (347/365/443 GB/s, NCCL 2.18.3 bare-metal) within 0.5%, so the
    attribution transfers to the committed scaling study. Bonus findings: the tuner leaves 3%
    on the table at 4 GPUs (picks Ring 365 over available NVLS 376); pinned NVLS at 2 GPUs
    collapses to 207 GB/s (40% slower than Ring); NVLS's real 6-GPU win is **+21% end-to-end
    algbw** (220 → 266 GB/s vs pinned Ring), achieved by removing traffic from the links, not
    by saturating them. Sweep: `scripts/run_scaling_attributed.sh`; analysis + chart:
    `analysis/scaling_attribution.py`.

- [ ] **Candidates (spec on demand):** MSCCL++ vs NCCL (reference 3.8× small / 2.2× large on
  8×H100, arXiv:2504.09014); SM cost of collectives (NCCL 2.27 SHARP offload: 16→6 SMs) —
  measures the compute *stolen* by communication, a dimension this repo has not touched.
