# Roadmap

## Phase 1 — Intra-node NVLink
- [ ] Build nccl-tests; verify topology with `nvidia-smi topo -m`.
- [ ] all_reduce / all_gather / reduce_scatter sweep, 8B..8GB, 4 GPUs.
- [ ] Parse -> CSV; plot busbw/algbw vs size; % of NVLink peak.

## Phase 2 — Interpretation
- [ ] Latency floor (small-size busbw) and steady-state bandwidth; mark the crossover.
- [ ] Tie to TP=4 LLM inference: estimate comms time per decode step at batch 1 vs 64.

## Phase 3 — Multi-node (when a second node + IB is available)
- [ ] Same sweep over InfiniBand; intra- vs inter-node bandwidth gap; SHARP if available.

## Out of scope
- Reimplementing collectives. (A hand-written ring all-reduce demo is a maybe, clearly labeled.)
