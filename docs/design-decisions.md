# Design decisions

**D1 — Report bus bandwidth (busbw), not just algorithmic bandwidth (algbw).** `nccl-tests`
reports both; busbw normalizes by the data each GPU actually moves on the ring, so it is the
number you compare to the hardware link. Comparing algbw across collectives is misleading.

**D2 — All-reduce is the headline.** Data-parallel gradient sync and tensor-parallel
activation reduction are dominated by all-reduce. We still sweep all-gather / reduce-scatter
because ZeRO/FSDP and sequence/tensor parallel use them.

**D3 — Sweep sizes from 8 B to 8 GB.** Small sizes expose the latency floor (kernel launch +
handshake); large sizes expose steady-state bandwidth. The crossover is the interesting part
and explains why small-batch TP inference is comms-bound.

**D4 — Compare to a stated theoretical budget.** `analysis/theoretical.py` encodes the NVLink
per-GPU bidirectional budget for H100 so every measured number is reported as "X GB/s = Y% of
peak", not a naked figure.
