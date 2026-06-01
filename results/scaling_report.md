## All-reduce busbw scaling with GPU count

NVSwitch fabric, all_reduce, peak busbw vs GPU count (NVLink budget 478 GB/s/GPU unidirectional):

| GPUs | peak busbw | % of NVLink budget | ring factor 2(N-1)/N |
|---|---|---|---|
| 2 | 347 GB/s | 73% | 1.00 |
| 4 | 365 GB/s | 76% | 1.50 |
| 6 | 443 GB/s | 93% | 1.67 |

Busbw climbs with N because higher GPU counts utilize the NVSwitch fabric more fully — more concurrent NVLink paths and better NVLS (in-switch reduction) efficiency keep the links saturated. (The ring factor 2(N-1)/N is *divided out* of algbw to define busbw, precisely so busbw is comparable across N; it is the divisor that produces busbw, not a mechanism that mechanically pushes it up.) The 4-GPU number quoted elsewhere in this repo is a mid-scale operating point, not the ceiling. The 6-GPU 93% should be read as an optimistic upper-bound framing rather than "near line-rate": the 478 GB/s budget is *unidirectional* per-GPU, while steady-state all-reduce traffic is simultaneously bidirectional, and the literature commonly reports ~75–85% of the unidirectional budget.

