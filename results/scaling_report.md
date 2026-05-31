## All-reduce busbw scaling with GPU count

NVSwitch fabric, all_reduce, peak busbw vs GPU count (NVLink budget 478 GB/s/GPU unidirectional):

| GPUs | peak busbw | % of NVLink budget | ring factor 2(N-1)/N |
|---|---|---|---|
| 2 | 347 GB/s | 73% | 1.00 |
| 4 | 365 GB/s | 76% | 1.50 |
| 6 | 443 GB/s | 93% | 1.67 |

Busbw climbs with N because the ring all-reduce moves 2(N-1)/N of the buffer per GPU — the factor rises toward 2 as N grows, and NVLS (in-switch reduction) keeps the link saturated. The 4-GPU number quoted elsewhere in this repo is a mid-scale operating point, not the ceiling; at 6 GPUs the fabric runs near the NVLink budget.

