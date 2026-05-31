#!/usr/bin/env python3
"""Report peak busbw achieved vs NVLink theoretical budget.

H100 SXM: 18 NVLink4 links x 25 GB/s/link (each direction) = 450 GB/s per-GPU
unidirectional, 900 GB/s bidirectional. Edit if your box differs (PCIe / NVLink count).
"""
import csv, glob, os
H100_NVLINK_UNI_GBS = 450.0
for csvf in sorted(glob.glob("results/*.csv")):
    busbw = [float(r["busbw_GBs"]) for r in csv.DictReader(open(csvf))]
    if not busbw:
        continue
    peak = max(busbw)
    name = os.path.basename(csvf).replace(".csv", "")
    print(f"{name:16s} peak busbw {peak:7.1f} GB/s = {100*peak/H100_NVLINK_UNI_GBS:5.1f}% of NVLink uni budget")
