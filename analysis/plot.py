#!/usr/bin/env python3
"""Plot busbw vs message size for each collective CSV in results/."""
import csv, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
for csvf in sorted(glob.glob("results/*.csv")):
    xs, ys = [], []
    for r in csv.DictReader(open(csvf)):
        xs.append(int(r["size_bytes"])); ys.append(float(r["busbw_GBs"]))
    if xs:
        plt.plot(xs, ys, marker=".", label=os.path.basename(csvf).replace(".csv", ""))
plt.xscale("log", base=2); plt.xlabel("message size (bytes)"); plt.ylabel("bus bandwidth (GB/s)")
plt.title("NCCL collectives — 4×H100 NVLink"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("results/busbw.png", dpi=130)
print("wrote results/busbw.png")
