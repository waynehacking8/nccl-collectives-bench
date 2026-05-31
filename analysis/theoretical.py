#!/usr/bin/env python3
"""Report measured bus bandwidth vs the H100 NVLink budget.

NVLink budget is taken from the *measured* per-link rate reported by
`nvidia-smi nvlink --status` on this box, not a datasheet round number:

    H100 SXM5 : 18 NVLink-4 links/GPU, measured 26.562 GB/s/link/direction
              => 18 * 26.562 = 478.1 GB/s per-GPU unidirectional
              => ~956 GB/s bidirectional

For a ring all-reduce the bus-bandwidth ceiling is the per-GPU unidirectional
link budget (each GPU drives one ring egress link at steady state), so we
report measured peak busbw as a fraction of that 478 GB/s.
"""
import csv, glob, os

NVLINK_PER_LINK_GBS = 26.562   # measured, nvidia-smi nvlink --status
NVLINK_LINKS = 18
H100_NVLINK_UNI_GBS = NVLINK_PER_LINK_GBS * NVLINK_LINKS  # 478.1 GB/s

def peak(csvf):
    rows = list(csv.DictReader(open(csvf)))
    bus = [float(r["busbw_GBs"]) for r in rows if r.get("busbw_GBs")]
    return max(bus) if bus else 0.0

def main():
    print(f"NVLink budget: {NVLINK_LINKS} links x {NVLINK_PER_LINK_GBS} GB/s "
          f"= {H100_NVLINK_UNI_GBS:.1f} GB/s per-GPU unidirectional\n")
    print(f"{'collective':18s} {'peak busbw':>12s} {'% of NVLink uni':>16s}")
    print("-" * 48)
    for csvf in sorted(glob.glob("results/*.csv")):
        name = os.path.basename(csvf).replace(".csv", "")
        if name in ("algo_sweep", "proto_sweep"):
            continue
        p = peak(csvf)
        if p:
            print(f"{name:18s} {p:9.1f} GB/s {100*p/H100_NVLINK_UNI_GBS:14.1f}%")

if __name__ == "__main__":
    main()
