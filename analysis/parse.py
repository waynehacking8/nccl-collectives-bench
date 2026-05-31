#!/usr/bin/env python3
"""Parse nccl-tests output into CSV (size_bytes, algbw, busbw, time_us)."""
import csv, re, sys, os
for path in sys.argv[1:]:
    rows = []
    for line in open(path):
        # nccl-tests data rows start with the message size in bytes
        m = re.match(r"\s*(\d+)\s+\d+\s+\S+\s+\S+\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            size, time_us, algbw, busbw = m.group(1), m.group(2), m.group(3), m.group(4)
            rows.append([size, time_us, algbw, busbw])
    out = path.replace(".txt", ".csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["size_bytes", "time_us", "algbw_GBs", "busbw_GBs"]); w.writerows(rows)
    print(f"{os.path.basename(path)}: {len(rows)} rows -> {out}")
