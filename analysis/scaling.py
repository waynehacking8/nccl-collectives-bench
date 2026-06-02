#!/usr/bin/env python3
"""All-reduce bandwidth scaling with GPU count on NVSwitch.

Parses results/scaling.txt (all_reduce_perf at -g 2/4/6, 256 MB .. 8 GB) and reports peak
busbw AND peak algbw per GPU count against the measured NVLink budget.

Interpretation note (the part that is easy to get wrong): busbw = algbw x 2(N-1)/N is
nccl-tests' RING-equivalent normalization — it equals physical per-link traffic only when
the algorithm actually moves ring-pattern traffic. These runs use NCCL's automatic algorithm
selection (no NCCL_ALGO pin, NCCL_DEBUG not captured), so the busbw column does not have a
single physical meaning across N:
  - at 4 GPUs the auto result matches the pinned-Ring sweep (366, results/algo_sweep.txt),
    not pinned NVLS (376) -> busbw there is a real per-link rate;
  - at 6 GPUs the 443 busbw is ambiguous: Ring would imply an unexplained jump in link
    efficiency (77% -> 93%), while NVLS (each GPU ships its data ONCE — in-switch reduction)
    implies physical per-link traffic of only ~algbw = 266 GB/s (56% of budget), with most of
    the busbw rise being the ring factor applied to non-ring traffic.
The committed logs cannot distinguish the two; see the report text and the roadmap item.
"""
import re, os

NVLINK_UNI_GBS = 26.562 * 18  # 478.1 GB/s, measured per-link x links


def parse_peak(path):
    """Return {gpu_count: (peak busbw, algbw at that row)} parsed from scaling.txt."""
    peak, cur = {}, None
    for line in open(path):
        m = re.search(r"GPUS=(\d+)", line)
        if m:
            cur = int(m.group(1)); peak[cur] = (0.0, 0.0); continue
        m = re.match(r"\s*\d+\s+\d+\s+\S+\s+\S+\s+\S+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)", line)
        if m and cur:
            algbw, busbw = float(m.group(1)), float(m.group(2))
            if busbw > peak[cur][0]:
                peak[cur] = (busbw, algbw)
    return peak


def plot(peak):
    """Emit results/scaling_busbw.png — peak busbw and algbw vs GPU count."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = sorted(peak)
    bus = [peak[n][0] for n in ns]
    alg = [peak[n][1] for n in ns]
    plt.figure(figsize=(8, 5))
    plt.plot(ns, bus, marker="o", color="tab:orange",
             label="all_reduce peak busbw (ring-equivalent)")
    plt.plot(ns, alg, marker="s", color="tab:blue", label="all_reduce peak algbw")
    plt.axhline(NVLINK_UNI_GBS, color="gray", linestyle="--", alpha=0.7,
                label=f"NVLink uni budget ({NVLINK_UNI_GBS:.0f} GB/s)")
    for n, y in zip(ns, bus):
        plt.annotate(f"{y:.0f}", (n, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)
    for n, y in zip(ns, alg):
        plt.annotate(f"{y:.0f}", (n, y), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=9)
    plt.xticks(ns)
    plt.ylim(0, NVLINK_UNI_GBS * 1.08)
    plt.xlabel("GPU count"); plt.ylabel("peak bandwidth (GB/s)")
    plt.title("All-reduce bandwidth scaling with GPU count — H100 NVSwitch")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("results/scaling_busbw.png", dpi=130)
    print("wrote results/scaling_busbw.png")


def main():
    path = "results/scaling.txt"
    if not os.path.exists(path):
        print("no results/scaling.txt"); return
    peak = parse_peak(path)

    L = ["## All-reduce bandwidth scaling with GPU count\n",
         f"NVSwitch fabric, all_reduce, peak bandwidth vs GPU count, NCCL automatic algorithm "
         f"selection (NVLink budget {NVLINK_UNI_GBS:.0f} GB/s/GPU unidirectional):\n",
         "| GPUs | peak busbw | peak algbw | ring factor 2(N-1)/N | busbw as % of NVLink budget |",
         "|---|---|---|---|---|"]
    for n in sorted(peak):
        bus, alg = peak[n]
        f = 2 * (n - 1) / n
        L.append(f"| {n} | {bus:.0f} GB/s | {alg:.0f} GB/s | {f:.2f} | "
                 f"{100*bus/NVLINK_UNI_GBS:.0f}% |")
    L += ["",
          "**How to read this (and how not to).** busbw is nccl-tests' *ring-equivalent* "
          "normalization: busbw = algbw x 2(N-1)/N, the per-link traffic a ring algorithm "
          "would generate at the measured algbw. It equals physical per-link traffic only "
          "when the algorithm actually is Ring. These runs use NCCL's automatic algorithm "
          "selection (no `NCCL_ALGO` pin; `NCCL_DEBUG` output was not captured), which changes "
          "what the busbw column means at each N:",
          "",
          "- **2 GPUs** — Ring and NVLS traffic patterns coincide (each GPU sends and receives "
          "the full buffer once), so 347 GB/s is a real per-link rate (73% of budget).",
          "- **4 GPUs** — the auto-selected result (365) matches the pinned-**Ring** row of the "
          "algorithm sweep (366, `results/algo_sweep.txt`), not pinned NVLS (376). busbw here "
          "is a real per-link rate (76% of budget).",
          "- **6 GPUs** — the 443 busbw is **ambiguous**. If the tuner stayed on Ring, it is a "
          "physical 93% of the link budget — a 17-point jump in link efficiency over the 4-GPU "
          "point with no mechanism to explain it. If the tuner switched to NVLS at the higher "
          "rank count (the parsimonious reading: pinned NVLS already beats Ring at 4 GPUs, and "
          "NVLS busbw is inflated by a ring factor that does not describe its traffic — with "
          "in-switch reduction each GPU ships its data once, moving only ~algbw = 266 GB/s, "
          "56% of budget), then most of the 365 -> 443 rise is the normalization formula, not "
          "extra bytes on the links. Decomposition under the NVLS reading: busbw ratio 443/376 "
          "= 1.18 = factor ratio (1.67/1.50 = 1.11) x physical algbw gain (266/251 = 1.06).",
          "",
          "The committed logs cannot distinguish the two readings (no `NCCL_DEBUG` output); "
          "resolving this is a roadmap item (re-run with `NCCL_DEBUG=INFO,TUNING` and "
          "`NCCL_ALGO` pinned to Ring and NVLS separately). Until then, the defensible "
          "statements are:",
          "",
          "1. End-to-end all-reduce **algbw** — what a training step actually sees per byte of "
          "gradient — drops from 347 GB/s (2 GPUs) to 243/266 GB/s (4/6 GPUs), because each "
          "GPU must move 2(N-1)/N x more data as N grows. There is no free lunch in scaling "
          "out the ring.",
          "2. The 4-GPU ~366 GB/s busbw quoted elsewhere in this repo **is** a physical "
          "per-link rate, confirmed independently by the pinned-Ring sweep.",
          "3. The 6-GPU \"93% of budget\" figure should **not** be quoted as physical link "
          "utilization — under the more likely NVLS reading the links carry ~56% of budget.\n"]
    open("results/scaling_report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plot(peak)

if __name__ == "__main__":
    main()
