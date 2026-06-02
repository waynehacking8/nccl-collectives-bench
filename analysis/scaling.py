#!/usr/bin/env python3
"""All-reduce bandwidth scaling with GPU count on NVSwitch.

Parses results/scaling.txt (all_reduce_perf at -g 2/4/6, 256 MB .. 8 GB) and reports peak
busbw AND peak algbw per GPU count against the measured NVLink budget.

Interpretation note (the part that is easy to get wrong): busbw = algbw x 2(N-1)/N is
nccl-tests' RING-equivalent normalization — it equals physical per-link traffic only when
the algorithm actually moves ring-pattern traffic. These runs use NCCL's automatic algorithm
selection, so the busbw column does not have a single physical meaning across N. The
attribution run (analysis/scaling_attribution.py, results/scaling_attributed/) re-ran the
sweep with NCCL_DEBUG captured and NCCL_ALGO pinned per arm, and measured:
  - at 2 and 4 GPUs the tuner picks RING -> busbw there is a real per-link rate
    (347 = 73%, 365 = 76% of the 478 GB/s budget);
  - at 6 GPUs the tuner switches to NVLS (each GPU ships its data ONCE — in-switch
    reduction) -> physical per-link traffic is only ~algbw = 266 GB/s (56% of budget); most
    of the busbw rise is the ring factor applied to non-ring traffic.
"""
import re, os

NVLINK_UNI_GBS = 26.562 * 18  # 478.1 GB/s, measured per-link x links

# Tuner algorithm selection per GPU count, measured by the attribution run
# (results/scaling_attributed/, NCCL_DEBUG=INFO,TUNING + pinned-NCCL_ALGO arms).
ATTRIBUTED_ALGO = {2: "Ring", 4: "Ring", 6: "NVLS"}


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
         f"selection (NVLink budget {NVLINK_UNI_GBS:.0f} GB/s/GPU unidirectional). The "
         f"\"tuner algo\" column is measured by the attribution run "
         f"(`results/scaling_attributed/report.md`); physical per-link traffic is busbw for "
         f"Ring and ~algbw for NVLS:\n",
         "| GPUs | peak busbw | peak algbw | ring factor 2(N-1)/N | tuner algo (measured) | "
         "physical per-link | % of NVLink budget |",
         "|---|---|---|---|---|---|---|"]
    for n in sorted(peak):
        bus, alg = peak[n]
        f = 2 * (n - 1) / n
        algo = ATTRIBUTED_ALGO[n]
        phys = bus if algo == "Ring" else alg
        L.append(f"| {n} | {bus:.0f} GB/s | {alg:.0f} GB/s | {f:.2f} | {algo} | "
                 f"~{phys:.0f} GB/s | {100*phys/NVLINK_UNI_GBS:.0f}% |")
    L += ["",
          "**How to read this (and how not to).** busbw is nccl-tests' *ring-equivalent* "
          "normalization: busbw = algbw x 2(N-1)/N, the per-link traffic a ring algorithm "
          "would generate at the measured algbw. It equals physical per-link traffic only "
          "when the algorithm actually is Ring. These runs use NCCL's automatic algorithm "
          "selection, and the attribution run (`results/scaling_attributed/report.md`, "
          "`analysis/scaling_attribution.py`: same sweep with `NCCL_DEBUG=INFO,TUNING` "
          "captured and `NCCL_ALGO` pinned per arm) **measured** what the tuner picks at "
          "each N:",
          "",
          "- **2 GPUs** — tuner picks **Ring** (debug-log evidence; pinned NVLS collapses to "
          "207 GB/s here). 347 GB/s is a real per-link rate (73% of budget).",
          "- **4 GPUs** — tuner picks **Ring** (auto 365 == pinned Ring 365; pinned NVLS would "
          "be 376, consistent with `results/algo_sweep.txt`). busbw here is a real per-link "
          "rate (76% of budget).",
          "- **6 GPUs** — tuner switches to **NVLS** (auto 443 == pinned NVLS 443; pinned Ring "
          "manages only 367). NVLS busbw is inflated by a ring factor that does not describe "
          "its traffic — with in-switch reduction each GPU ships its data once, moving only "
          "~algbw = 266 GB/s, **56% of budget**. Most of the 365 -> 443 rise is the "
          "normalization formula plus the algorithm switch, not extra bytes on the links. "
          "Decomposition: busbw ratio 443/376 = 1.18 = factor ratio (1.67/1.50 = 1.11) x "
          "physical algbw gain (266/251 = 1.06).",
          "",
          "The attribution run reproduces these peaks within 0.5% under NCCL 2.29.2, so its "
          "tuner choices (Ring/Ring/NVLS) attribute the numbers in this table. The defensible "
          "statements:",
          "",
          "1. End-to-end all-reduce **algbw** — what a training step actually sees per byte of "
          "gradient — drops from 347 GB/s (2 GPUs) to 243/266 GB/s (4/6 GPUs), because each "
          "GPU must move 2(N-1)/N x more data as N grows. There is no free lunch in scaling "
          "out the ring.",
          "2. The 4-GPU ~366 GB/s busbw quoted elsewhere in this repo **is** a physical "
          "per-link rate, confirmed by the pinned-Ring sweep and now by the tuner's logged "
          "choice.",
          "3. The 6-GPU \"93% of budget\" figure is **not** physical link utilization — the "
          "measured algorithm is NVLS and the links carry ~56% of budget. Ring's physical "
          "link efficiency is flat across N (73% -> 76% -> 77%); the apparent jump was an "
          "artifact of the normalization.\n"]
    open("results/scaling_report.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plot(peak)

if __name__ == "__main__":
    main()
