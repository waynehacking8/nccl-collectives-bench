#!/usr/bin/env python3
"""Scaling-study algorithm attribution: which algorithm does the tuner actually pick?

Parses results/scaling_attributed/g{2,4,6}_{auto,Ring,NVLS}.log (all_reduce_perf, 256 MB..8 GB,
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING captured in the same file) and answers the
question the committed scaling study could not: is the 6-GPU 443 GB/s busbw Ring traffic
(physical per-link rate) or NVLS traffic (busbw inflated by the ring normalization)?

Physical per-link traffic per algorithm:
  Ring  -> busbw   (the ring-equivalent normalization is exact for actual ring traffic)
  NVLS  -> ~algbw  (in-switch reduction: each GPU ships its data once and receives the
                    result once, so the links carry ~algbw, not busbw)

Writes results/scaling_attributed/report.md and results/scaling_attributed/attribution.png.
"""
import re, os, glob

NVLINK_UNI_GBS = 26.562 * 18  # 478.1 GB/s, measured per-link x links
DIR = "results/scaling_attributed"
ARMS = ["auto", "Ring", "NVLS"]
NS = [2, 4, 6]


def parse_arm(path):
    """Parse one arm log -> dict with per-size rows, peaks, and the tuner's algorithm choice.

    NCCL debug lines (NCCL_DEBUG=INFO goes to stdout) interleave mid-row with the test
    output, so first strip every debug segment (hostname:pid:tid [rank] NCCL INFO ... \\n)
    from the byte stream, which re-assembles the original test rows.
    """
    content = open(path).read()
    algos = sorted(set(re.findall(r"AllReduce: \d+ Bytes -> Algo (\w+) proto", content)))
    channels = sorted(set(re.findall(r"-> Algo \w+ proto \w+ channel\{Lo\.\.Hi\}=\{(\d+\.\.\d+)\}",
                                     content)))
    clean = re.sub(r"\S+:\d+:\d+ \[\d+\] NCCL INFO [^\n]*\n?", "", content)
    rows = []
    for line in clean.split("\n"):
        if not re.match(r"\s+\d+\s+\d+\s+float", line):
            continue
        toks = line.split()
        # after stripping, the root (-1) / #wrong (0) fields glued to debug text may be gone;
        # numeric fields left are: oop time/algbw/busbw[/wrong], ip time/algbw/busbw/wrong
        nums = [float(t) for t in toks[4:] if t != "-1"]
        rows.append({"size": int(toks[0]), "algbw": nums[1], "busbw": nums[2]})
    peak = max(rows, key=lambda r: r["busbw"])
    return {"rows": rows, "peak_busbw": peak["busbw"], "peak_algbw": peak["algbw"],
            "algos": algos, "channels": channels}


def physical_per_link(arm):
    """Physical per-link traffic for an arm: Ring -> busbw, NVLS -> ~algbw."""
    algo = arm["algos"][0] if len(arm["algos"]) == 1 else None
    return arm["peak_busbw"] if algo == "RING" else arm["peak_algbw"]


def load():
    data = {}
    for n in NS:
        for a in ARMS:
            data[(n, a)] = parse_arm(f"{DIR}/g{n}_{a}.log")
    return data


def plot(data):
    """Emit attribution.png: reported busbw vs physical per-link traffic, per arm."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    colors = {"auto": "tab:green", "Ring": "tab:blue", "NVLS": "tab:orange"}
    markers = {"auto": "o", "Ring": "s", "NVLS": "^"}
    styles = {"auto": "-", "Ring": "--", "NVLS": "--"}

    for ax, metric, title in [
            (axes[0], "peak_busbw", "What nccl-tests reports: busbw (ring-equivalent)"),
            (axes[1], None, "What the links carry: physical per-link traffic")]:
        for a in ARMS:
            ys = [data[(n, a)][metric] if metric else physical_per_link(data[(n, a)])
                  for n in NS]
            label = {"auto": "auto (tuner)", "Ring": "NCCL_ALGO=Ring",
                     "NVLS": "NCCL_ALGO=NVLS"}[a]
            ax.plot(NS, ys, marker=markers[a], linestyle=styles[a], color=colors[a],
                    label=label, markersize=7, linewidth=2 if a == "auto" else 1.5)
        ax.axhline(NVLINK_UNI_GBS, color="gray", linestyle=":",
                   label=f"NVLink uni budget ({NVLINK_UNI_GBS:.0f} GB/s)")
        ax.set_xticks(NS)
        ax.set_xlabel("GPU count")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 560)
    axes[0].set_ylabel("GB/s")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("All-reduce algorithm attribution — H100 NVSwitch, NCCL 2.29.2", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(f"{DIR}/attribution.png", dpi=130, bbox_inches="tight")
    print(f"wrote {DIR}/attribution.png")


def main():
    if not glob.glob(f"{DIR}/g*_auto.log"):
        print(f"no logs under {DIR}"); return
    data = load()
    ncclver = open(f"{DIR}/nccl_version.txt").read().strip()

    L = ["# Scaling-study algorithm attribution",
         "",
         "**Question (from `results/scaling_report.md`):** the committed 2/4/6-GPU scaling runs "
         "(`results/scaling.txt`, NCCL 2.18.3, automatic algorithm selection, no `NCCL_DEBUG` "
         "capture) report a 6-GPU peak busbw of 443 GB/s. Is that Ring traffic — a physical 93% "
         "of the 478 GB/s per-GPU NVLink budget — or NVLS traffic, where busbw is inflated by a "
         "ring-factor normalization that does not describe in-switch-reduction traffic?",
         "",
         f"**Setup:** all_reduce_perf, 256 MB..8 GB, `-f 2 -w 5 -n 50`, NCCL {ncclver.split(': ')[1]} "
         "(pytorch:26.02 container), host GPUs 2..7 (GPU 0 = production, never used; box state in "
         "`boxstate.csv` / `other_tenants.csv`). 9 arms: `-g 2/4/6` x {auto, `NCCL_ALGO=Ring`, "
         "`NCCL_ALGO=NVLS`}, each with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING` captured "
         "in the same per-arm log (`g{N}_{arm}.log`).",
         "",
         "## 1. What the tuner actually picks (from the NCCL debug log)",
         "",
         "| GPUs | auto selection (all 6 sizes) | channels | evidence |",
         "|---|---|---|---|"]
    for n in NS:
        arm = data[(n, "auto")]
        algo = arm["algos"][0]
        chan = arm["channels"][0]
        L.append(f"| {n} | **{algo}** (proto SIMPLE) | {chan} | "
                 f"`g{n}_auto.log`: `AllReduce: ... Bytes -> Algo {algo} proto SIMPLE` |")
    L += ["",
          "The tuner picks **Ring at 2 and 4 GPUs** and switches to **NVLS at 6 GPUs**. The "
          "selection is uniform across all message sizes (256 MB..8 GB) in every arm.",
          "",
          "## 2. All nine arms: reported vs physical bandwidth",
          "",
          "busbw is nccl-tests' ring-equivalent normalization (algbw x 2(N-1)/N). Physical "
          "per-link traffic: Ring -> busbw (the normalization is exact for ring traffic); "
          "NVLS -> ~algbw (in-switch reduction ships each GPU's data once).",
          "",
          "| GPUs | arm | algo used | peak busbw | peak algbw | physical per-link | % of "
          f"{NVLINK_UNI_GBS:.0f} GB/s budget |",
          "|---|---|---|---|---|---|---|"]
    for n in NS:
        for a in ARMS:
            arm = data[(n, a)]
            algo = arm["algos"][0]
            phys = physical_per_link(arm)
            name = {"auto": "auto", "Ring": "`NCCL_ALGO=Ring`", "NVLS": "`NCCL_ALGO=NVLS`"}[a]
            L.append(f"| {n} | {name} | {algo} | {arm['peak_busbw']:.0f} GB/s | "
                     f"{arm['peak_algbw']:.0f} GB/s | ~{phys:.0f} GB/s | "
                     f"{100 * phys / NVLINK_UNI_GBS:.0f}% |")
    a6, r6, v6 = (data[(6, a)] for a in ARMS)
    a4, r4, v4 = (data[(4, a)] for a in ARMS)
    a2, r2, v2 = (data[(2, a)] for a in ARMS)
    L += ["",
          "## 3. Findings",
          "",
          f"1. **The 6-GPU 443 GB/s is NVLS, and it is not physical link traffic.** The tuner "
          f"selects NVLS at 6 GPUs (auto {a6['peak_busbw']:.0f} == pinned NVLS "
          f"{v6['peak_busbw']:.0f} GB/s; pinned Ring manages only {r6['peak_busbw']:.0f}). With "
          f"in-switch reduction the links physically carry ~algbw = {a6['peak_algbw']:.0f} GB/s "
          f"= **{100 * a6['peak_algbw'] / NVLINK_UNI_GBS:.0f}% of budget**, not 93%. Most of the "
          f"4->6 GPU busbw rise ({a4['peak_busbw']:.0f} -> {a6['peak_busbw']:.0f}) is the ring "
          f"factor (1.50 -> 1.67) applied to non-ring traffic, plus the algorithm switch.",
          "",
          f"2. **Ring's physical link efficiency is flat across N — there never was a jump to "
          f"explain.** Pinned Ring busbw (= physical): {r2['peak_busbw']:.0f} "
          f"({100 * r2['peak_busbw'] / NVLINK_UNI_GBS:.0f}%) -> {r4['peak_busbw']:.0f} "
          f"({100 * r4['peak_busbw'] / NVLINK_UNI_GBS:.0f}%) -> {r6['peak_busbw']:.0f} GB/s "
          f"({100 * r6['peak_busbw'] / NVLINK_UNI_GBS:.0f}%) at 2/4/6 GPUs. The \"93% link "
          f"utilization\" reading of the committed 443 was an artifact of applying the ring "
          f"formula to NVLS traffic.",
          "",
          f"3. **The attribution transfers to the committed scaling study.** This run reproduces "
          f"the committed peaks (347/365/443 GB/s, NCCL 2.18.3 bare-metal) at "
          f"{a2['peak_busbw']:.0f}/{a4['peak_busbw']:.0f}/{a6['peak_busbw']:.0f} GB/s under "
          f"NCCL 2.29.2 in a container — within 0.5% at every point. The tuner's choices "
          f"measured here (Ring/Ring/NVLS) are therefore the choices behind the committed "
          f"numbers.",
          "",
          f"4. **What NVLS actually buys at 6 GPUs is +{100 * (a6['peak_algbw'] / r6['peak_algbw'] - 1):.0f}% "
          f"end-to-end bandwidth, not 93% link utilization.** algbw — what a training step sees "
          f"per byte of gradient — is {r6['peak_algbw']:.0f} GB/s under Ring vs "
          f"{v6['peak_algbw']:.0f} GB/s under NVLS. The win comes from *removing* traffic from "
          f"the links (in-switch reduction), not from saturating them.",
          "",
          f"5. **Tuner non-optimality, both directions.** At 4 GPUs the tuner picks Ring "
          f"({a4['peak_busbw']:.0f}) although pinned NVLS is "
          f"{100 * (v4['peak_busbw'] / a4['peak_busbw'] - 1):.0f}% faster "
          f"({v4['peak_busbw']:.0f}) — consistent with the repo's 4-GPU algorithm sweep "
          f"(`results/algo_sweep.txt`). At 2 GPUs pinning NVLS is a disaster: "
          f"{v2['peak_busbw']:.0f} vs {r2['peak_busbw']:.0f} GB/s "
          f"({100 * (1 - v2['peak_busbw'] / r2['peak_busbw']):.0f}% slower) — the multicast path "
          f"has per-message overhead that a 2-rank ring does not.",
          "",
          "## 4. Cross-run caveats",
          "",
          "- The committed scaling study ran NCCL 2.18.3 bare-metal on an unrecorded idle GPU "
          "slice; this run is NCCL 2.29.2 in the pytorch:26.02 container on host GPUs 2..7. "
          "Peaks agree within 0.5%, so the version/container difference does not matter in this "
          "large-message regime (it does matter for small-message latency — see "
          "`results/symmetric/report.md`).",
          "- GPU 7 held an idle ollama allocation (522 MiB, 0% util) during the run "
          "(`other_tenants.csv`); GPU 0's production service never participates.",
          "- Physical per-link traffic for NVLS is approximated as algbw. The exact multicast "
          "fan-out traffic is switch-internal and not observable from the host; ~algbw is the "
          "per-GPU NVLink port traffic implied by ship-once/receive-once.",
          "",
          "## 5. Raw data",
          "",
          "- `g{2,4,6}_{auto,Ring,NVLS}.log` — full nccl-tests output with interleaved "
          "`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING` lines (the attribution evidence).",
          "- `boxstate.csv`, `other_tenants.csv`, `nccl_version.txt`, `nccl_tests_build.log`.",
          "- Sweep script: `scripts/run_scaling_attributed.sh`; this report + chart: "
          "`analysis/scaling_attribution.py` (re-run -> zero diff).",
          ""]
    with open(f"{DIR}/report.md", "w") as f:
        f.write("\n".join(L))
    print(f"wrote {DIR}/report.md")
    plot(data)


if __name__ == "__main__":
    main()
