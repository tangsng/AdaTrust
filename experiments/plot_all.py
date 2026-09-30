"""Generate publication-grade figures (300 DPI, PNG+PDF) from results/raw/*.csv.

Every figure reads only the exported CSVs - numbers are never hand-edited.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT.parents[0] / "results" / "raw"
FIG = ROOT.parents[0] / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("[fig]", name)


def rq1():
    df = pd.read_csv(RAW / "rq1_summary.csv")
    wls = list(df.workload.unique()) if "workload" in df else [None]
    schemes = list(df.scheme.unique())
    import numpy as np
    xpos = np.arange(len(schemes))
    w = 0.8 / max(len(wls), 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = {"real": "#4878d0", "bursty": "#ee854a", None: "#4878d0"}
    for i, wl in enumerate(wls):
        sub = df[df.workload == wl] if wl else df
        sub = sub.set_index("scheme").reindex(schemes)
        off = (i - (len(wls) - 1) / 2) * w
        lbl = f"{wl} trace" if wl else ""
        axes[0].bar(xpos + off, sub.tps_m, w, yerr=sub.tps_s, capsize=2,
                    color=colors.get(wl), label=lbl)
        axes[1].bar(xpos + off, sub.lat_m, w, yerr=sub.lat_s, capsize=2,
                    color=colors.get(wl), label=lbl)
        axes[2].bar(xpos + off, sub.dr_m, w, color=colors.get(wl), label=lbl)
    axes[0].set_ylabel("Throughput (TPS)")
    axes[1].set_ylabel("Mean confirmation latency (s)")
    axes[2].set_ylabel("Detection rate")
    for ax in axes:
        ax.set_xticks(xpos)
        ax.set_xticklabels(schemes, rotation=25, ha="right")
        ax.legend(fontsize=8)
    axes[0].set_title("(a) Throughput")
    axes[1].set_title("(b) Latency")
    axes[2].set_title("(c) Byzantine detection rate")
    fig.suptitle("RQ1: AdaTrust vs. baselines (n=64, f=10%)", y=1.03)
    save(fig, "fig3_rq1_performance")


def rq2():
    df = pd.read_csv(RAW / "rq2_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for s, g in df.groupby("scheme"):
        g = g.sort_values("n")
        axes[0].plot(g.n, g.tps_m, marker="o", label=s)
        axes[1].plot(g.n, g.lat_m, marker="s", label=s)
        axes[2].plot(g.n, g.msgs_m, marker="^", label=s)
    axes[0].set_ylabel("Throughput (TPS)")
    axes[1].set_ylabel("Mean latency (s)")
    axes[2].set_ylabel("Messages / epoch")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([4, 16, 64, 256])
        ax.set_xticklabels([4, 16, 64, 256])
        ax.set_xlabel("Number of validators n")
        ax.legend()
    axes[0].set_title("(a) Throughput vs. n")
    axes[1].set_title("(b) Latency vs. n")
    axes[2].set_title("(c) Communication overhead")
    fig.suptitle("RQ2: Scalability", y=1.03)
    save(fig, "fig4_rq2_scalability")


def rq3():
    df = pd.read_csv(RAW / "rq3_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (s, sched), g in df[df.schedule == "adaptive"].groupby(["scheme", "attack"]):
        g = g.sort_values("f_ratio")
        axes[0].plot(g.f_ratio * 100, g.tps_m, marker="o", label=f"{s}-{sched}")
    for (s, sched), g in df.groupby(["scheme", "schedule"]):
        g = g.groupby("f_ratio").mean(numeric_only=True).reset_index()
        axes[1].plot(g.f_ratio * 100, g.dr_m, marker="s", label=f"{s}-{sched}")
    axes[0].set_xlabel("Byzantine ratio (%)")
    axes[0].set_ylabel("Throughput (TPS)")
    axes[0].set_title("(a) Degradation under attacks (adaptive)")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Byzantine ratio (%)")
    axes[1].set_ylabel("Detection rate")
    axes[1].set_title("(b) Byzantine detection")
    axes[1].legend()
    fig.suptitle("RQ3: Attack resilience", y=1.03)
    save(fig, "fig5_rq3_attacks")


def rq4():
    df = pd.read_csv(RAW / "rq4_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = range(len(df))
    for ax, col, ylab in zip(axes, ["tps_m", "lat_m", "dr_m"],
                             ["Throughput (TPS)", "Mean latency (s)",
                              "Detection rate"]):
        ax.bar(x, df[col], color="#956cb4")
        ax.set_ylabel(ylab)
        ax.set_xticks(list(x))
        ax.set_xticklabels(df.scheme, rotation=25, ha="right")
    axes[0].set_title("(a) Throughput")
    axes[1].set_title("(b) Latency")
    axes[2].set_title("(c) Detection")
    fig.suptitle("RQ4: Ablation (n=64, f=20%, A3 collusion)", y=1.03)
    save(fig, "fig6_rq4_ablation")


def rq5():
    df = pd.read_csv(RAW / "rq5_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, p, xl in zip(axes, ["delta_h", "lambda_d", "k_max"],
                         [r"Hysteresis margin $\delta_h$",
                          r"Trust decay $\lambda_d$", r"$k_{max}$"]):
        g = df[df.param == p].sort_values("value")
        ax.plot(g.value, g.tps_m, marker="o", label="TPS")
        ax.set_xlabel(xl)
        ax.set_ylabel("Throughput (TPS)")
        ax2 = ax.twinx()
        ax2.plot(g.value, g.sw_m, marker="s", color="#d65f5f", label="Switches")
        ax2.set_ylabel("Switches / run", color="#d65f5f")
        ax2.spines["right"].set_visible(True)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    fig.suptitle("RQ5: Parameter sensitivity", y=1.03)
    save(fig, "fig7_rq5_sensitivity")


if __name__ == "__main__":
    rq1(); rq2(); rq3(); rq4(); rq5()
    print("figures ->", FIG)
