#!/usr/bin/env python3
"""replot_from_excel.py — regenerate the AdaTrust paper's figures (and print
tables) FROM the per-figure Excel files in the workspace data/ directory.

Design contract (user requirement):
  * each figure/table has its own Excel: data/Table_I.xlsx … Table_VI.xlsx,
    data/Fig_2.xlsx … Fig_6.xlsx (sheets organized by axis/panel);
  * this script reads ONLY those Excel files (no raw runs/ access);
  * each regenerated artifact is named by its paper number
    (fig2_rq1_real.png/pdf … fig6_rq5_real.png/pdf; tables are printed and
    also saved as CSV next to the figures).

Usage:
  python replot_from_excel.py            # all figures
  python replot_from_excel.py fig2 fig4  # subset
Output dir: data/ (same directory as the Excel files).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
C = {"ada": "#b2182b", "tm": "#2166ac", "qbft": "#4dac26", "orc": "#984ea3",
     "full": "#b2182b", "notrust": "#e08214", "nosizing": "#2166ac",
     "nohyst": "#4dac26", "nosw": "#7570b3"}


def load(wb_name, sheet=None):
    wb = openpyxl.load_workbook(os.path.join(DATA, wb_name), data_only=True)
    ws = wb[sheet] if sheet else wb.active
    rows = list(ws.iter_rows(values_only=True))
    header, body = rows[0], [r for r in rows[1:]
                             if r and not (isinstance(r[0], str)
                                           and r[0] in ("NOTE", ""))]
    return header, body


def csvlist(cell):
    """'653.1, 655.2, …' -> [floats] (single float passthrough)."""
    if cell is None:
        return []
    if isinstance(cell, (int, float)):
        return [float(cell)]
    return [float(x) for x in str(cell).split(",")]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(DATA, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", f"{name}.png/.pdf")


def table_csv(name, header, rows):
    import csv as _csv
    p = os.path.join(DATA, f"{name}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", f"{name}.csv")


# ------------------------------------------------------------------ fig 2
def fig2():
    _, body = load("Fig_2.xlsx", "panel_a_bars")
    labels, means, stds, per = [], [], [], []
    for r in body:
        labels.append(r[0]); means.append(float(r[1]))
        stds.append(float(r[2])); per.append(csvlist(r[3]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))
    x = np.arange(len(labels))
    colors = [C["ada"], C["tm"], C["qbft"], C["orc"]]
    ax1.bar(x, means, yerr=stds, capsize=3, color=colors, width=0.62,
            label="mean ± std; dots = per-round")
    for xi, pv in zip(x, per):
        ax1.scatter(np.full(len(pv), xi), pv, color="black", s=6,
                    zorder=5, alpha=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels([l.replace(" ", "\n") for l in labels])
    ax1.set_ylabel("Committed TPS (mean ± std)")
    ax1.set_title("(a) Headline throughput (n = 6 paired rounds)")
    ax1.axhline(653.7, color="#888888", lw=1.0, ls="--",
                label="clairvoyant oracle (degenerate) = 653.7")
    h, l = ax1.get_legend_handles_labels()
    ax1.legend(h, l, loc="upper right", fontsize=6.5)
    ax1.set_ylim(0, 780)

    _, bodyb = load("Fig_2.xlsx", "panel_b_backlog")
    bl = [float(r[1]) for r in bodyb]
    b2 = ax2.bar(x, bl, color=colors, width=0.62)
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace(" ", "\n") for l in labels], fontsize=7)
    ax2.set_ylabel("Mean mempool backlog (tx)")
    ax2.set_title("(b) Backlog under square-wave load")
    ax2.legend([b2], ["run-averaged backlog"], loc="upper left")
    save(fig, "fig2_rq1_real")


# ------------------------------------------------------------------ fig 3
def fig3():
    _, body = load("Fig_3.xlsx", "panel_a_reconfig")
    wall = {}
    for r in body:
        wall.setdefault(int(r[0]), []).append(float(r[2]))
    batches = sorted(wall)
    _, bodyb = load("Fig_3.xlsx", "panel_b_kscan")
    krep = {}
    for r in bodyb:
        krep.setdefault(int(r[0]), []).append(float(r[2]))
    ks = sorted(krep)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6))
    wm = [np.mean(wall[b]) for b in batches]
    wlo = [min(wall[b]) for b in batches]
    whi = [max(wall[b]) for b in batches]
    ax1.bar([str(b) for b in batches], wm,
            yerr=[np.array(wm) - wlo, np.array(whi) - wm], capsize=3,
            color=C["tm"], width=0.5, label="wall time (range, 3 reps)")
    ax1.set_xlabel("Batch size $B$ (ValidatorUpdates)")
    ax1.set_ylabel("Wall time (s)"); ax1.set_ylim(0, 9.5)
    ax1.set_title("(a) Reconfiguration cost vs. batch")
    ax1.legend(loc="upper left")

    ax2.errorbar(ks, [np.mean(krep[k]) for k in ks],
                 yerr=[np.std(krep[k]) for k in ks], fmt="s-", color=C["ada"],
                 capsize=3, label="mean ± SD TPS (3 serial reps)")
    for k in ks:
        ax2.scatter(np.full(len(krep[k]), k), krep[k], color="black",
                    s=6, alpha=0.5, zorder=5)
    ax2.set_xticks(ks); ax2.set_xlabel("Committee size $k$")
    ax2.set_ylabel("Mean TPS"); ax2.set_ylim(640, 710)
    ktps = [np.mean(krep[k]) for k in ks]
    ax2.axhspan(min(ktps), max(ktps), color=C["ada"], alpha=0.08)
    ax2.set_title("(b) Throughput is flat in $k$ (3 reps)")
    ax2.legend(loc="lower right")
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig3_rq2_real")


# ------------------------------------------------------------------ fig 4
def fig4():
    _, body = load("Fig_4.xlsx", "panel_a_timeline_r5")
    ta = [r[0] for r in body if r[0] is not None]
    sa = [r[1] for r in body if r[1] is not None]
    ss = [r[2] for r in body if r[2] is not None]
    _, kbody = load("Fig_4.xlsx", "committee_trace")
    tax = [r[0] for r in kbody]; ka = [r[1] for r in kbody]
    _, rbody = load("Fig_4.xlsx", "panel_b_retention")
    ret = {r[0]: csvlist(r[1]) for r in rbody}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.7),
                                   gridspec_kw={"width_ratios": [1.7, 1]})
    ax1.plot(ta, sa, color=C["ada"], lw=1.2, label="AdaTrust TPS")
    ax1.plot(ta[:len(ss)], ss, color=C["tm"], lw=1.2, alpha=0.85,
             label="StaticTM TPS")
    ax1b = ax1.twinx(); ax1b.grid(False)
    ax1b.step(tax, ka, color="#555555", lw=1.0, ls=":", where="post",
              label="AdaTrust $k$")
    ax1b.set_ylabel("Committee size $k$"); ax1b.set_ylim(0, 18)
    ax1.set_ylim(0, 1420)
    ax1.axvline(300.0, color="black", ls="--", lw=1)
    ax1.annotate("attack @ $t$=300 s", (292, 1400), fontsize=6.5,
                 ha="right", va="top")
    ax1.set_xlabel("Wall time (s)"); ax1.set_ylabel("TPS (per epoch)")
    ax1.set_title("(a) Silence attack: response (run r5)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left")

    x = np.arange(2)
    means = [np.mean(ret["AdaTrust"]), np.mean(ret["StaticTM"])]
    stds = [np.std(ret["AdaTrust"]), np.std(ret["StaticTM"])]
    bars = ax2.bar(x, means, yerr=stds, capsize=3,
                   color=[C["ada"], C["tm"]], width=0.5)
    for xi, key in zip(x, ["AdaTrust", "StaticTM"]):
        ax2.scatter(np.full(len(ret[key]), xi), ret[key], color="black",
                    s=6, zorder=5, alpha=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(["AdaTrust", "StaticTM"])
    ax2.set_ylabel("Retention (post/pre TPS)"); ax2.set_ylim(0.4, 1.05)
    ax2.set_title("(b) Retention, n = 6 (t = 3.17, p < 0.05)")
    ax2.legend([bars], ["mean ± std; dots = per-run"], loc="lower left")
    fig.subplots_adjust(wspace=0.55)
    save(fig, "fig4_rq3_real")


# ------------------------------------------------------------------ fig 5
def fig5():
    _, body = load("Fig_5.xlsx", "panel_a_real")
    labels, means, stds, per = [], [], [], []
    for r in body:
        labels.append(r[0]); per.append(csvlist(r[1]))
        means.append(np.mean(per[-1])); stds.append(np.std(per[-1]))
    _, bodyb = load("Fig_5.xlsx", "panel_b_sim")
    blabels, bmeans, bstds = [], [], []
    for r in bodyb:
        if r[0] in (None, "NOTE", ""):
            continue
        blabels.append(r[0])
        bmeans.append(float(r[1]) if r[1] is not None else np.nan)
        bstds.append(float(r[2]) if r[2] is not None else 0.0)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    fig.subplots_adjust(wspace=0.35)
    x = np.arange(len(labels))
    colors_a = [C["full"], C["notrust"], C["nosizing"], C["nohyst"]]
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors_a, width=0.6)
    for xi, pv in zip(x, per):
        ax.scatter(np.full(len(pv), xi), pv, color="black", s=7,
                   zorder=5, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("TPS (mean ± std)"); ax.set_ylim(540, 700)
    ax.set_title("(a) Real chain (n = 6 runs)", fontsize=8.5)
    ax.legend([bars], ["mean ± std; dots = per-run"], loc="lower left")

    xb = np.arange(len(blabels))
    keep = [i for i, v in enumerate(bmeans) if not np.isnan(v)]
    barsb = axb.bar(xb[keep], [bmeans[i] for i in keep],
                    yerr=[bstds[i] for i in keep], capsize=3,
                    color=[C["full"], C["nosw"]][:len(keep)], width=0.6)
    axb.set_xticks(xb)
    axb.set_xticklabels([l.replace(" ", "\n") for l in blabels], fontsize=7)
    axb.set_ylabel("Sim TPS (last-6 mean ± std)")
    axb.set_title("(b) Simulator, trained variants", fontsize=8.5)
    axb.legend([barsb], ["terminal-window mean"], loc="lower left")
    save(fig, "fig5_rq4_real")


# ------------------------------------------------------------------ fig 6
def fig6():
    _, body = load("Fig_6.xlsx", "bars")
    data = {}
    for r in body:
        data.setdefault((int(r[0]), r[1]), csvlist(r[2]))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    periods = [60, 240]
    x = np.arange(2); w = 0.34
    am = [np.mean(data[(p, "adatrust")]) for p in periods]
    asm = [np.std(data[(p, "adatrust")]) for p in periods]
    sm = [np.mean(data[(p, "static_tm")]) for p in periods]
    ssm = [np.std(data[(p, "static_tm")]) for p in periods]
    ax.bar(x - w / 2, am, w, yerr=asm, capsize=3, color=C["ada"],
           label="AdaTrust ($k{=}4$)")
    ax.bar(x + w / 2, sm, w, yerr=ssm, capsize=3, color=C["tm"],
           label="StaticTM ($k{=}16$)")
    for xi, p in zip(x, periods):
        ax.scatter(np.full(len(data[(p, "adatrust")]), xi - w / 2),
                   data[(p, "adatrust")], color="black", s=6, zorder=5,
                   alpha=0.6)
        ax.scatter(np.full(len(data[(p, "static_tm")]), xi + w / 2),
                   data[(p, "static_tm")], color="black", s=6, zorder=5,
                   alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"$P$={p} s" for p in periods])
    ax.set_ylabel("Mean TPS"); ax.set_ylim(0, 800)
    ax.set_title("RQ6 phase-period sensitivity")
    ax.legend(loc="upper right")
    save(fig, "fig6_rq5_real")


# ------------------------------------------------------------------ tables
def tables():
    spec = [("Table_I.xlsx", "Table_I"), ("Table_II.xlsx", "Table_IIa"),
            ("Table_II.xlsx", "Table_IIb"), ("Table_III.xlsx", "Table_III"),
            ("Table_IV.xlsx", "Table_IV"), ("Table_V.xlsx", "Table_V"),
            ("Table_VI.xlsx", "Table_VI")]
    for fn, sheet in spec:
        wb = openpyxl.load_workbook(os.path.join(DATA, fn), data_only=True)
        ws = wb[sheet]
        rows = list(ws.iter_rows(values_only=True))
        rows = [r for r in rows
                if not (r and isinstance(r[0], str) and r[0] == "NOTE")]
        table_csv(sheet, rows[0],
                  [r for r in rows[1:] if any(v is not None for v in r)])
        print(f"--- {sheet} ---")
        for r in rows[:9]:
            print(" | ".join("" if v is None else str(v) for v in r[:6]))


FUNCS = {"fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5,
         "fig6": fig6, "tables": tables}

if __name__ == "__main__":
    args = sys.argv[1:]
    targets = args if args else list(FUNCS)
    for t in targets:
        FUNCS[t]()
    print("DONE ->", DATA)
