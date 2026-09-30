#!/usr/bin/env python3
"""build_paper_excels.py — distill the AdaTrust paper's every figure/table
into ONE Excel file per figure/table, with sheets organized by axes/panels.

Outputs into ../data (workspace data/ directory):
  Table_I.xlsx … Table_VI.xlsx   (one sheet: the table as published)
  Fig_2.xlsx … Fig_6.xlsx        (one sheet per axis/panel; columns = axes)

Raw sources (all already mirrored locally):
  runs/raw_export/{rq1_summary.md, rq2_committee*.csv, rq2/k_capacity_rep*.csv,
                   rq3/*, rq3d5/restore_r*.log, rq4/*, rq5_sensitivity_run.log}
  runs/e3/e3_shard*.csv   (E3 fair baseline, Section IV-B text)
  runs/e5/*.csv           (E5 halt regime = Table V)
"""
import csv
import os
import re
import sys

import numpy as np

try:
    import openpyxl
except ImportError:
    sys.exit("openpyxl missing: pip install openpyxl")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "runs", "raw_export")
E3 = os.path.join(ROOT, "runs", "e3")
E5 = os.path.join(ROOT, "runs", "e5")
OUT = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def wb_save(wb, name):
    p = os.path.join(OUT, name)
    wb.save(p)
    print("wrote", name)


def sheet_rows(ws, header, rows, note=None):
    ws.append(header)
    for r in rows:
        ws.append(r)
    if note:
        ws.append([])
        ws.append(["NOTE", note])


# ---------------------------------------------------------------- helpers
def rq1_per_round():
    """Parse rq1_summary.md -> per-scheme per-round TPS + backlog."""
    txt = open(os.path.join(RAW, "rq1_summary.md")).read()
    per = {}
    for m in re.finditer(
            r"\| (AdaTrust|B5 StaticTM|B1 StaticQBFT|B4 Oracle) \| r\d \| "
            r"[\d.]+ \| \d+ \| ([\d.]+) \|", txt):
        per.setdefault(m.group(1), []).append(float(m.group(2)))
    bl_matches = re.findall(
        r"\*\*(AdaTrust|B5 StaticTM|B1 StaticQBFT|B4 Oracle)\*\*: "
        r"([\d.]+) ± [\d.]+ TPS, avg switches [\d.]+, mean backlog ([\d.]+)",
        txt)
    bl = {name: float(bg) for name, _tps, bg in bl_matches}
    return per, bl


def run_tps(path):
    rows = read_csv(path)
    return float(rows[-1]["cum_tx"]) / float(rows[-1]["t"])


def rq3_retention():
    summ = read_csv(os.path.join(RAW, "rq3", "rq3_summary.csv"))
    return {"adatrust": [float(r["retention"]) for r in summ
                         if r["mode"] == "adatrust"],
            "static_tm": [float(r["retention"]) for r in summ
                          if r["mode"] == "static_tm"]}


# ---------------------------------------------------------------- Table I
def table_i():
    per, bl = rq1_per_round()
    schemes = [("AdaTrust (TM, k=4)", "AdaTrust"),
               ("StaticTM (k=16)", "B5 StaticTM"),
               ("StaticQBFT", "B1 StaticQBFT"),
               ("B3 Reactive switcher", "B4 Oracle")]
    stats = {  # published paired stats (Table I)
        "StaticTM (k=16)": (-1.10, 0.32, -0.45, "[-18.2, +7.3]", 659.2, 9.2, 0, "-"),
        "StaticQBFT": (267.2, "<1e-3", 109.1, "[+563.9, +574.9]", 84.3, 1.5, 0, "-"),
        "B3 Reactive switcher": (42.9, "<1e-3", 17.5, "[+403.6, +455.0]", 224.4, 20.0, 15.0, "10.8±9.4"),
        "AdaTrust (TM, k=4)": ("-", "-", "-", "-", None, None, 0, "-"),
    }
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Table_I"
    ws.append(["scheme", "tps_mean", "tps_std", "switches", "drain_s_per_switch",
               "mean_backlog", "t_vs_AdaTrust", "p", "cohens_dz",
               "ci95_of_delta_tps", "per_round_tps"])
    for label, key in schemes:
        vals = per[key]
        s = stats[label]
        ws.append([label,
                   round(float(np.mean(vals)), 1) if s[4] is None else s[4],
                   round(float(np.std(vals)), 1) if s[5] is None else s[5],
                   s[6] if label != "AdaTrust (TM, k=4)" else 0,
                   s[7],
                   bl.get(key, "-"),
                   s[0], s[1], s[2], s[3],
                   ", ".join(f"{v:.1f}" for v in vals)])
    ws.append([])
    ws.append(["NOTE", "B4 parameter-only PPO (n=3 separate window, 648.4±6.2, "
                        "t=-0.45 p≈0.69) omitted from bars; see paper Table I "
                        "dagger footnote. B3 = 'B4 Oracle' legacy driver name."])
    wb_save(wb, "Table_I.xlsx")


# ---------------------------------------------------------------- Table II
def table_ii():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Table_IIa"
    ws.append(["batch_B", "direction", "wall_s", "blocks", "per_tx_s", "rep"])
    for i, fn in enumerate(("rq2_committee.csv", "rq2_committee_r2.csv",
                            "rq2_committee_r3.csv"), 1):
        for r in read_csv(os.path.join(RAW, fn)):
            ws.append([int(r["batch"]), r["direction"], float(r["wall_s"]),
                       int(r["blocks"]), float(r["per_tx_s"]), i])
    ws2 = wb.create_sheet("Table_IIb")
    ws2.append(["k", "rep", "mean_tps", "hi_tps", "lo_tps"])
    for i in (1, 2, 3):
        for r in read_csv(os.path.join(RAW, "rq2",
                                       f"k_capacity_rep{i}.csv")):
            ws2.append([int(r["k"]), i, float(r["mean_tps"]),
                        float(r["hi_tps"]), float(r["lo_tps"])])
    sheet_rows(ws2, [], [], note=None) if False else None
    ws2.append([])
    ws2.append(["NOTE", "Table II(b) published means: k4 673.6±10.4, "
                        "k8 662.3±5.3, k12 674.3±7.1, k16 675.1±16.3"])
    wb_save(wb, "Table_II.xlsx")


# ---------------------------------------------------------------- Table III
def table_iii():
    ret = rq3_retention()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Table_III"
    ws.append(["metric", "AdaTrust", "StaticTM_k16", "test"])
    ada_ret, st_ret = ret["adatrust"], ret["static_tm"]
    rows = [
        ["retention_post_pre", f"{np.mean(ada_ret):.3f} ± {np.std(ada_ret):.3f}",
         f"{np.mean(st_ret):.3f} ± {np.std(st_ret):.3f}",
         "t=3.17, p<0.05, dz=1.29"],
        ["overall_tps", "630.8 ± 8.0", "570.6 ± 25.8", "+10.5%, t=4.37, p<0.01"],
        ["post_attack_tps", "600.9 ± 16.1", "492.1 ± 55.8", "+22.1%"],
        ["statictm_maxdose_5of16", "-", "retention 0.808±0.047; post 507.5±12.4",
         "separate campaign"],
        ["detection_latency_s", "142.1", "154.9", "both detect (n.s.)"],
        ["healing_k4to13", "6/6 runs", "0/6", "-"],
        ["recovery_ge80pct", "3/6 (best 15.2 s)", "3/6", "n.s."],
    ]
    for r in rows:
        ws.append(r)
    ws.append([])
    ws.append(["per_run_retention_adatrust", ", ".join(f"{v:.3f}" for v in ada_ret)])
    ws.append(["per_run_retention_static_tm", ", ".join(f"{v:.3f}" for v in st_ret)])
    wb_save(wb, "Table_III.xlsx")


# ---------------------------------------------------------------- Table IV
def table_iv():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Table_IV"
    ws.append(["variant", "eval", "tps_mean", "tps_sd", "per_round_tps",
               "paired_delta_vs_full", "stats"])
    variants = ["full", "notrust", "nosizing", "nohyst"]
    res = {}
    for t in variants:
        vals = [run_tps(os.path.join(RAW, "rq4", f"{t}_r{r}.csv"))
                for r in range(1, 7)]
        res[t] = vals
    published = {
        "full": ("646.6", "18.8", "—"),
        "notrust": ("635.8", "16.1",
                    "−10.7 ± 10.3, t=−2.55, p≈0.05, dz=−1.04, [−21.5, +0.1]"),
        "nosizing": ("648.3", "15.4",
                     "+1.8 ± 15.1, t=0.29, n.s., [−14.0, +17.6]"),
        "nohyst": ("640.9", "34.5",
                   "−5.6 ± 35.9, t=−0.38, n.s., [−43.3, +32.1]"),
    }
    names = {"full": "Full AdaTrust", "notrust": "w/o trust observations",
             "nosizing": "w/o sizing (k frozen at 4)",
             "nohyst": "w/o hysteresis"}
    for t in variants:
        m, s, stat = published[t]
        ws.append([names[t], "real chain (n=6)", float(m), float(s),
                   ", ".join(f"{v:.1f}" for v in res[t]), stat, ""])
    ws.append(["w/o switching-cost (ω4=0, retrained)",
               "offline (training env, 4 seeds)", "261.5", "186.6",
               "per-seed terminal-window sim TPS; corner in 4/4 seeds "
               "(slot: TM×3, HS×1); full retrain 460.5",
               "training-time pathology", ""])
    wb_save(wb, "Table_IV.xlsx")


# ---------------------------------------------------------------- Table V
def table_v():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Table_V"
    ws.append(["metric", "AdaTrust", "StaticTM_k16", "test"])
    rows = [
        ["pre_halt_tps", "646.0 ± 18.8", "630.8 ± 14.7", "parity"],
        ["liveness_after_halt", "6/6", "0/6", "-"],
        ["escape_latency_s", "95.1 ± 5.6", "-", "-"],
        ["post_halt_tps", "115.6 ± 24.4", "0.0 (cumulative frozen)", "-"],
        ["overall_tps_600s", "347.5 ± 4.0", "315.1 ± 9.9",
         "+32.4 ± 9.4, t=7.66, p<1e-3"],
    ]
    for r in rows:
        ws.append(r)
    ws.append([])
    ws.append(["NOTE", "per-run raw CSVs: runs/e5/{adatrust,static_tm}_r{1..6}.csv; "
                        "analyzer code/experiments/analyze_e5.py"])
    wb_save(wb, "Table_V.xlsx")


# ---------------------------------------------------------------- Table VI
def table_vi():
    txt = open(os.path.join(RAW, "rq5_sensitivity_run.log")).read()
    data = {}
    for m in re.finditer(
            r"rq5 period=(60|240) mode=(adatrust|static_tm) r\d start"
            r".*?mean_tps=([\d.]+)", txt, re.S):
        data.setdefault((m.group(1), m.group(2)),
                        []).append(float(m.group(3)))
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Table_VI"
    ws.append(["period_s", "mode", "per_run_tps", "mean", "sd", "paired_delta",
               "ci95", "wilcoxon"])
    published = {
        "60": ("+3.0% (t=1.91, dz=1.10)", "[-23.0, +59.6]", "5/1"),
        "240": ("-6.4% (t=-1.84, dz=-1.06)", "[-125.5, +50.2]", "0/6"),
    }
    for p in ("60", "240"):
        for mode in ("adatrust", "static_tm"):
            v = data[(p, mode)]
            ws.append([int(p), mode,
                       ", ".join(f"{x:.1f}" for x in v),
                       round(float(np.mean(v)), 1), round(float(np.std(v)), 1),
                       published[p][0] if mode == "adatrust" else "",
                       published[p][1] if mode == "adatrust" else "",
                       published[p][2] if mode == "adatrust" else ""])
    ws.append([])
    ws.append(["interaction_doubledelta", "56.0 ± 40.9 TPS (t=2.37, df=2, "
               "p≈0.14)", "CI [-46.4, +158.4]"])
    wb_save(wb, "Table_VI.xlsx")


# ---------------------------------------------------------------- Fig 2
def fig2():
    per, bl = rq1_per_round()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "panel_a_bars"
    ws.append(["scheme", "tps_mean", "tps_std", "per_round_tps",
               "oracle_ref"])
    for label, key in [("AdaTrust (k=4)", "AdaTrust"),
                       ("StaticTM (k=16)", "B5 StaticTM"),
                       ("Static QBFT", "B1 StaticQBFT"),
                       ("B3 React. switcher", "B4 Oracle")]:
        v = per[key]
        ws.append([label, round(float(np.mean(v)), 2),
                   round(float(np.std(v)), 2),
                   ", ".join(f"{x:.1f}" for x in v), 653.7])
    ws2 = wb.create_sheet("panel_b_backlog")
    ws2.append(["scheme", "mean_mempool_backlog_tx"])
    for label, key in [("AdaTrust (k=4)", "AdaTrust"),
                       ("StaticTM (k=16)", "B5 StaticTM"),
                       ("Static QBFT", "B1 StaticQBFT"),
                       ("B3 React. switcher", "B4 Oracle")]:
        ws2.append([label, bl.get(key)])
    wb_save(wb, "Fig_2.xlsx")


# ---------------------------------------------------------------- Fig 3
def fig3():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "panel_a_reconfig"
    ws.append(["batch_B", "rep", "wall_s", "per_tx_s"])
    for i, fn in enumerate(("rq2_committee.csv", "rq2_committee_r2.csv",
                            "rq2_committee_r3.csv"), 1):
        for r in read_csv(os.path.join(RAW, fn)):
            ws.append([int(r["batch"]), i, float(r["wall_s"]),
                       float(r["per_tx_s"])])
    ws2 = wb.create_sheet("panel_b_kscan")
    ws2.append(["k", "rep", "mean_tps"])
    for i in (1, 2, 3):
        for r in read_csv(os.path.join(RAW, "rq2",
                                       f"k_capacity_rep{i}.csv")):
            ws2.append([int(r["k"]), i, float(r["mean_tps"])])
    wb_save(wb, "Fig_3.xlsx")


# ---------------------------------------------------------------- Fig 4
def fig4():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "panel_a_timeline_r5"
    ws.append(["t_s", "adatrust_tps_per_epoch", "statictm_tps_per_epoch",
               "adatrust_k"])
    ada = read_csv(os.path.join(RAW, "rq3", "adatrust_r5.csv"))
    sta = read_csv(os.path.join(RAW, "rq3", "static_tm_r5.csv"))

    def tps_series(rows):
        t = [float(r["t"]) for r in rows]
        cum = [float(r["cum_tx"]) for r in rows]
        dt = np.diff(t); dc = np.diff(cum)
        s = np.where(dt > 0, dc / dt, np.nan)
        return t[1:], s

    ta, sa = tps_series(ada)
    ts, ss = tps_series(sta)
    ka = [int(float(r["k"])) for r in ada]
    tax = [float(r["t"]) for r in ada]
    for i in range(max(len(ta), len(ts))):
        ws.append([round(ta[i], 1) if i < len(ta) else None,
                   round(float(sa[i]), 1) if i < len(ta) else None,
                   round(float(ss[i]), 1) if i < len(ts) else None,
                   None])
    ws2 = wb.create_sheet("panel_b_retention")
    ret = rq3_retention()
    ws2.append(["mode", "retention_per_run", "mean", "std"])
    for mode, label in (("adatrust", "AdaTrust"),
                        ("static_tm", "StaticTM")):
        v = ret[mode]
        ws2.append([label, ", ".join(f"{x:.3f}" for x in v),
                    round(float(np.mean(v)), 3), round(float(np.std(v)), 3)])
    ws3 = wb.create_sheet("committee_trace")
    ws3.append(["t_s", "adatrust_k"])
    for t, k in zip(tax, ka):
        ws3.append([round(t, 1), k])
    wb_save(wb, "Fig_4.xlsx")


# ---------------------------------------------------------------- Fig 5
def fig5():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "panel_a_real"
    ws.append(["variant", "per_run_tps", "mean", "std"])
    for t, label in (("full", "Full"), ("notrust", "w/o trust obs"),
                     ("nosizing", "w/o sizing"), ("nohyst", "w/o hysteresis")):
        vals = [run_tps(os.path.join(RAW, "rq4", f"{t}_r{r}.csv"))
                for r in range(1, 7)]
        ws.append([label, ", ".join(f"{v:.1f}" for v in vals),
                   round(float(np.mean(vals)), 1),
                   round(float(np.std(vals)), 1)])
    ws2 = wb.create_sheet("panel_b_sim")
    ws2.append(["variant", "sim_terminal_window_tps_mean", "sd", "n_seeds",
                "note"])
    sim = [("Full", 460.5, None, 1, "retrain s0"),
           ("w/o trust", None, None, 1, "abl_notrust_s0"),
           ("w/o sizing", None, None, 1, "abl_nosizing_s0"),
           ("w/o hysteresis", None, None, 1, "abl_nohyst_s0"),
           ("w/o sw. cost", 261.5, 186.6, 4,
            "4/4 seeds static corner; slots TM×3 HS×1")]
    for r in sim:
        ws2.append(list(r))
    ws2.append([])
    ws2.append(["NOTE", "panel (b) plots sim terminal-window TPS; values for "
                        "notrust/nosizing/nohyst single-seed available in H100 "
                        "train logs; headline numbers as published."])
    wb_save(wb, "Fig_5.xlsx")


# ---------------------------------------------------------------- Fig 6
def fig6():
    txt = open(os.path.join(RAW, "rq5_sensitivity_run.log")).read()
    data = {}
    for m in re.finditer(
            r"rq5 period=(60|240) mode=(adatrust|static_tm) r\d start"
            r".*?mean_tps=([\d.]+)", txt, re.S):
        data.setdefault((m.group(1), m.group(2)),
                        []).append(float(m.group(3)))
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "bars"
    ws.append(["period_s", "mode", "per_run_tps", "mean", "std"])
    for p in ("60", "240"):
        for mode in ("adatrust", "static_tm"):
            v = data[(p, mode)]
            ws.append([int(p), mode, ", ".join(f"{x:.1f}" for x in v),
                       round(float(np.mean(v)), 1),
                       round(float(np.std(v)), 1)])
    wb_save(wb, "Fig_6.xlsx")


if __name__ == "__main__":
    table_i(); table_ii(); table_iii(); table_iv(); table_v(); table_vi()
    fig2(); fig3(); fig4(); fig5(); fig6()
    print("ALL EXCELS DONE ->", OUT)
