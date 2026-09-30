"""Generate markdown table rows for the paper from results/raw CSVs.

Reads rq1_summary.csv + rq1_stats.csv (Welch tests) and rq3_summary.csv +
rq3_stats.csv and prints ready-to-paste markdown rows. All numbers in the
paper's Tables IV/VI come from this script's output.
"""
import sys
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "results" / "raw"

ORDER = ["AdaTrust", "B1-PBFT", "B2-HotStuff", "B5-Tendermint",
         "B3-DAPBFT", "B4-StaticTrust"]
LABEL = {"AdaTrust": "**AdaTrust**", "B1-PBFT": "B1 PBFT",
         "B2-HotStuff": "B2 HotStuff", "B5-Tendermint": "B5 Tendermint",
         "B3-DAPBFT": "B3 DA-PBFT", "B4-StaticTrust": "B4 StaticTrust"}


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def rq1():
    s = pd.read_csv(RAW / "rq1_summary.csv")
    st = pd.read_csv(RAW / "rq1_stats.csv")
    print("### TABLE IV rows (RQ1)")
    for wl in ["bursty", "real"]:
        sub = s[s.workload == wl].set_index("scheme")
        if not len(sub):
            continue
        for sch in ORDER:
            if sch not in sub.index:
                continue
            r = sub.loc[sch]
            sig = ""
            if sch != "AdaTrust":
                q = st[(st.workload == wl) & (st.scheme == sch)]
                if len(q):
                    sig = stars(q.iloc[0].tps_p)
            print(f"| {LABEL[sch]} | {wl} | {r.tps_m:.1f} ± {r.tps_s:.1f}{sig} "
                  f"| {r.lat_m:.3f} ± {r.lat_s:.3f} | {r.lat95_m:.3f} "
                  f"| {r.msgs_m:,.0f} | {r.k_m:.1f} | {r.b_m:.2f} "
                  f"| {r.energy_m:.2f} |")
        print()
    print("### RQ1 Welch stats (vs AdaTrust, per workload)")
    print(st.to_string(index=False))
    print()


def rq3():
    s = pd.read_csv(RAW / "rq3_summary.csv")
    st = pd.read_csv(RAW / "rq3_stats.csv")
    print("### TABLE VI candidates (RQ3, adaptive schedule)")
    a = s[(s.schedule == "adaptive")]
    for (atk, f), g in a.groupby(["attack", "f_ratio"]):
        ada = g[g.scheme == "AdaTrust"]
        pb = g[g.scheme == "B1-PBFT"]
        if not len(ada) or not len(pb):
            continue
        ada, pb = ada.iloc[0], pb.iloc[0]
        q = st[(st.attack == atk) & (st.f_ratio == f)
               & (st.schedule == "adaptive") & (st.scheme == "B1-PBFT")]
        sig = stars(q.iloc[0].tps_p) if len(q) else ""
        print(f"| {atk} | {f:.0%} | {ada.tps_m:.1f}{sig} | {pb.tps_m:.1f} "
              f"| {ada.lat_m:.3f} | {pb.lat_m:.3f} | {ada.k_m:.1f} "
              f"| {ada.b_m:.2f} | {ada.dr_m:.2f} / {pb.dr_m:.2f} |")
    print()
    print("### RQ3 Welch stats (A1 rows)")
    print(st[st.attack == "A1"].to_string(index=False))


if __name__ == "__main__":
    rq1()
    rq3()
