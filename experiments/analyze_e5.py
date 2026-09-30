#!/usr/bin/env python3
"""analyze_e5.py - compute E5 halt-regime statistics from the 6 paired
rounds. For each adatrust round: escape latency (attack t=300 -> first QBFT
epoch), pre/post-attack cumulative commits, post-attack TPS, overall TPS.
For each static_tm round: post-attack commit delta (expect ~0) and overall TPS.
Prints per-round table and cross-round mean±std.
"""
import csv
import os
import sys
import numpy as np

D = os.path.expanduser('~/runs/real/e5')
ATTACK_T = 300.0
DUR = 600.0


def load(rounds, mode):
    out = []
    for r in rounds:
        p = f'{D}/{mode}_r{r}.csv'
        rows = []
        with open(p) as f:
            rd = csv.DictReader(f)
            for row in rd:
                rows.append(row)
        out.append(rows)
    return out


def tps_of(cum0, cum1, t0, t1):
    dt = max(t1 - t0, 1e-6)
    return max(cum1 - cum0, 0.0) / dt


ad = load(range(1, 7), 'adatrust')
st = load(range(1, 7), 'static_tm')

print("=== adatrust (n=6) ===")
esc_lat, post_tps, overall = [], [], []
for r, rows in enumerate(ad, 1):
    # pre-attack: last cometbft epoch with t <= ATTACK_T
    pre_cum = None
    pre_t = None
    esc_ep = None
    esc_t = None
    for ep, x in enumerate(rows, 1):
        t = float(x['t'])
        if x['chain'] == 'cometbft':
            if t <= ATTACK_T:
                pre_cum = float(x['cum_tx'])
                pre_t = t
        else:  # besu
            if esc_ep is None:
                esc_ep, esc_t = ep, t
    final_cum = float(rows[-1]['cum_tx'])
    final_t = float(rows[-1]['t'])
    if esc_t is None:
        esc_lat.append(float('nan'))
        post_tps.append(0.0)
    else:
        esc_lat.append(esc_t - ATTACK_T)
        post_tps.append(tps_of(pre_cum or 0.0, final_cum, esc_t, final_t))
    overall.append(final_cum / DUR)
    print(f"r{r}: pre_cum={pre_cum:.0f} escape_t={esc_t} "
          f"lat={esc_lat[-1]:.1f}s post_tps={post_tps[-1]:.1f} "
          f"overall={overall[-1]:.1f}")

esc_lat = np.array(esc_lat)
post_tps = np.array(post_tps)
overall = np.array(overall)
print(f"escape latency: {np.nanmean(esc_lat):.1f} ± {np.nanstd(esc_lat):.1f} s "
      f"(n={np.sum(~np.isnan(esc_lat))}/6)")
print(f"post-attack TPS: {post_tps.mean():.1f} ± {post_tps.std(ddof=0):.1f}")
print(f"overall TPS: {overall.mean():.1f} ± {overall.std(ddof=0):.1f}")

print("\n=== static_tm (n=6, control) ===")
st_overall, st_pre_tps, st_post_tps = [], [], []
for r, rows in enumerate(st, 1):
    # halt epoch = first epoch after which cum_tx stops increasing
    cum_at_halt = None
    halt_t = None
    prev_cum = -1.0
    for x in rows:
        c = float(x['cum_tx'])
        if c == prev_cum and cum_at_halt is None:
            cum_at_halt = prev_cum
            halt_t = float(x['t'])
        prev_cum = c
    # pre-halt TPS
    pre_cum = None
    pre_t = 0.0
    for x in rows:
        t = float(x['t'])
        if t <= ATTACK_T:
            pre_cum = float(x['cum_tx']); pre_t = t
    final_cum = float(rows[-1]['cum_tx'])
    post_delta = max(final_cum - (cum_at_halt or 0.0), 0.0)
    st_post_tps.append(post_delta / max(DUR - (halt_t or ATTACK_T), 1.0))
    st_pre_tps.append((pre_cum or 0.0) / max(pre_t, 1e-6))
    st_overall.append(final_cum / DUR)
    print(f"r{r}: pre_tps={st_pre_tps[-1]:.0f} halt_t={halt_t} "
          f"post_halt_delta={post_delta:.0f} post_tps={st_post_tps[-1]:.1f} "
          f"overall={st_overall[-1]:.1f}")
st_post_tps = np.array(st_post_tps)
st_overall = np.array(st_overall)
print(f"post-halt TPS: {st_post_tps.mean():.1f} ± {st_post_tps.std(ddof=0):.1f} "
      f"(0/6 resume service)")
print(f"overall TPS: {st_overall.mean():.1f} ± {st_overall.std(ddof=0):.1f}")

print("\n=== paired (adatrust - static_tm) overall TPS ===")
d = overall - st_overall
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
print(f"delta: {d.mean():.1f} ± {d.std(ddof=0):.1f}  t={t:.2f} (df={len(d)-1})")
