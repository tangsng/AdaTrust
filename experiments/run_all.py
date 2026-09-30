"""Run RQ1-RQ5 evaluations and export raw CSVs to results/raw/.

All numbers in the paper's Section 8 come from these CSVs and nowhere else.
Usage: python run_all.py [--quick]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from scipy import stats as _scipy_stats
except ImportError:                      # fallback: normal approximation
    _scipy_stats = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adatrust.env import ConsensusEnv
from adatrust.ppo import PPO
from adatrust.policies import (StaticPolicy, StaticTrustPolicy, TrainedPolicy)
from adatrust.protocols import PBFT, HOTSTUFF, TENDERMINT

RAW = ROOT.parents[0] / "results" / "raw"
SEEDS = list(range(10))                  # R1: 10 seeds for statistical power
E_MSG_J, E_SIG_J = 1e-4, 3e-3


def load_agent(path, obs_dim=15):
    agent = PPO(obs_dim=obs_dim)
    agent.net.load_state_dict(torch.load(path, map_location="cpu"))
    agent.net.eval()
    return agent


def make_policy(name, agents):
    if name == "AdaTrust":
        return TrainedPolicy(agents["main"])
    if name == "B1-PBFT":
        return StaticPolicy(PBFT)
    if name == "B2-HotStuff":
        return StaticPolicy(HOTSTUFF)
    if name == "B5-Tendermint":
        return StaticPolicy(TENDERMINT)
    if name == "B3-DAPBFT":
        return TrainedPolicy(agents["pbft"], restrict_pid=PBFT, use_gate=False)
    if name == "B4-StaticTrust":
        return StaticTrustPolicy()
    if name == "A1-NoSwitch":
        class P(StaticTrustPolicy):
            pass
        return P()                                   # trust election, fixed PBFT
    if name == "A2-NoTrust":
        return TrainedPolicy(agents["main"])
    if name == "A3-NoHyst":
        return TrainedPolicy(agents["main"], use_gate=False)
    if name == "A4-NoSwCost":
        return TrainedPolicy(agents["nosw"])
    raise ValueError(name)


def env_kwargs_for(name, base):
    kw = dict(base)
    if name in ("B1-PBFT", "B2-HotStuff", "B5-Tendermint",
                "B3-DAPBFT", "A2-NoTrust"):
        kw["committee_mode"] = "full"
    elif name in ("B4-StaticTrust",):
        kw["committee_mode"] = "static_trust"
    else:
        kw["committee_mode"] = "trust"
    return kw


def evaluate(name, agents, env_kw, epochs, seed):
    env = ConsensusEnv(seed=seed, **env_kw)
    policy = make_policy(name, agents)
    obs = env.observe()
    rows = []
    for _ in range(epochs):
        action, _ = policy.act(env, obs)
        obs, r, info = env.run_epoch(action)
        info.update(epoch=env.t, seed=seed, scheme=name,
                    energy=info["msgs"] * (E_MSG_J + E_SIG_J))
        rows.append(info)
    return rows


def summarize(df, group_cols):
    agg = df.groupby(group_cols).agg(
        tps_m=("tps", "mean"), tps_s=("tps", "std"),
        tps_med=("tps", "median"),
        tps_q1=("tps", lambda x: x.quantile(0.25)),
        tps_q3=("tps", lambda x: x.quantile(0.75)),
        lat_m=("lat", "mean"), lat_s=("lat", "std"),
        lat_med=("lat", "median"),
        lat_q1=("lat", lambda x: x.quantile(0.25)),
        lat_q3=("lat", lambda x: x.quantile(0.75)),
        lat95_m=("lat_p95", "mean"),
        msgs_m=("msgs", "mean"), dr_m=("dr", "mean"), fpr_m=("fpr", "mean"),
        alarms_m=("alarms", "mean"), stalls_m=("stalls", "mean"),
        k_m=("k", "mean"), b_m=("b", "mean"),
        sw_m=("switched", "sum"), energy_m=("energy", "mean"),
    ).reset_index()
    return agg


def welch_vs_adatrust(df, group_cols, metrics=("tps", "lat")):
    """Paired-by-seed Welch t-tests of every scheme against AdaTrust.

    Per (group, scheme) the per-seed epoch means are computed first, so each
    sample point is one independent run (n = len(SEEDS)). Reports Welch's t,
    p-value, and Cohen's d (pooled SD). CRN pairing makes per-seed differences
    meaningful; we additionally report the mean paired difference.
    """
    rows = []
    per_seed = (df.groupby(group_cols + ["scheme", "seed"])[list(metrics)]
                .mean().reset_index())
    for keys, sub in per_seed.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ref = sub[sub.scheme == "AdaTrust"]
        for scheme in sorted(sub.scheme.unique()):
            if scheme == "AdaTrust":
                continue
            cur = sub[sub.scheme == scheme]
            m = ref.merge(cur, on="seed", suffixes=("_ada", "_cur"))
            row = dict(zip(group_cols, keys))
            row["scheme"] = scheme
            row["n_seeds"] = len(m)
            for met in metrics:
                a, c = m[f"{met}_ada"].to_numpy(), m[f"{met}_cur"].to_numpy()
                if len(a) < 2:
                    continue
                if _scipy_stats is not None:
                    t, p = _scipy_stats.ttest_ind(a, c, equal_var=False)
                else:
                    va, vc = a.var(ddof=1), c.var(ddof=1)
                    t = (a.mean() - c.mean()) / math.sqrt(va / len(a) + vc / len(c) + 1e-12)
                    p = math.erfc(abs(t) / math.sqrt(2.0))
                sp = math.sqrt((a.var(ddof=1) + c.var(ddof=1)) / 2.0) + 1e-12
                row[f"{met}_t"] = float(t)
                row[f"{met}_p"] = float(p)
                row[f"{met}_cohend"] = float((a.mean() - c.mean()) / sp)
                row[f"{met}_paired_diff"] = float(np.mean(a - c))
            rows.append(row)
    return pd.DataFrame(rows)


def main(quick=False):
    epochs = 60 if quick else 150
    RAW.mkdir(parents=True, exist_ok=True)
    trace = None
    tpath = ROOT.parents[0] / "data" / "eth_trace.csv"
    if tpath.exists():
        trace = pd.read_csv(tpath)["rate_tps"].to_numpy()
        # scale the real trace's *shape* to a system-relevant mean load
        # (documented in the paper: preserves burst structure, not absolute rate)
        trace = trace / trace.mean() * 800.0
        print(f"[trace] loaded {len(trace)} samples from {tpath} "
              f"(rescaled to mean 800 tx/s)")
    else:
        print("[trace] NOT FOUND - real-trace rows will use bursty workload")

    agents = {
        "main": load_agent(ROOT / "experiments" / "runs" / "ppo.pt"),
        "pbft": load_agent(ROOT / "experiments" / "runs" / "ppo_pbft.pt"),
        "nosw": load_agent(ROOT / "experiments" / "runs" / "ppo_nosw.pt"),
    }

    # ---------- RQ1: performance vs baselines ----------
    schemes = ["AdaTrust", "B1-PBFT", "B2-HotStuff", "B5-Tendermint",
               "B3-DAPBFT", "B4-StaticTrust"]
    rows = []
    for wl in (["real"] if trace is not None else []) + ["bursty"]:
        for s in schemes:
            for seed in SEEDS:
                kw = env_kwargs_for(s, dict(n=64, f_ratio=0.10, attack="A1",
                                            schedule="adaptive", workload=wl,
                                            trace=trace))
                for r in evaluate(s, agents, kw, epochs, seed):
                    r["workload"] = wl
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RAW / "rq1_epochs.csv", index=False)
    summarize(df, ["scheme", "workload"]).to_csv(RAW / "rq1_summary.csv",
                                                 index=False)
    welch_vs_adatrust(df, ["workload"]).to_csv(RAW / "rq1_stats.csv", index=False)
    print("[rq1] done", len(df))

    # ---------- RQ2: scalability ----------
    rows = []
    for n in [4, 16, 64, 256]:
        for s in ["AdaTrust", "B1-PBFT", "B2-HotStuff"]:
            for seed in SEEDS:
                kw = env_kwargs_for(s, dict(n=n, f_ratio=0.10, attack="A1",
                                            schedule="static", workload="bursty",
                                            k_max=min(64, n)))
                for r in evaluate(s, agents, kw, epochs, seed):
                    r["n"] = n
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RAW / "rq2_epochs.csv", index=False)
    summarize(df, ["scheme", "n"]).to_csv(RAW / "rq2_summary.csv", index=False)
    print("[rq2] done", len(df))

    # ---------- RQ3: attack resilience ----------
    rows = []
    for atk in ["A1", "A2", "A3", "A4"]:
        for f in [0.10, 0.20, 0.25, 0.30]:
            for sched in ["static", "adaptive"]:
                for s in ["AdaTrust", "B1-PBFT"]:
                    for seed in SEEDS:
                        kw = env_kwargs_for(s, dict(n=64, f_ratio=f, attack=atk,
                                                    schedule=sched,
                                                    workload="bursty"))
                        for r in evaluate(s, agents, kw, epochs, seed):
                            r.update(attack=atk, f_ratio=f, schedule=sched)
                            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RAW / "rq3_epochs.csv", index=False)
    summarize(df, ["scheme", "attack", "f_ratio", "schedule"]).to_csv(
        RAW / "rq3_summary.csv", index=False)
    welch_vs_adatrust(df, ["attack", "f_ratio", "schedule"]).to_csv(
        RAW / "rq3_stats.csv", index=False)
    print("[rq3] done", len(df))

    # ---------- RQ4: ablation ----------
    rows = []
    for s in ["AdaTrust", "A1-NoSwitch", "A2-NoTrust", "A3-NoHyst", "A4-NoSwCost"]:
        for seed in SEEDS:
            kw = env_kwargs_for(s, dict(n=64, f_ratio=0.20, attack="A3",
                                        schedule="adaptive", workload="bursty"))
            rows += evaluate(s, agents, kw, epochs, seed)
    df = pd.DataFrame(rows)
    df.to_csv(RAW / "rq4_epochs.csv", index=False)
    summarize(df, ["scheme"]).to_csv(RAW / "rq4_summary.csv", index=False)
    print("[rq4] done", len(df))

    # ---------- RQ5: sensitivity ----------
    rows = []
    for dh in [0.01, 0.05, 0.10, 0.20]:
        for seed in SEEDS:
            kw = env_kwargs_for("AdaTrust", dict(n=64, f_ratio=0.15, attack="A1",
                                                 schedule="adaptive",
                                                 workload="bursty",
                                                 hysteresis=(dh, 5)))
            for r in evaluate("AdaTrust", agents, kw, epochs, seed):
                r.update(param="delta_h", value=dh)
                rows.append(r)
    for lam in [0.90, 0.95, 0.98, 0.999]:
        for seed in SEEDS:
            kw = env_kwargs_for("AdaTrust", dict(n=64, f_ratio=0.15, attack="A2",
                                                 schedule="static",
                                                 workload="bursty", lam=lam))
            for r in evaluate("AdaTrust", agents, kw, epochs, seed):
                r.update(param="lambda_d", value=lam)
                rows.append(r)
    for km in [16, 32, 64, 128]:
        for seed in SEEDS:
            kw = env_kwargs_for("AdaTrust", dict(n=256, f_ratio=0.10, attack="A1",
                                                 schedule="static",
                                                 workload="bursty", k_max=km))
            for r in evaluate("AdaTrust", agents, kw, epochs, seed):
                r.update(param="k_max", value=km)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RAW / "rq5_epochs.csv", index=False)
    summarize(df, ["param", "value"]).to_csv(RAW / "rq5_summary.csv", index=False)
    print("[rq5] done", len(df))
    print("[all] CSVs ->", RAW)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)
