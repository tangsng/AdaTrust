"""Select the best main-agent checkpoint on held-out validation seeds.

Candidates: runs/ppo_s{0,1,2}.pt. Validation: seeds {100, 101} (disjoint from
the evaluation seeds 0-9 used by run_all.py), bursty workload, f=15%, A1
adaptive — the training distribution. The winner is copied to runs/ppo.pt.
This is standard model selection and is disclosed in the paper.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adatrust.env import ConsensusEnv
from adatrust.ppo import PPO
from adatrust.policies import TrainedPolicy

CANDS = [0, 1, 2]
VAL_SEEDS = [100, 101]
EPOCHS = 40


def validate(ckpt):
    agent = PPO(obs_dim=15)
    agent.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    agent.net.eval()
    pol = TrainedPolicy(agent)
    trace = None
    tpath = ROOT.parents[0] / "data" / "eth_trace.csv"
    if tpath.exists():
        import pandas as pd
        trace = pd.read_csv(tpath)["rate_tps"].to_numpy()
        trace = trace / trace.mean() * 800.0
    rets = []
    for s in VAL_SEEDS:
        for wl, tr in (("bursty", None), ("real", trace)):
            if wl == "real" and tr is None:
                continue
            env = ConsensusEnv(n=64, f_ratio=0.15, attack="A1",
                               schedule="adaptive", workload=wl, trace=tr,
                               seed=s, committee_mode="trust",
                               weights=(3.0, 1.0, 1.0, 1.0), tps_ref=1200.0)
            obs = env.observe()
            tot = 0.0
            for _ in range(EPOCHS):
                action, _ = pol.act(env, obs)
                obs, r, _ = env.run_epoch(action)
                tot += r
            rets.append(tot / EPOCHS)
    return float(np.mean(rets))


def main():
    scores = {}
    for s in CANDS:
        p = ROOT / "experiments" / "runs" / f"ppo_s{s}.pt"
        if p.exists():
            scores[s] = validate(p)
            print(f"seed {s}: val reward {scores[s]:+.4f}", flush=True)
    if not scores:
        print("no candidates found; keeping existing ppo.pt")
        return
    best = max(scores, key=scores.get)
    src = ROOT / "experiments" / "runs" / f"ppo_s{best}.pt"
    dst = ROOT / "experiments" / "runs" / "ppo.pt"
    shutil.copy(src, dst)
    print(f"[selected] seed {best} -> {dst}")


if __name__ == "__main__":
    main()
