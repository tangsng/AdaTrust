#!/usr/bin/env python3
"""Quick sanity check of the calibrated simulator + retrained ppo.pt:
RQ1-style bursty comparison, 5 seeds, before real-chain deployment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from experiments.run_all import evaluate, env_kwargs_for, load_agent

EPOCHS = 50
SEEDS = [0, 1, 2, 3, 4]
schemes = ["AdaTrust", "B1-PBFT", "B2-HotStuff", "B5-Tendermint",
           "B4-StaticTrust"]

agents = {"main": load_agent("experiments/runs/ppo.pt"),
          "pbft": None, "nosw": None}

env_kw = dict(n=64, f_ratio=0.10, attack="A1", workload="bursty",
              committee_mode="trust")
rows = {s: [] for s in schemes}
for seed in SEEDS:
    for s in schemes:
        kw = env_kwargs_for(s, dict(env_kw))
        for r in evaluate(s, agents, kw, EPOCHS, seed):
            rows[s].append(r)

print(f"{'scheme':22s} {'tps':>8} {'lat':>7} {'msgs':>8} {'k':>5} {'b':>5} {'sw':>4}")
for s in schemes:
    tps = np.mean([r['tps'] for r in rows[s]])
    lat = np.mean([r['lat'] for r in rows[s]])
    msgs = np.mean([r['msgs'] for r in rows[s]])
    k = np.mean([r['k'] for r in rows[s]])
    b = np.mean([r['b'] for r in rows[s]])
    sw = np.mean([r['switched'] for r in rows[s]])
    print(f"{s:22s} {tps:8.1f} {lat:7.2f} {msgs:8.0f} {k:5.1f} {b:5.2f} {sw:4.2f}")
