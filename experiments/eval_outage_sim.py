#!/usr/bin/env python3
"""eval_outage_sim.py - E5 sim-side check: scripted Tendermint-niche outage
window; does the frozen policy escape (switch protocol), and how fast?

Usage: python3 eval_outage_sim.py CKPT [t0] [t1] [seed]
Prints per-epoch protocol/tps/outage around the window and summary metrics:
escape latency (epochs from outage start to first epoch finalized on another
niche), mean tps in-window (policy) vs static-TM control (always 0).
"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/projects/code'))
import numpy as np
import torch

from adatrust.env import ConsensusEnv
from adatrust.ppo import PPO
from adatrust.policies import apply_gate

CKPT = sys.argv[1]
T0 = int(sys.argv[2]) if len(sys.argv) > 2 else 60
T1 = int(sys.argv[3]) if len(sys.argv) > 3 else 120
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7

TM = 2
script = [(TM, T0, T1)]

def run_policy():
    env = ConsensusEnv(n=64, f_ratio=0.15, attack="A1", schedule="adaptive",
                       workload="bursty", seed=SEED,
                       hysteresis=(0.05, 5), weights=(3.0, 1.0, 1.0, 1.0),
                       outage_script=script)
    env.protocol = TM   # deployment-faithful init: the real agent starts on
                      # CometBFT (protocol=2); the sim default (PBFT) would
                      # trap escape-trained policies on the wrong niche
    ppo = PPO(obs_dim=15)
    ppo.net.load_state_dict(torch.load(os.path.expanduser(CKPT),
                                       map_location='cpu'))
    ppo.net.eval()
    obs = env.observe()
    log = []
    for ep in range(T1 + 40):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, v = ppo.net(x)
        acts = [int(torch.argmax(lg, dim=-1).item()) for lg in logits]
        if acts[0] == 1:
            acts[0] = 2
        margin = abs(float(v.item())) * 0.1 + (0.06 if acts[0] != env.protocol
                                               else 0.0)
        gated, _ = apply_gate(env, tuple(acts), margin)
        obs, r, info = env.run_epoch(gated)
        log.append(info)
    return log

def run_static_tm():
    env = ConsensusEnv(n=64, f_ratio=0.15, attack="A1", schedule="adaptive",
                       workload="bursty", seed=SEED, committee_mode="full",
                       outage_script=script)
    obs = env.observe()
    log = []
    for ep in range(T1 + 40):
        obs, r, info = env.run_epoch((TM, 0, 0, 0))
        log.append(info)
    return log

pol, sta = run_policy(), run_static_tm()

escape = None
for i, info in enumerate(pol):
    if i >= T0 and info["protocol"] != TM and info["tps"] > 1.0:
        escape = i - T0
        break
in_win = pol[T0:T1]
tps_win = float(np.mean([x["tps"] for x in in_win]))
tps_pre = float(np.mean([x["tps"] for x in pol[T0 - 20:T0]]))
sta_win = float(np.mean([x["tps"] for x in sta[T0:T1]]))
sw_in = sum(x["switched"] for x in in_win)

print(f"CKPT {CKPT}")
print(f"window [{T0},{T1}) TM outage | pre-window tps {tps_pre:.0f}")
print(f"policy: in-window mean tps {tps_win:.0f} | switches {sw_in} | "
      f"escape latency {escape if escape is not None else 'NEVER'} epochs")
print(f"static-TM: in-window mean tps {sta_win:.0f}")
print("ep: proto tps outage (window +/- 5)")
for i in range(T0 - 5, min(T1 + 6, len(pol))):
    x = pol[i]
    print(f"  {i:3d}: p={x['protocol']} tps={x['tps']:6.0f} "
          f"out={x['outage']} sw={x['switched']} k={x['k']}")
