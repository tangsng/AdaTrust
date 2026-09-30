"""diag_e5_counter.py - E5 v6 forensics: dump per-epoch zero-TPS counter,
protocol probabilities, chosen protocol, and realized TPS across a scripted
TM-outage window, to determine whether (a) the counter climbs in eval and
(b) the trained net responds to it.
Usage: python3 diag_e5_counter.py CKPT [t0] [t1] [seed]
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adatrust.env import ConsensusEnv
from adatrust.ppo import PPO
from adatrust.policies import apply_gate

CKPT = sys.argv[1]
T0 = int(sys.argv[2]) if len(sys.argv) > 2 else 60
T1 = int(sys.argv[3]) if len(sys.argv) > 3 else 120
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7

script = [(2, T0, T1)]
env = ConsensusEnv(n=64, f_ratio=0.15, attack="A1", schedule="adaptive",
                   workload="bursty", seed=SEED,
                   committee_mode="trust", hysteresis=(0.05, 5),
                   weights=(3.0, 1.0, 1.0, 1.0),
                   outage_script=script)
env.protocol = 2   # deployment-faithful init (real agent starts on TM)
ppo = PPO(obs_dim=15)
ppo.net.load_state_dict(torch.load(os.path.expanduser(CKPT),
                                   map_location='cpu'))
ppo.net.eval()

obs = env.observe()
print("ep: cnt obs14 | probs[PBFT,HS,TM] | pid(gated) | tps out")
for ep in range(T1 + 10):
    acts, logp, v = ppo.act(obs)
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _ = ppo.net(x)
    probs = torch.softmax(logits[0], dim=-1).squeeze(0).numpy()
    margin = abs(v) * 0.1 + (0.06 if acts[0] != env.protocol else 0.0)
    gated, _ = apply_gate(env, tuple(acts), margin)
    nxt, r, info = env.run_epoch(gated)
    if T0 - 8 <= ep <= T1 + 5:
        print(f"{ep:4d}: {env._zero_tps:3d} {obs[14]:.2f} | "
              f"[{probs[0]:.3f} {probs[1]:.3f} {probs[2]:.3f}] | "
              f"p{gated[0]} | tps={info['tps']:5.0f} out={info['outage']}")
    obs = nxt
