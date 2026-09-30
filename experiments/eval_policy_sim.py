#!/usr/bin/env python3
"""eval_policy_sim.py - probe ppo.pt chain-choice probabilities in SIM.
Answers: is the 50/50 Besu affinity a policy defect or a sim-to-real gap?
Run on H100: python3 eval_policy_sim.py
"""
import sys, os, collections
sys.path.insert(0, os.path.expanduser('~/projects/code'))
import numpy as np
import torch

from adatrust.env import ConsensusEnv
from adatrust.ppo import PPO

env = ConsensusEnv(n=16, workload='bursty', attack='A1')
env.protocol = 2   # deployment-faithful init (real agent starts on TM)
ppo = PPO(obs_dim=15)
CKPT = sys.argv[1] if len(sys.argv) > 1 else '~/projects/code/experiments/runs/ppo.pt'
ppo.net.load_state_dict(torch.load(os.path.expanduser(CKPT), map_location='cpu'))
print('CKPT:', CKPT)
ppo.net.eval()

obs = env.observe()
tally = collections.Counter()
prob_hi, prob_lo = [], []
DEV = next(ppo.net.parameters()).device
for ep in range(600):
    x = torch.as_tensor(obs, dtype=torch.float32, device=DEV).unsqueeze(0)
    with torch.no_grad():
        logits, v = ppo.net(x)
    probs = [torch.softmax(lg, dim=-1).squeeze(0).cpu().numpy()
             for lg in logits]
    p_pid = probs[0]           # protocol head
    lam = getattr(env, '_last_lam', 800.0)
    phase = 'HI' if lam > 600 else 'LO'
    pid_argmax = int(np.argmax(p_pid))
    (prob_hi if phase == 'HI' else prob_lo).append(p_pid)
    # act like the real agent: SAMPLE
    acts, _, _ = ppo.act(obs)
    pid = int(acts[0])
    if pid == 1:
        pid = 2
    tally[(phase, pid, pid_argmax)] += 1
    obs, r, info = env.run_epoch([pid_argmax, 0, 0, 0])

print('=== (phase, sampled_pid, argmax_pid): count ===')
for k in sorted(tally):
    print(k, tally[k])
ph = np.mean(prob_hi, axis=0) if prob_hi else None
pl = np.mean(prob_lo, axis=0) if prob_lo else None
print('mean protocol probs HI:', np.round(ph, 3) if ph is not None else None)
print('mean protocol probs LO:', np.round(pl, 3) if pl is not None else None)
print('action head count:', len(probs), 'head sizes:',
      [len(p) for p in probs])
