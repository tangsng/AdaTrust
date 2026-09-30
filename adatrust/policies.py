"""Baseline and ablation policies + the hysteresis gate (paper Sec. 4.3, Eq. 11)."""
import numpy as np
from .protocols import PBFT, HOTSTUFF, TENDERMINT


def apply_gate(env, proposed, value_margin):
    """Hysteresis gate: accept protocol change only if margin>d_h and dwell ok."""
    margin, dwell = env.hyst_margin, env.hyst_dwell
    pid = proposed[0]
    if pid != env.protocol:
        if value_margin > margin and (env.t - env.last_switch) >= dwell:
            return proposed, True
        return (env.protocol, proposed[1], proposed[2], proposed[3]), False
    return proposed, False


class StaticPolicy:
    """B1/B2: fixed protocol, default params, full-set committee."""
    def __init__(self, pid):
        self.pid = pid

    def act(self, env, obs):
        return (self.pid, 1, 1, 2), False      # 512KB, 1.0s, hint 16 (unused)


class StaticTrustPolicy:
    """B4: trust-elected committee of genesis-fixed size running PBFT."""
    def act(self, env, obs):
        return (PBFT, 1, 1, 2), False


class TrainedPolicy:
    """Wraps a PPO agent; optional protocol restriction (B3: PBFT-only tuning)."""
    def __init__(self, agent, restrict_pid=None, use_gate=True):
        self.agent = agent
        self.restrict_pid = restrict_pid
        self.use_gate = use_gate

    def act(self, env, obs):
        acts, _, _ = self.agent.act(obs)
        if self.restrict_pid is not None:
            acts[0] = self.restrict_pid
        proposed = tuple(acts)
        if not self.use_gate:
            return proposed, proposed[0] != env.protocol
        return apply_gate(env, proposed, value_margin=self._margin(obs, proposed, env))

    def _margin(self, obs, proposed, env):
        # approximate V(s,a*)-V(s,cur) with advantage proxy: use |v| scaled sign-free
        import torch
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, v = self.agent.net(x.to(next(self.agent.net.parameters()).device))
        return float(abs(v.item())) * 0.1 + (0.06 if proposed[0] != env.protocol else 0.0)


class GreedyOraclePolicy:
    """Clairvoyant-ish heuristic used only as a reference row in RQ1 tables:
    Tendermint when f_hat~0 & low delay; PBFT under attack; HotStuff at large k."""
    def act(self, env, obs):
        f_hat, _ = env.trust.adv_mass()
        if f_hat < 0.02 and env.delay_mean < 0.08:
            pid = TENDERMINT
        elif env.n >= 64:
            pid = HOTSTUFF
        else:
            pid = PBFT
        proposed = (pid, 1, 1, 2)
        return apply_gate(env, proposed, value_margin=1.0)
