"""Train AdaptSwitch PPO agent (Alg. 2) and, for B3, a PBFT-restricted tuner.

Usage:
  python train.py --iters 200 --horizon 256 --seed 0 --out runs/ppo.pt
  python train.py --restrict-pid 0 --out runs/ppo_pbft.pt   # B3 tuner (PBFT=0)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adatrust.env import ConsensusEnv
from adatrust.ppo import PPO, compute_gae
from adatrust.policies import apply_gate


def train(iters, horizon, seed, restrict_pid, use_gate, no_sw_cost, out, ent=0.01,
          w1=3.0, tps_ref=1200.0, no_trust_obs=False, committee="trust",
          outage=None, outage_perm=None, bc_iters=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # mixed-workload training: alternate rollouts between the bursty synthetic
    # workload and the real Ethereum trace so the policy serves both regimes
    trace = None
    tpath = Path(__file__).resolve().parents[2] / "data" / "eth_trace.csv"
    if tpath.exists():
        import pandas as pd
        trace = pd.read_csv(tpath)["rate_tps"].to_numpy()
        trace = trace / trace.mean() * 800.0
    def _mk(wl, tr, sd):
        return ConsensusEnv(n=64, f_ratio=0.15, attack="A1", schedule="adaptive",
                            workload=wl, trace=tr, seed=sd,
                            committee_mode=committee,
                            hysteresis=(0.05, 5) if use_gate else (0.0, 1),
                            weights=(w1, 1.0, 1.0, 0.0) if no_sw_cost
                            else (w1, 1.0, 1.0, 1.0),
                            tps_ref=tps_ref, outage=outage)
    envs = [_mk("bursty", None, seed)]
    if trace is not None:
        envs.append(_mk("real", trace, seed + 500))
    def _mask(o):
        o = np.asarray(o, dtype=np.float32).copy()
        if no_trust_obs:      # RQ4 ablation: zero trust feature block (4:8)
            o[4:8] = 0.0
        return o
    obses = [_mask(e.observe()) for e in envs]
    env, obs = envs[0], obses[0]
    agent = PPO(obs_dim=len(obs), ent=ent)

    def _sched_outage(e):
        """E5 v4+ curriculum: with prob p_perm, schedule a Tendermint-niche
        outage at a random onset lasting to the END of this rollout
        (permanent within the episode, absolute-time scripted window).
        Recoverable Markov outages taught "wait it out" (v1-v3 all
        NEVER-escape); only a non-recovering outage makes escaping the
        unique reward-positive response."""
        if outage_perm is None:
            return
        p_perm, o_lo, o_hi = outage_perm
        e.outage_script = []
        # CRITICAL: reset the stale niche-down flags as well. Clearing only
        # the script left outage_down[2]==True from the previous outage
        # rollout, permanently killing TM in the training env (this single
        # bug poisoned generations v4-v7).
        e.outage_down = [False, False, False]
        if np.random.random() < p_perm:
            on = int(np.random.randint(o_lo, o_hi + 1))
            e.outage_script = [(2, e.t + on, e.t + horizon)]

    t0 = time.time()
    if bc_iters > 0:
        # E5 v6/v7: teacher-forced warmup. Teacher (privileged): TM while the
        # TM niche serves; PBFT while TM is down (scripted state or zero-TPS
        # counter >= 3). Anchors the conditional escape policy BEFORE PPO
        # fine-tuning, so the protocol head never saturates on a switchless
        # attractor.
        for it in range(1, bc_iters + 1):
            ei = it % len(envs)
            env, obs = envs[ei], obses[ei]
            _sched_outage(env)
            obs_b, pid_b = [], []
            for _ in range(horizon):
                acts, logp, v = agent.act(obs)
                # Teacher v3 (privileged, FULL action): escape to the PBFT
                # slot (the only genuinely separate failure domain on the
                # real testbed — Besu QBFT; HS shares the CometBFT backend)
                # and STAY there while TM remains down. Full action (128KB
                # blocks, 1s timeout, k=4 hint) keeps PBFT legs serving
                # (~82 TPS) so the zero-counter resets after escape; the
                # v2 half-random action left the untrained block head on
                # tiny blocks -> PBFT tarpit -> "on PBFT -> PBFT" self-lock
                # (v8: BC nets output PBFT 0.99 everywhere).
                tm_dead = env.outage_down[2] or env._zero_tps >= 3
                acts[0] = 0 if tm_dead else 2                # PBFT=0, TM=2
                acts[1], acts[2], acts[3] = 2, 1, 0          # 128KB, 1s, k=4
                if restrict_pid is not None:
                    acts[0] = restrict_pid
                if use_gate:
                    margin = abs(v) * 0.1 + \
                        (0.06 if acts[0] != env.protocol else 0.0)
                    gated, _ = apply_gate(env, tuple(acts), margin)
                else:
                    gated = tuple(acts)
                obs_b.append(obs); pid_b.append(acts[0])
                nxt, r, info = env.run_epoch(gated)
                obs = _mask(nxt)
            obses[ei] = obs
            # Counter-sweep augmentation (v10/v11): the (cnt>=3 -> PBFT)
            # escape trigger occurs exactly ONCE per outage rollout (~45
            # positives vs ~4.4k TM labels, and never beyond cnt=0.3).
            # Replicate a SUBSET of states with the counter swept over the
            # escape region, labelled PBFT — globally consistent with the
            # teacher rule. v10 swept ALL states (4:1 PBFT majority ->
            # net collapsed to PBFT-always); v11 balances classes ~1:1 and
            # triples the BC epochs so the cnt boundary actually forms.
            obs_aug, pid_aug = [], []
            idx = np.random.choice(len(obs_b),
                                   size=min(64, len(obs_b)), replace=False)
            for i in idx:
                for cval in (0.3, 0.5, 0.7, 1.0):
                    o2 = np.array(obs_b[i], dtype=np.float32, copy=True)
                    o2[14] = cval
                    obs_aug.append(o2); pid_aug.append(0)
            agent.bc_update(obs_b + obs_aug, pid_b + pid_aug, epochs=12)
            if it % 20 == 0:
                print(f"bc iter {it:4d} | tps {info['tps']:.0f} | "
                      f"proto {info['protocol']} | out {info['outage']} | "
                      f"elapsed {time.time()-t0:.0f}s", flush=True)
        bc_out = out.replace(".pt", "_bc.pt")
        Path(bc_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.net.state_dict(), bc_out)
        print(f"[saved-bc] {bc_out}", flush=True)

    for it in range(1, iters + 1):
        ei = it % len(envs)                    # alternate workload per iteration
        env, obs = envs[ei], obses[ei]
        _sched_outage(env)
        buf = {"obs": [], "acts": [], "logp": [], "adv": [], "ret": []}
        rewards, values, dones = [], [], []
        for _ in range(horizon):
            acts, logp, v = agent.act(obs)
            if restrict_pid is not None:
                acts[0] = restrict_pid
            if use_gate:
                margin = abs(v) * 0.1 + (0.06 if acts[0] != env.protocol else 0.0)
                gated, _ = apply_gate(env, tuple(acts), margin)
            else:
                gated = tuple(acts)
            nxt, r, info = env.run_epoch(gated)
            buf["obs"].append(obs); buf["acts"].append(acts)
            buf["logp"].append(logp)
            rewards.append(r); values.append(v); dones.append(0.0)
            obs = _mask(nxt)
        obses[ei] = obs
        adv, ret = compute_gae(np.array(rewards), values, dones,
                               agent.gamma, agent.lam)
        buf["adv"], buf["ret"] = adv, ret
        agent.update(buf)
        if it % 20 == 0:
            print(f"iter {it:4d} | ret {np.mean(rewards):+.3f} | "
                  f"tps {info['tps']:.0f} | lat {info['lat']:.3f} | "
                  f"k {info['k']} | b {info['b']} | dr {info['dr']:.2f} | "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.net.state_dict(), out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--restrict-pid", type=int, default=None)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-sw-cost", action="store_true",
                    help="ablation A4: remove switching-cost term (w4=0)")
    ap.add_argument("--no-trust-obs", action="store_true",
                    help="RQ4 ablation: zero trust features (obs[4:8])")
    ap.add_argument("--committee", default="trust",
                    choices=["trust", "full", "static_trust"],
                    help="RQ4 ablation: 'full' = no adaptive sizing")
    ap.add_argument("--ent", type=float, default=0.01,
                    help="entropy coefficient (exploration)")
    ap.add_argument("--out", default="runs/ppo.pt")
    ap.add_argument("--outage", nargs=2, type=float, default=None,
                    metavar=("P_ON", "P_OFF"),
                    help="E5: per-epoch Markov backend-outage probabilities "
                         "(per protocol niche, CRN-paired)")
    ap.add_argument("--outage-perm", nargs=3, type=float, default=None,
                    metavar=("P_PERM", "ONSET_LO", "ONSET_HI"),
                    help="E5 v4: per-rollout prob of a permanent (to rollout "
                         "end) Tendermint-niche outage with onset ~ U(lo,hi)")
    ap.add_argument("--bc-iters", type=int, default=0,
                    help="E5 v6: teacher-forced warmup iterations before PPO "
                         "(teacher: TM healthy / HS when zero-TPS count>=3); "
                         "warmup checkpoint saved as <out>_bc.pt")
    a = ap.parse_args()
    train(a.iters, a.horizon, a.seed, a.restrict_pid, not a.no_gate,
          a.no_sw_cost, a.out, a.ent, no_trust_obs=a.no_trust_obs,
          committee=a.committee, outage=tuple(a.outage) if a.outage else None,
          outage_perm=tuple(a.outage_perm) if a.outage_perm else None,
          bc_iters=a.bc_iters)
