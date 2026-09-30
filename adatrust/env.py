"""ConsensusEnv: epoch-granularity adaptive-consensus environment (paper Sec. 3.5, 4.4).

One env step = one epoch of E block rounds:
  1. committee election (mode-dependent)
  2. E simulated block rounds under the chosen protocol + params
  3. trust update from round evidence
  4. metrics + reward (Eq. 6)

Baselines are expressed as fixed policies that call step() with their own
action/mode conventions (see experiments/policies.py).

Common random numbers (CRN, revision R1, reviewer comment #9): exogenous
randomness (network-delay drift, workload arrivals) is drawn from a dedicated
stream rng_env that is consumed identically by every scheme sharing a seed, so
paired comparisons are free of drift confounds. Endogenous draws (per-member
evidence, election) use separate streams whose consumption may legitimately
differ across schemes.
"""
import numpy as np
from .protocols import PROTOCOLS, PBFT, BLOCK_SIZES_KB, TIMEOUTS_S, COMMITTEE_HINTS
from .trust import TrustLayer
from .adversary import Adversary

E_BLOCKS = 10
TX_SIZE_B = 48            # kvstore-scale tx (real-chain CometBFT, R1 calibration)
SIG_VERIFY_S = 2e-3       # BLS-scale verification cost (s) — makes O(k) CPU matter
MSG_BYTES = 256
T_BUF = 20.0          # backlog buffer horizon (s of load) before latency explodes

# --- R1 real-chain calibration (measured 2026-09-07 on H100 testbed) ---
GAMMA_LAT = 1.2         # sim->real latency factor (sim underestimates 15-30%)
BLOCK_PERIOD_S = 1.0    # hard block period (timeout_commit / blockperiodseconds)
BURST_LOW, BURST_HIGH = 150.0, 1200.0   # bursty load, real capacity scale
COLLAPSE_BACKLOG = 0.8  # EVM txpool nonce-gap collapse threshold (PBFT niche)
COLLAPSE_FACTOR = 0.1   # throughput multiplier once collapsed (measured ~0.07)


class ConsensusEnv:
    def __init__(self, n=64, f_ratio=0.10, attack="A1", schedule="static",
                 workload="bursty", trace=None, seed=0, committee_mode="trust",
                 hysteresis=(0.05, 5), weights=(1.0, 1.0, 1.0, 1.0),
                 lam=0.98, kappa=50.0, eta=2.0, theta_det=0.35,
                 k_min=4, k_max=64, eps=1e-3, tps_ref=1200.0,
                 outage=None, outage_script=None):
        # CRN streams: spawn independent sub-generators from the run seed.
        _ss = np.random.SeedSequence(seed)
        self.rng_env, self.rng_ev, self.rng_el = (
            np.random.default_rng(s) for s in _ss.spawn(3))
        self.rng = self.rng_ev                        # legacy alias (endogenous)
        self.n, self.f_ratio = n, f_ratio
        self.committee_mode = committee_mode          # 'trust'|'full'|'static_trust'
        self.hyst_margin, self.hyst_dwell = hysteresis
        self.w = weights                              # (w1..w4) of Eq. 6
        self.k_min, self.k_max, self.eps = k_min, k_max, eps
        self.trust = TrustLayer(n, lam, kappa, eta, theta_det, rng=self.rng_el)
        self.adv = Adversary(n, f_ratio, attack, schedule, seed=seed)
        self.workload, self.trace = workload, trace
        self.t = 0
        self.protocol = PBFT                          # start on PBFT
        self.last_switch = -10**9
        self.backlog = 0.0                            # in transactions
        self.delay_mean = 0.05                        # s, drifts over time
        self.tx_size = TX_SIZE_B
        self._static_k = None
        self._zero_tps = 0  # consecutive epochs with ~0 realized TPS (obs[14])
        # E5 correlated-failure (backend-outage) regime: an entire protocol
        # niche goes down (service = 0) while the others stay alive, modelling
        # the real small-n case (e.g. 6/16 validators stopped -> CometBFT
        # halts; the Besu QBFT backend is a separate failure domain).
        #   outage        : (p_on, p_off) per-epoch Markov switching per niche
        #                   (drawn from rng_env -> CRN-paired across schemes)
        #   outage_script : list of (pid, t0, t1) deterministic windows (eval)
        self.outage = outage
        self.outage_script = outage_script or []
        self.outage_down = [False, False, False]
        # fixed normalization references for reward (reproducible, no data leakage)
        self.tps_ref, self.lat_ref = tps_ref, 5.0

    # ---------------- workload ----------------
    def arrival_rate(self):
        """Transactions per second demanded this epoch."""
        if self.workload == "real" and self.trace is not None:
            return float(self.trace[self.t % len(self.trace)])
        if self.workload == "constant":
            return 800.0
        if self.workload == "poisson":
            return float(self.rng_env.poisson(800.0))
        # bursty: Markov-modulated, 8x peak/off ratio calibrated so that the
        # peak saturates the Tendermint niche (~1000 TPS real capacity) while
        # the trough fits even the PBFT niche (~82 TPS) — R1 real-chain scale
        if not hasattr(self, "_burst_state"):
            self._burst_state = 0
        p_switch = 0.1
        if self.rng_env.random() < p_switch:
            self._burst_state = 1 - self._burst_state
        base = BURST_HIGH if self._burst_state else BURST_LOW
        return float(self.rng_env.normal(base, 0.1 * base))

    # ---------------- election ----------------
    def elect_committee(self, k_hint):
        if self.committee_mode == "full":
            return np.arange(self.n), self.n
        if self.committee_mode == "static_trust":
            if self._static_k is None:
                f0 = max(self.adv.f / self.n, 1e-4)
                self._static_k = int(3 * f0 * self.n) + 1      # classic 3f+1 once
                self._static_k = int(np.clip(self._static_k, self.k_min, self.k_max))
            k = self._static_k
        else:                                                  # BayesElect
            f_hat, _ = self.trust.adv_mass()
            k_safe = TrustLayer.committee_size(f_hat, self.k_min, self.k_max, self.eps)
            k = max(k_safe, k_hint)
        k = int(np.clip(k, self.k_min, min(self.k_max, self.n)))
        return self.trust.elect(k, self.rng), k

    # ---------------- epoch simulation ----------------
    def step(self, action):
        """action = (protocol_id, B_idx, tau_idx, k_idx) already hysteresis-gated
        by the caller (policies.apply_gate). Returns (state, reward, info)."""
        pid, bi, ti, ki = action
        switched = pid != self.protocol
        in_handover = switched or (self.t - self.last_switch <= 1)
        if switched:
            self.last_switch = self.t
        self.protocol = pid
        proto = PROTOCOLS[pid]
        B_kb, tau = BLOCK_SIZES_KB[bi], TIMEOUTS_S[ti]
        k_hint = COMMITTEE_HINTS[ki]

        self.adv.maybe_retarget(self.t, self.trust.scores)
        committee, k = self.elect_committee(k_hint)
        byz_in = [i for i in committee if i in self.adv.byz]
        b = len(byz_in)
        p_silent, p_equiv = self.adv.epoch_behavior(self.t, in_handover)

        # network drift (non-stationarity) -- exogenous stream (CRN-paired)
        self.delay_mean = float(np.clip(self.delay_mean + self.rng_env.normal(0, 0.01),
                                        0.02, 0.30))
        lam_t = self.arrival_rate()

        committed, latencies, msgs = 0, [], 0
        stalls = alarms = slashes = 0
        good = np.zeros(self.n); bad = np.zeros(self.n); slash = np.zeros(self.n)

        round_lat = proto.round_latency(k, self.delay_mean, SIG_VERIFY_S, B_kb)
        capacity = (B_kb * 1024 / self.tx_size) * proto.capacity_mult   # tx per block
        quorum = proto.quorum(k)

        # E5: backend-outage update (Markov via rng_env, CRN-paired; scripted
        # windows are deterministic). A down niche finalizes nothing.
        if self.outage is not None:
            p_on, p_off = self.outage
            for j in range(3):
                if self.outage_down[j]:
                    if self.rng_env.random() < p_off:
                        self.outage_down[j] = False
                elif self.rng_env.random() < p_on:
                    self.outage_down[j] = True
        for spid, t0, t1 in self.outage_script:
            self.outage_down[spid] = (t0 <= self.t < t1)
        niche_down = self.outage_down[pid]

        for _ in range(E_BLOCKS):
            silent = sum(1 for i in byz_in if self.rng_ev.random() < p_silent)
            equiv = [i for i in byz_in if self.rng_ev.random() < p_equiv]
            # evidence for committee members
            for i in committee:
                if i in self.adv.byz:
                    if i in equiv:
                        bad[i] += 1
                        if self.rng_ev.random() < 0.5:         # proof found
                            slash[i] += 1
                            slashes += 1
                    elif self.rng_ev.random() < p_silent:
                        bad[i] += 1
                    else:
                        good[i] += 1
                else:
                    if self.rng_ev.random() < 0.01:            # honest noise
                        bad[i] += 1
                    else:
                        good[i] += 1
            # liveness: backend outage (whole niche down) or not enough
            # votes -> timeout stall, nothing finalized
            if niche_down or k - silent < quorum:
                stalls += 1
                lat = tau
                block_tx = 0
            else:
                # R1 calibration: gamma factor + hard 1s block period
                lat = round_lat * GAMMA_LAT + BLOCK_PERIOD_S
                service_cap = capacity
                if pid == PBFT and (self.backlog + lam_t * lat) > capacity:
                    # EVM-niche nonce-gap collapse (measured on Besu QBFT:
                    # offered > service rate -> throughput drops ~10x)
                    service_cap = capacity * COLLAPSE_FACTOR
                block_tx = int(min(service_cap, self.backlog + lam_t * lat))
            # safety: equivocation beyond the 1/3 BFT tolerance
            if equiv and b >= (k + 2) // 3:
                alarms += 1
            # finite buffer: backlog capped at T_BUF seconds of service capacity
            # (excess arrivals are dropped, standard finite-queue model)
            self.backlog = min(max(0.0, self.backlog + lam_t * lat - block_tx),
                               T_BUF * max(capacity, 1.0))
            queue_pen = 1.0 + self.backlog / max(capacity, 1.0)
            latencies.append(lat * queue_pen + proto.finality_blocks * lat)
            committed += block_tx
            msgs += proto.msgs_per_round(k)

        self.trust.update(good, bad, slash)
        dr, fpr = self.trust.detection_stats(self.adv.byz)

        epoch_time = max(sum(latencies), 1e-9)
        tps = committed / epoch_time
        lat_mean = float(np.mean(latencies))
        lat_p95 = float(np.percentile(latencies, 95))
        penalty = 0.4 * slashes + 1.0 * (1 if alarms else 0) + 0.2 * stalls
        w1, w2, w3, w4 = self.w
        reward = (w1 * min(tps / self.tps_ref, 1.0)
                  - w2 * min(lat_mean / self.lat_ref, 1.0)
                  - w3 * penalty
                  - w4 * (1.0 if switched else 0.0) * 0.5)

        self.t += 1
        info = dict(tps=tps, lat=lat_mean, lat_p95=lat_p95, k=k, b=b,
                    protocol=pid, msgs=msgs, dr=dr, fpr=fpr, alarms=alarms,
                    stalls=stalls, slashes=slashes, switched=int(switched),
                    lam=lam_t, delay=self.delay_mean,
                    outage=int(niche_down))
        return self.observe(), reward, info

    def observe(self):
        s = self.trust.scores
        f_hat, _ = self.trust.adv_mass()
        onehot = np.eye(3)[self.protocol]
        return np.concatenate([
            [min(getattr(self, "_last_lam", 800.0) / self.tps_ref, 1.5)],
            [self.tx_size / 1024.0],
            [self.delay_mean / 0.3, 0.2],
            [s.mean(), s.var(), np.quantile(s, 0.1), f_hat],
            [min(getattr(self, "_last_alarm", 0.0), 1.0)],
            [min(getattr(self, "_last_tps", 0.0) / self.tps_ref, 1.0),
             min(getattr(self, "_last_lat", 0.0) / self.lat_ref, 1.0)],
            onehot,
            # E5 v5: explicit outage evidence — consecutive ~0-TPS epochs
            # (normalized). Without it the controller cannot tell "backend
            # down" from "I should avoid this protocol", and collapses to
            # unconditional niche avoidance (v4: TM probability -> 0).
            [min(self._zero_tps / 10.0, 1.0)],
        ]).astype(np.float32)

    def run_epoch(self, action):
        s, r, info = self.step(action)
        self._last_tps, self._last_lat = info["tps"], info["lat"]
        self._last_lam, self._last_alarm = info["lam"], info["alarms"]
        self._zero_tps = self._zero_tps + 1 if info["tps"] < 1.0 else 0
        return s, r, info
