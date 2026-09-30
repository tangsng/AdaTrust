#!/usr/bin/env python3
"""
adatrust_agent.py - AdaTrust real-chain controller (RQ1-Real, runs on AI-server).

Drives the trained PPO policy (calibrated simulator, sim-to-real) on the live
testbed: CometBFT testnet16 (Tendermint) + Besu QBFT x4 (PBFT family).

Per epoch (E blocks on the active chain):
  1. collect REAL evidence: CometBFT last_commit signatures -> good/bad/slash
  2. update real-chain TrustLayer (same Beta-posterior math as the simulator)
  3. assemble the 15-dim observation (identical layout to ConsensusEnv.observe)
  4. PPO inference -> (pid, B_idx, tau_idx, k_idx); HotStuff actions are
     realized on the Tendermint backend (nearest operational niche)
  5. hysteresis gate (same rule as policies.apply_gate)
  6. execute:
     - committee k on CometBFT: BayesElect top-k by trust score via val= txs
     - protocol switch: drain active chain -> move injector (hot standby)
     - B/tau: logged only (need chain restart; noted in paper)
  7. append epoch row -> CSV

Modes:
  --mode adatrust : full closed loop
  --mode static_tm : CometBFT only, no control (B5 baseline)
  --mode static_qbft: Besu only, no control (B1 baseline)
  --mode oracle   : rule switcher (trough QBFT small-k / peak TM large-k)

Run: python3 -u experiments/adatrust_agent.py --mode adatrust --duration 300 \
        --out runs/real/adatrust_run1.csv
"""
import argparse, csv, json, os, subprocess, sys, time, urllib.parse, urllib.request

CODE = "/home/user1/projects/code"
sys.path.insert(0, CODE)
import numpy as np  # noqa: E402
from adatrust.trust import TrustLayer  # noqa: E402

CB_RPCS = [f"http://localhost:{26700+i}" for i in range(16)]
BESU_RPC = "http://127.0.0.1:8540"
PROTO_DIR = "/home/user1/projects/prototype/testnet16"
E_BLOCKS = 10
TPS_REF, LAT_REF = 1200.0, 5.0
COMMITTEE_HINTS = [4, 8, 16, 32]
K_MIN, K_MAX = 4, 16          # real testbed: 16 CometBFT nodes
BURST_PERIOD = 60.0           # s, square-wave bursty load 150<->1200


# ---------------------------------------------------------------- rpc helpers
def cb_get(base, path, timeout=10):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode())

def cb_any(path, timeout=10):
    for base in CB_RPCS:
        try:
            return cb_get(base, path, timeout)
        except Exception:
            continue
    return None

def cb_height():
    d = cb_any('/status')
    if not d:
        return -1
    return int(d['result']['sync_info']['latest_block_height'])

def besu_height():
    """eth_blockNumber via raw JSON-RPC (-1 on failure)"""
    try:
        req = urllib.request.Request(
            BESU_RPC,
            data=json.dumps({'jsonrpc': '2.0', 'method': 'eth_blockNumber',
                             'params': [], 'id': 1}).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=8) as r:
            return int(json.loads(r.read().decode())['result'], 16)
    except Exception:
        return -1

def cb_block(h):
    return cb_any(f'/block?height={h}')

def cb_validators():
    d = cb_any('/validators')
    return d['result']['validators'] if d else []

_rr = [0]
def cb_send_val(pk_b64, power):
    tx = urllib.parse.quote(f'val=ed25519!{pk_b64}!{power}', safe='')
    for _ in range(16):
        base = CB_RPCS[_rr[0] % 16]; _rr[0] += 1
        try:
            cb_get(base, f'/broadcast_tx_async?tx=%22{tx}%22')
            return True
        except Exception:
            continue
    return False

def cb_pending_val_pks():
    """pubkeys that already have an unconfirmed val= tx in the mempool.
    Two val= txs for the same validator in ONE block produce duplicate
    ValidatorUpdates -> 'changing validator set: duplicate entry' -> chain
    halt (observed 2026-09-07, block 9518). Never send a second one."""
    import base64
    try:
        d = cb_any('/unconfirmed_txs?limit=1000')
        txs = d['result']['txs'] if d else []
    except Exception:
        return set()
    out = set()
    for t in txs:
        try:
            raw = base64.b64decode(t).decode(errors='ignore')
        except Exception:
            continue
        if raw.startswith('val=ed25519!'):
            parts = raw.split('!')
            if len(parts) >= 2:
                out.add(parts[1])
    return out

# ------------------------------------------------------- validator key mapping
def load_node_keys():
    """map validator address (hex, upper) -> (node_idx, pubkey_b64)"""
    m = {}
    for i in range(16):
        p = json.load(open(f'{PROTO_DIR}/node{i}/config/priv_validator_key.json'))
        m[p['address'].upper()] = (i, p['pub_key']['value'])
    return m

# -------------------------------------------------------------- trust tracker
class RealTrust:
    """TrustLayer fed by real CometBFT last_commit evidence."""
    def __init__(self, n=16):
        self.n = n
        self.tl = TrustLayer(n, lam=0.98, kappa=50.0, eta=2.0, theta_det=0.35,
                             rng=np.random.default_rng(0))
        self.addr2idx = load_node_keys()
        self.alarms = 0
        self._vset_cache = {}   # validators_hash -> ordered address list

    def _vset_order(self, h, vhash):
        """canonical validator-set order for commit-sig positional mapping.
        CometBFT v0.38 RPC returns ABSENT CommitSig with EMPTY
        validator_address (observed 2026-09-08) - the signatures array is
        POSITIONAL, aligned with the validator set order. We key the cache
        by block-H's validators_hash (= set(H)); block H's last_commit was
        signed by set(H-1), which differs only on reconfig blocks (2-3
        blocks per committee change) - acceptable single-block trust noise
        in exchange for 1 extra RPC per set change instead of per block."""
        if vhash in self._vset_cache:
            return self._vset_cache[vhash]
        order = []
        for path in (f'/validators?height={h}&per_page=100', '/validators?per_page=100'):
            d = cb_any(path)
            if d:
                order = [(v.get('address') or '').upper()
                         for v in d['result'].get('validators', [])]
                if order:
                    break
        self._vset_cache[vhash] = order
        if len(self._vset_cache) > 16:          # trim oldest
            for k in list(self._vset_cache)[:-16]:
                del self._vset_cache[k]
        return order

    def ingest_block(self, h):
        """one block's last_commit -> good/bad/slash vectors"""
        d = cb_block(h)
        if not d:
            return False
        good = np.zeros(self.n); bad = np.zeros(self.n); slash = np.zeros(self.n)
        try:
            blk = d['result']['block']
            sigs = blk['last_commit']['signatures']
            vhash = blk['header'].get('validators_hash', '')
        except (KeyError, TypeError):
            return False
        order = self._vset_order(h, vhash)
        for i, s in enumerate(sigs):
            addr = (s.get('validator_address') or '').upper()
            if not addr and i < len(order):     # ABSENT: positional fill
                addr = order[i]
            if addr not in self.addr2idx:
                continue
            idx = self.addr2idx[addr][0]
            flag = s.get('block_id_flag', 0)
            if flag == 2:      # BLOCK_ID_FLAG_COMMIT
                good[idx] += 1
            else:              # absent / nil
                bad[idx] += 1
        ev = d['result']['block'].get('evidence') or {}
        if ev.get('evidence'):
            for e in ev['evidence']:
                va = (e.get('validator', {}).get('address') or '').upper()
                if va in self.addr2idx:
                    slash[self.addr2idx[va][0]] += 1
                    self.alarms += 1
        self.tl.update(good, bad, slash)
        return True

# ------------------------------------------------------------------ injector
class Injector:
    """manages the tx blaster subprocess on the active chain (bursty load)"""
    def __init__(self, chain, workdir='/home/user1/runs/real/inj'):
        self.chain = chain
        self.workdir = workdir
        self.proc = None
        self.csv = None
        self._csv_offset = 0.0   # committed tx carried over from prior chain

    def start(self, duration=7200):
        self.stop()
        os.makedirs(self.workdir, exist_ok=True)
        self.csv = f'{self.workdir}/inj_{self.chain}_{int(time.time())}.csv'
        if self.chain == 'cometbft':
            cmd = ['python3', os.path.expanduser('~/uploads/tx_blaster.py'),
                   '--rpc', CB_RPCS[0], '--mode', 'bursty',
                   '--high', '1200', '--low', '150', '--period', str(BURST_PERIOD),
                   '--duration', str(duration), '--out', self.csv]
        else:
            cmd = ['python3', os.path.expanduser('~/uploads/web3_blaster.py'),
                   '--rpc', BESU_RPC, '--mode', 'bursty',
                   '--high', '1200', '--low', '150', '--period', str(BURST_PERIOD),
                   '--duration', str(duration), '--out', self.csv]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        # carry committed-tx total across chain switches
        self._csv_offset += self._cum_from_csv()
        self.proc = None
        self.csv = None      # prevent double-count on a second stop()

    def _cum_from_csv(self):
        if not self.csv or not os.path.exists(self.csv):
            return 0.0
        try:
            with open(self.csv) as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                return 0.0
            hdr = rows[0]
            i_bn = hdr.index('block_ntxs')
            return sum(float(r[i_bn]) for r in rows[1:]
                       if len(r) > i_bn and float(r[i_bn]) > 0)
        except Exception:
            return 0.0

    def cumulative_committed(self):
        """total committed tx since experiment start (switch-safe)"""
        return self._csv_offset + self._cum_from_csv()

    def latest(self):
        """(target_rate, achieved_block_tx_last_s, backlog) from blaster csv.

        Sticky fallback: a fresh blaster CSV has no rows for ~1s after a
        chain switch. Returning a fabricated low value (150/0/-1) here
        poisoned oracle/PPO decisions and caused switch thrash (empty-CSV
        -> lam=150 -> instant switch back). Return last real sample instead.
        """
        if not hasattr(self, '_last_good'):
            self._last_good = (150.0, 0, -1)
        if not self.csv or not os.path.exists(self.csv):
            return self._last_good
        try:
            with open(self.csv) as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                return self._last_good
            r = rows[-1]
            hdr = rows[0]
            d = dict(zip(hdr, r))
            rate = float(d.get('target_rate', 150))
            bn = float(d.get('block_ntxs', 0) or 0)
            backlog = float(d.get('mempool', d.get('pending', -1)) or -1)
            self._last_good = (rate, bn, backlog)
            return self._last_good
        except Exception:
            return self._last_good

# ------------------------------------------------------------ chain switching
def drain(chain, tbuf=20.0):
    """wait until active chain mempool/pending empties (or tbuf); return secs"""
    t0 = time.time()
    while time.time() - t0 < tbuf:
        try:
            if chain == 'cometbft':
                d = cb_any('/num_unconfirmed_txs')
                m = int(d['result']['n_txs']) if d else -1
            else:
                from web3 import Web3
                w3 = Web3(Web3.HTTPProvider(BESU_RPC,
                                            request_kwargs={'timeout': 10}))
                st = w3.provider.make_request('txpool_besuTransactions', {})
                m = len(st['result']) if isinstance(st.get('result'), list) else -1
            if m == 0:
                break
        except Exception:
            pass
        time.sleep(1)
    return time.time() - t0

# ------------------------------------------------------------------- agent
class AdaTrustReal:
    def __init__(self, mode, ppo_path, out, duration,
                 no_trust_obs=False, no_sizing=False, no_hyst=False,
                 restrict_pid=None, allow_stall=False):
        self.mode = mode
        self.out = out
        self.duration = duration
        # RQ4 ablation flags (deployment-time)
        self.no_trust_obs = no_trust_obs    # zero trust features in obs
        self.no_sizing = no_sizing          # skip Eq.9 k_safe, k_hint only
        self.no_hyst = no_hyst              # disable hysteresis/dwell
        # E4 (M10): pin the protocol head at deployment, matching the
        # training-time restriction of a parameter-only checkpoint (whose
        # protocol head is intentionally untrained and must not be read)
        self.restrict_pid = restrict_pid
        # E5 halt regime: tick empty epochs on a stalled backend instead of
        # dying on the 180 s stall watchdog (default off — protects all
        # non-E5 campaigns with the original abort behavior)
        self.allow_stall = allow_stall
        # stall-timing state (E5): reset only when height actually changes
        self._h_stall_t = time.time()
        self._h_prev = None
        self.chain = 'cometbft' if mode in ('adatrust', 'static_tm', 'oracle') \
            else 'besu'
        self.inj = Injector(self.chain)
        self.trust = RealTrust(16)
        self.agent = None
        if mode == 'adatrust':
            import torch
            from adatrust.ppo import PPO
            self.agent = PPO(obs_dim=15)   # +consecutive-zero-TPS counter (E5 v5)
            self.agent.net.load_state_dict(
                torch.load(ppo_path, map_location='cpu'))
            self.agent.net.eval()
            self._dev = next(self.agent.net.parameters()).device
        self.protocol = 2 if self.chain == 'cometbft' else 0  # TM=2, PBFT=0
        self.last_switch_ep = -10**9
        self.k_cur = 16
        self.k_pending, self.k_pending_n = None, 0   # committee-change dwell
        self._val_intended = {}     # pk -> (power, expiry_ts) sent-but-unseen
        self.hyst_margin, self.hyst_dwell = 0.05, 5
        self.k_dwell = 3            # epochs a new k must persist before acting
        if self.no_hyst:            # RQ4 ablation: no hysteresis/dwell
            self.hyst_margin, self.hyst_dwell, self.k_dwell = 0.0, 0, 1
        self.last_tps, self.last_lat = 0.0, 1.3
        self.last_lam = 150.0
        self._epoch_reset = False      # set by switch_to; resets height baseline
        self._last_switch_t = 0.0      # wall time of last protocol switch
        self.oracle_min_dwell = 15.0   # s; oracle may not re-switch sooner

    # ---------------- observation (identical layout to ConsensusEnv.observe)
    def observe(self):
        s = self.trust.tl.scores
        f_hat, _ = self.trust.tl.adv_mass()
        onehot = np.eye(3)[self.protocol]
        trust_feats = [0.0, 0.0, 0.0, 0.0] if self.no_trust_obs else \
            [s.mean(), s.var(), np.quantile(s, 0.1), f_hat]
        return np.concatenate([
            [min(self.last_lam / TPS_REF, 1.5)],
            [48.0 / 1024.0],
            [0.05 / 0.3, 0.2],                 # no netem in RQ1 main runs
            trust_feats,
            [min(float(self.trust.alarms), 1.0)],
            [min(self.last_tps / TPS_REF, 1.0),
             min(self.last_lat / LAT_REF, 1.0)],
            onehot,
            # E5 v5: consecutive ~0-TPS epochs (identical layout/semantics
            # to ConsensusEnv.observe — updated AFTER observe(), see loop)
            [min(getattr(self, '_zero_tps', 0) / 10.0, 1.0)],
        ]).astype(np.float32)

    # ---------------- actions
    def elect_committee(self, k):
        """BayesElect on the real chain: top-k trust scores keep power=1,
        others get power=0 (CometBFT side only).

        Safety (learned the hard way, 2026-09-07 chain halt at h2419):
        - k-dwell: a new target k must persist K_DWELL epochs before any
          val tx is sent (kills 16<->4 oscillation val-tx storms).
        - intention ledger: a sent val tx is remembered for 30 s; the
          chain view (/validators) lags ~2 heights, and re-removing an
          already-removed validator panics consensus
          ('failed to find validator ... to remove')."""
        if self.chain != 'cometbft':
            return
        k = int(np.clip(k, K_MIN, K_MAX))
        vals = cb_validators()
        # detection-triggered membership heal: if a CURRENT committee
        # member's trust score has collapsed (silence detected), re-elect
        # with the same k to swap the dead member out - the k==k_cur
        # early-return must not block recovery, and the k-dwell applies to
        # k CHANGES only, not to membership heals (RQ3, 2026-09-08).
        cur_nodes = set()
        for v in vals:
            a = (v.get('address') or '').upper()
            if a in self.trust.addr2idx:
                cur_nodes.add(self.trust.addr2idx[a][0])
        det = set(np.where(
            np.asarray(self.trust.tl.scores)
            < self.trust.tl.theta_det)[0].tolist())
        heal_needed = bool(det & cur_nodes)
        if k == self.k_cur and not heal_needed:
            self.k_pending, self.k_pending_n = None, 0
            return
        if k != self.k_cur:
            if self.k_pending == k:
                self.k_pending_n += 1
            else:
                self.k_pending, self.k_pending_n = k, 1
            if self.k_pending_n < self.k_dwell:
                return
            self.k_pending, self.k_pending_n = None, 0
        scores = np.asarray(self.trust.tl.scores)
        top = set(np.argsort(-scores)[:k].tolist())
        cur_pks = {v['pub_key']['value'] for v in vals}
        pending = cb_pending_val_pks()
        now = time.time()
        self._val_intended = {p: v for p, v in self._val_intended.items()
                              if v[1] > now}
        for idx, (addr, (i, pk)) in enumerate(self.trust.addr2idx.items()):
            if pk in pending:
                continue          # val tx for this pk already in flight
            want = 1 if i in top else 0
            if pk in self._val_intended:
                have = self._val_intended[pk][0]
            else:
                have = 1 if pk in cur_pks else 0
            if want != have:
                if cb_send_val(pk, want):
                    self._val_intended[pk] = (want, now + 30.0)
        self.k_cur = k

    def switch_to(self, pid):
        target = 'cometbft' if pid == 2 else 'besu'
        if target == self.chain:
            return 0.0
        print(f'  ** SWITCH {self.chain} -> {target} (drain first)', flush=True)
        self.inj.stop()
        d = drain(self.chain)
        self.chain = target
        self.protocol = pid
        self.inj.chain = target
        self.inj.start()
        self.last_switch_ep = self.ep
        self._last_switch_t = time.time()
        # the new leg must wait E_BLOCKS *fresh* blocks; block height advanced
        # while we were on the other chain, so without this reset the first
        # epoch ends in ~1s and the newborn blaster CSV is still empty
        self._epoch_reset = True
        return d

    # ---------------- decision
    def decide(self, obs):
        """returns (pid, k_hint, gated) according to mode"""
        if self.mode == 'static_tm':
            return 2, 16, False
        if self.mode == 'static_qbft':
            return 0, 4, False
        if self.mode == 'oracle':
            f_hat, _ = self.trust.tl.adv_mass()
            if time.time() - self._last_switch_t < self.oracle_min_dwell:
                return self.protocol, self.k_cur, False   # dwell: no thrash
            if self.last_lam > 600:          # peak: TM + full committee
                return 2, 16, False
            return 0, 4, False               # trough: QBFT
        # adatrust: PPO + gate. DETERMINISTIC argmax inference (deployment
        # standard): the trained protocol head is weakly peaked
        # [0.32, 0.08, 0.60]; stochastic act() sampled QBFT ~1/3 of epochs
        # and hysteresis stretched that to a 50/50 split (v3 root cause).
        import torch
        x = torch.as_tensor(obs, dtype=torch.float32,
                            device=self._dev).unsqueeze(0)
        with torch.no_grad():
            logits, v_t = self.agent.net(x)
        acts = [int(torch.argmax(lg, dim=-1).item()) for lg in logits]
        v = float(v_t.item())
        pid = int(acts[0])
        if self.restrict_pid is not None:
            pid = self.restrict_pid      # parameter-only deployment (E4)
        elif pid == 1:
            pid = 2    # HotStuff realized on Tendermint backend
        k_hint = COMMITTEE_HINTS[int(acts[3])]
        gated = False
        if pid != self.protocol:
            # hysteresis: margin from value head + dwell (v from act(),
            # already on the right device)
            if abs(v) > self.hyst_margin and \
               (self.ep - self.last_switch_ep) >= self.hyst_dwell:
                gated = True
            else:
                pid = self.protocol
        return pid, k_hint, gated

    # ---------------- main loop
    def run(self):
        try:
            self._run_inner()
        finally:
            self.inj.stop()      # never leave a zombie blaster behind
            total_tx = self.inj.cumulative_committed()
            print(f'CLEANUP done, total committed={total_tx:.0f}', flush=True)

    def _run_inner(self):
        self.inj.start()
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        self.ep = 0
        t0 = time.time()
        h_last = cb_height() if self.chain == 'cometbft' else None
        h_change_t = time.time()      # stall watchdog: chain must keep moving
        besu_bn, besu_change_t = 0, time.time()   # besu liveness watchdog
        with open(self.out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ep', 't', 'chain', 'k', 'lam', 'block_tx', 'backlog',
                        'cum_tx', 'trust_mean', 'f_hat', 'alarms', 'pid_dec',
                        'k_dec', 'gated', 'drain_s', 'n_det'])
            while time.time() - t0 < self.duration:
                time.sleep(1.0)
                if self.chain == 'cometbft':
                    h = cb_height()
                    if self._epoch_reset:
                        # fresh leg: count E_BLOCKS from *now*, not from the
                        # height the chain reached while on the other chain
                        h_last = h
                        h_change_t = time.time()
                        self._h_stall_t = time.time()
                        self._h_prev = h
                        self._epoch_reset = False
                        continue
                    # stall timer resets ONLY when the height actually
                    # changes; h_change_t (used by the abort watchdog) also
                    # refreshes on partial progress, which is why the old
                    # empty-tick gate never fired when the chain froze a few
                    # blocks past h_last (v13 r1: 18 epochs / 600 s)
                    if h != getattr(self, '_h_prev', h):
                        self._h_stall_t = time.time()
                        self._h_prev = h
                    if h > (h_last if h_last is not None else -1):
                        h_change_t = time.time()
                    elif time.time() - h_change_t > 180 and \
                            not self.allow_stall:
                        raise SystemExit(
                            f'FATAL: chain stalled at height {h_last} '
                            'for >180s, aborting run')
                    if h < 0 or h_last is None or h < h_last + E_BLOCKS:
                        if self.allow_stall and h_last is not None and \
                                h >= 0 and \
                                time.time() - self._h_stall_t > 15:
                            # E5 halt regime (--allow-stall): the incumbent
                            # backend finalizes nothing, so tick an EMPTY
                            # epoch (~15 s pace) instead of blocking — the
                            # cumulative-stall counter climbs, the controller
                            # sees obs[14] rise and escapes. Reset the stall
                            # timer so empty epochs are paced, not spammed.
                            self._h_stall_t = time.time()
                        else:
                            continue
                    else:
                        for hh in range(h_last + 1, h + 1):
                            self.trust.ingest_block(hh)
                        h_last = h
                else:
                    bn = besu_height()
                    if self._epoch_reset:
                        besu_bn, besu_change_t = bn, time.time()
                        self._epoch_reset = False
                    if bn > besu_bn:
                        besu_bn, besu_change_t = bn, time.time()
                    elif time.time() - besu_change_t > 300:
                        raise SystemExit(
                            f'FATAL: besu stalled at block {besu_bn} '
                            'for >300s, aborting run')
                    time.sleep(E_BLOCKS)     # QBFT side: fixed epoch length
                self.ep += 1
                lam, btx, backlog = self.inj.latest()
                self.last_lam = lam
                self.last_tps = btx          # last-second committed (approx)
                cum = self.inj.cumulative_committed()
                s = self.trust.tl.scores
                f_hat, _ = self.trust.tl.adv_mass()
                n_det = int((s < self.trust.tl.theta_det).sum())
                obs = self.observe()
                # E5 v5/v12: the death signal is a STALL of the cumulative
                # committed counter, updated AFTER observe() (one-epoch-lag,
                # matching ConsensusEnv.run_epoch). Two deployment lessons:
                # (i) the per-second btx reading flickers -1/0 on a HEALTHY
                # chain (injector CSV lag; r1 2026-09-15: false escape at
                # ep 6 while cum_tx grew 13.9k->41.6k) — only a cumulative
                # stall is a robust halt signal; (ii) the counter must not
                # run during warm-up (no service yet -> trivially stalled),
                # hence the _seen_svc latch. In sim the niche either serves
                # (>1 TPS) or is down (0), so stall-counting matches the
                # trained input distribution.
                stalled = cum <= getattr(self, '_last_cum', -1.0)
                self._last_cum = cum
                if cum > 0:
                    self._seen_svc = True
                if getattr(self, '_seen_svc', False):
                    self._zero_tps = getattr(self, '_zero_tps', 0) + 1 \
                        if stalled else 0
                else:
                    self._zero_tps = 0
                pid, k_hint, gated = self.decide(obs)
                drain_s = 0.0
                if self.mode == 'adatrust':
                    # adv_mass is trust-WEIGHT based and self-deflates once
                    # detection works (detected node's weight -> 0 -> f_hat
                    # -> 0); committee sizing needs the detected COUNT
                    # fraction (RQ3 f_hat blindness, 2026-09-08)
                    f_eff = max(float(f_hat), n_det / 16.0)
                    k_safe = TrustLayer.committee_size(
                        max(f_eff, 1e-4), K_MIN, K_MAX, 1e-3)
                    if self.no_sizing:          # RQ4 ablation: k_hint only
                        self.elect_committee(min(k_hint, K_MAX))
                    else:
                        self.elect_committee(max(k_safe, min(k_hint, K_MAX)))
                    if pid != self.protocol and gated:
                        drain_s = self.switch_to(pid)
                elif self.mode == 'oracle':
                    if pid != self.protocol:
                        drain_s = self.switch_to(pid)
                w.writerow([self.ep, round(time.time() - t0, 1), self.chain,
                            self.k_cur, lam, btx, backlog,
                            round(self.inj.cumulative_committed(), 0),
                            round(float(s.mean()), 4), round(float(f_hat), 4),
                            self.trust.alarms, pid, k_hint, int(gated),
                            round(drain_s, 2), n_det])
                f.flush()
                print(f'ep={self.ep} chain={self.chain} k={self.k_cur} '
                      f'lam={lam:.0f} btx={btx:.0f} f_hat={f_hat:.3f} '
                      f'n_det={n_det} '
                      f'pid_dec={pid} gated={int(gated)}', flush=True)
        self.inj.stop()
        total_tx = self.inj.cumulative_committed()
        elapsed = time.time() - t0
        print(f'done -> {self.out}', flush=True)
        print(f'TOTAL committed_tx={total_tx:.0f} elapsed={elapsed:.1f}s '
              f'mean_tps={total_tx/max(elapsed,1e-9):.2f}', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['adatrust', 'static_tm', 'static_qbft',
                                       'oracle'], required=True)
    ap.add_argument('--ppo', default=CODE + '/experiments/runs/ppo.pt')
    ap.add_argument('--duration', type=float, default=300)
    ap.add_argument('--out', required=True)
    ap.add_argument('--port0', type=int, default=26700,
                    help='first RPC port of the target testnet '
                         '(26700=testnet16, 27700=testnet16b)')
    ap.add_argument('--proto-dir',
                    default='/home/user1/projects/prototype/testnet16')
    ap.add_argument('--burst-period', type=float, default=0.0,
                    help='override bursty load square-wave period (s)')
    # RQ4 deployment-time ablation flags
    ap.add_argument('--no-trust-obs', action='store_true')
    ap.add_argument('--no-sizing', action='store_true')
    ap.add_argument('--no-hyst', action='store_true')
    ap.add_argument('--allow-stall', action='store_true',
                    help='E5 halt regime: tick empty epochs on a stalled '
                         'backend instead of aborting after 180 s')
    ap.add_argument('--restrict-pid', type=int, default=None,
                    help='pin protocol head at deployment (E4 parameter-only)')
    args = ap.parse_args()
    if args.port0 != 26700:
        CB_RPCS[:] = [f"http://localhost:{args.port0 + i}" for i in range(16)]
    if args.proto_dir:
        global PROTO_DIR
        PROTO_DIR = args.proto_dir
    if args.burst_period > 0:
        global BURST_PERIOD
        BURST_PERIOD = args.burst_period
    AdaTrustReal(args.mode, args.ppo, args.out, args.duration,
                 no_trust_obs=args.no_trust_obs, no_sizing=args.no_sizing,
                 no_hyst=args.no_hyst, restrict_pid=args.restrict_pid,
                 allow_stall=args.allow_stall).run()

if __name__ == '__main__':
    main()
