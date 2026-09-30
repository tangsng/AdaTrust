#!/usr/bin/env python3
"""
web3_blaster_e3.py (v4) - E3 fair-baseline load injector for Besu QBFT (M3).

Failure history:
  v1: prewarm 500x1 ETH > genesis balance.  Fixed: budget-scaled funding.
  v2: unbounded per-send threads + 30 s timeouts -> thread explosion.
  v3: fixed pool but ONE process -> GIL serializes sign_transaction
      (measured 7.3 ms/sign, RPC only 2.0 ms; 96 threads x 9.3 ms GIL
      rotation = the observed ~0.9 s/send, ~104 TPS hard cap per process).
  v4: SHARDED MULTIPROCESS injector. A single prewarm process funds 500
      accounts and dumps the keys to JSON; N shard processes (each with its
      own GIL, ~105 signs/s cap) load a strided subset of the keys and
      follow the same wall-clock ladder at target/N each. Total injection
      capacity ~ N x 105 TPS.

Ladder (global): 200 / 400 / 600 / 800 TPS, 150 s each (600 s total).
Modes:
  --prewarm-only --keys-out f.json : fund accounts, dump keys, exit
  --keys-file f.json --shard i N   : shard i of N runs the ladder
  (no keys-file)                   : single-process mode (v3 behaviour)
"""
import argparse
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

PRIV = "0x8f2a55949038a9610f50fb23b5883af3b4ecb3c3bb792cbcefbd1542c692be63"
LADDER = [(200.0, 150.0), (400.0, 150.0), (600.0, 150.0), (800.0, 150.0)]
N_SESSIONS = 4
MAX_WORKERS = 12
MAX_BACKLOG = 1000
SEND_TIMEOUT = 8


def make_rate_fn(ladder):
    edges, acc = [], 0.0
    for r, dur in ladder:
        acc += dur
        edges.append((acc, r))

    def rate_fn(t):
        for edge, r in edges:
            if t < edge:
                return r
        return edges[-1][1]
    return rate_fn


def prewarm(w3, fund, n, keys_out):
    """Fund n fresh accounts (150 ETH budget), dump keys to JSON."""
    per_acct_wei = w3.to_wei(150.0 / n, 'ether')
    accts = [w3.eth.account.create() for _ in range(n)]
    base = w3.eth.get_transaction_count(fund.address, 'pending')
    chain_id = w3.eth.chain_id
    for i, a in enumerate(accts):
        tx = {'to': a.address, 'value': per_acct_wei,
              'gas': 21000, 'gasPrice': 0, 'nonce': base + i,
              'chainId': chain_id}
        signed = fund.sign_transaction(tx)
        w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f'prewarm: {n} funding txs sent (nonce {base}..{base + n - 1})',
          flush=True)
    for _ in range(180):
        time.sleep(1)
        if w3.eth.get_transaction_count(fund.address,
                                        'latest') >= base + n:
            print('prewarm: confirmed', flush=True)
            break
    else:
        raise RuntimeError('prewarm not confirmed within 180 s')
    with open(keys_out, 'w') as f:
        json.dump([{'priv': a.key.hex(),
                    'addr': a.address} for a in accts], f)
    print(f'prewarm: keys dumped -> {keys_out}', flush=True)


class MultiInjector:
    def __init__(self, w3s, senders, rate_fn, stop_ev, counter):
        self.w3s, self.senders = w3s, senders
        self.rate_fn, self.stop_ev, self.counter = rate_fn, stop_ev, counter
        self.t0 = time.time()
        self.rr_send = 0
        self.rr_sess = 0
        self.pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.err_samples, self.err_lock = [], threading.Lock()

    def _note_err(self, e):
        with self.err_lock:
            if len(self.err_samples) < 5:
                msg = str(e)[:160]
                if msg not in self.err_samples:
                    self.err_samples.append(msg)
                    print(f'ERR-SAMPLE: {msg}', flush=True)

    def _send_one(self, s):
        w3 = self.w3s[self.rr_sess]
        self.rr_sess = (self.rr_sess + 1) % len(self.w3s)
        # per-sender lock: with a deep backlog the same sender can have
        # several queued tasks; without the lock two workers read the same
        # nonce -> duplicate txs ('Known transaction') and stale nonces
        # ('Nonce too low') - v5 saw 76% of attempts error this way while
        # the txs actually landed (committed 179k vs 27k counted successes)
        with s['lock']:
            try:
                tx = {'to': s['acct'].address, 'value': 1, 'gas': 21000,
                      'gasPrice': 0, 'nonce': s['nonce'],
                      'chainId': s['chain_id']}
                signed = s['acct'].sign_transaction(tx)
                w3.eth.send_raw_transaction(signed.raw_transaction)
                s['nonce'] += 1
                with self.counter['lock']:
                    self.counter['n'] += 1
            except Exception as e:
                msg = str(e).lower()
                if 'nonce' in msg or 'already known' in msg or \
                        'replacement' in msg:
                    try:
                        s['nonce'] = w3.eth.get_transaction_count(
                            s['acct'].address, 'pending')
                    except Exception:
                        pass
                self._note_err(e)
                with self.counter['lock']:
                    self.counter['err'] += 1

    def run(self):
        tick = 0.05
        while not self.stop_ev.is_set():
            r = self.rate_fn(time.time() - self.t0)
            per_tick = max(0.0, r * tick)
            n = int(per_tick)
            if random.random() < per_tick - n:
                n += 1
            backlog = self.pool._work_queue.qsize()
            if backlog + n > MAX_BACKLOG:
                drop = backlog + n - MAX_BACKLOG
                n -= drop
                with self.counter['lock']:
                    self.counter['drop'] += drop
            for _ in range(max(n, 0)):
                s = self.senders[self.rr_send]
                self.rr_send = (self.rr_send + 1) % len(self.senders)
                self.pool.submit(self._send_one, s)
            time.sleep(tick)

    def stop(self):
        self.pool.shutdown(wait=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rpc', default='http://127.0.0.1:8540')
    ap.add_argument('--accounts', type=int, default=500)
    ap.add_argument('--duration', type=float, default=600)
    ap.add_argument('--out', default=None,
                    help='output CSV (not needed with --prewarm-only)')
    ap.add_argument('--prewarm-only', action='store_true')
    ap.add_argument('--keys-out')
    ap.add_argument('--keys-file')
    ap.add_argument('--shard', type=int, default=-1,
                    help='shard index; requires --nshards')
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()

    w3 = Web3(Web3.HTTPProvider(args.rpc, request_kwargs={'timeout': 30}))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    assert w3.is_connected(), 'cannot connect to besu rpc'
    fund = w3.eth.account.from_key(PRIV)
    chain_id = w3.eth.chain_id

    if args.prewarm_only:
        assert args.keys_out, '--prewarm-only requires --keys-out'
        prewarm(w3, fund, args.accounts, args.keys_out)
        return

    if args.keys_file:
        with open(args.keys_file) as f:
            keys = json.load(f)
        if args.shard >= 0:
            keys = keys[args.shard::args.nshards]
        accts = [w3.eth.account.from_key(k['priv']) for k in keys]
    else:
        # single-process fallback: fund internally, keep keys in memory
        per = w3.to_wei(150.0 / args.accounts, 'ether')
        accts = [w3.eth.account.create() for _ in range(args.accounts)]
        base = w3.eth.get_transaction_count(fund.address, 'pending')
        for i, a in enumerate(accts):
            tx = {'to': a.address, 'value': per, 'gas': 21000,
                  'gasPrice': 0, 'nonce': base + i, 'chainId': chain_id}
            w3.eth.send_raw_transaction(
                fund.sign_transaction(tx).raw_transaction)
        for _ in range(180):
            time.sleep(1)
            if w3.eth.get_transaction_count(
                    fund.address, 'latest') >= base + args.accounts:
                break

    senders = [{'acct': a, 'chain_id': chain_id, 'lock': threading.Lock(),
                'nonce': w3.eth.get_transaction_count(a.address, 'pending')}
               for a in accts]
    print(f'shard {args.shard}/{args.nshards}: {len(senders)} senders, '
          f'height={w3.eth.block_number}', flush=True)

    # global ladder split across shards
    split = [(r / args.nshards, d) for r, d in LADDER]
    rate_fn = make_rate_fn(split)

    w3s = []
    for _ in range(N_SESSIONS):
        wi = Web3(Web3.HTTPProvider(
            args.rpc, request_kwargs={'timeout': SEND_TIMEOUT}))
        wi.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        w3s.append(wi)

    stop_ev = threading.Event()
    counter = {'n': 0, 'err': 0, 'drop': 0, 'lock': threading.Lock()}
    inj = MultiInjector(w3s, senders, rate_fn, stop_ev, counter)
    loop = threading.Thread(target=inj.run, daemon=True)
    loop.start()

    t0 = time.time()
    w3m = Web3(Web3.HTTPProvider(args.rpc, request_kwargs={'timeout': 30}))
    w3m.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    last_h = w3m.eth.block_number
    committed = 0
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'height', 'committed_cum', 'block_tx_last',
                    'pending', 'injected', 'errors', 'dropped',
                    'target_rate'])
        while time.time() - t0 < args.duration:
            time.sleep(1.0)
            t = round(time.time() - t0, 1)
            try:
                h = w3m.eth.block_number
                bn_last = 0
                for hh in range(last_h + 1, h + 1):
                    bn = len(w3m.eth.get_block(hh).transactions)
                    committed += bn
                    bn_last = bn
                last_h = h
                st = w3m.provider.make_request('txpool_besuTransactions', {})
                pend = len(st['result']) if isinstance(
                    st.get('result'), list) else -1
            except Exception:
                h, bn_last, pend = -1, -1, -1
            with counter['lock']:
                n, e, d = counter['n'], counter['err'], counter['drop']
            w.writerow([t, h, committed, bn_last, pend, n, e, d,
                        round(rate_fn(t) * args.nshards, 1)])
            f.flush()
    stop_ev.set()
    inj.stop()
    print(f"done shard {args.shard}: injected={counter['n']} "
          f"errors={counter['err']} dropped={counter['drop']} "
          f"committed={committed} -> {args.out}", flush=True)


if __name__ == '__main__':
    main()
