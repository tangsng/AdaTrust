"""Download real Ethereum mainnet arrival trace via public JSON-RPC (concurrent).

Builds an arrival-rate series (tx/s) from block timestamps and tx counts.
Output: <repo>/data/eth_trace.csv  (path relative to this file, so run it from
the project tree: python experiments/download_trace.py)
"""
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ENDPOINTS = [
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.flashbots.net",
    "https://cloudflare-eth.com",
    "https://eth.llamarpc.com",
]
OUT = Path(__file__).resolve().parents[2] / "data" / "eth_trace.csv"
N_BLOCKS = 1500          # ~5 h of mainnet
WORKERS = 12


def rpc_call(url, method, params, timeout=6):
    r = requests.post(url, json={"jsonrpc": "2.0", "id": 1,
                                 "method": method, "params": params},
                      timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if "result" not in j or j["result"] is None:
        raise ValueError(f"bad reply: {str(j)[:120]}")
    return j["result"]


def pick_endpoint():
    for url in ENDPOINTS:
        try:
            latest = int(rpc_call(url, "eth_blockNumber", []), 16)
            print(f"[ok] endpoint {url}, latest {latest}", flush=True)
            return url, latest
        except Exception as e:
            print(f"[fail] {url}: {type(e).__name__} {str(e)[:80]}", flush=True)
    return None, None


def fetch_block(n):
    for url in ENDPOINTS[:2]:
        try:
            blk = rpc_call(url, "eth_getBlockByNumber", [hex(n), False])
            return n, int(blk["timestamp"], 16), len(blk["transactions"])
        except Exception:
            continue
    return None


def main():
    url, latest = pick_endpoint()
    if url is None:
        print("ALL_ENDPOINTS_FAILED", flush=True)
        sys.exit(2)
    start = latest - N_BLOCKS
    t0 = time.time()
    rows = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in enumerate(ex.map(fetch_block, range(start, latest))):
            if res:
                rows[res[0]] = res
            if i % 150 == 0:
                print(f"  {i}/{N_BLOCKS} ({time.time()-t0:.0f}s)", flush=True)
    rows = [rows[n] for n in sorted(rows)]
    assert len(rows) > N_BLOCKS * 0.9, f"too many gaps: {len(rows)}"
    ts = [r[1] for r in rows]
    assert all(b >= a for a, b in zip(ts, ts[1:])), "non-monotonic timestamps"
    rates = []
    for i in range(10, len(rows)):
        dt = rows[i][1] - rows[i - 10][1]
        cnt = sum(r[2] for r in rows[i - 10:i])
        rates.append(cnt / max(dt, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("idx,rate_tps\n")
        for i, r in enumerate(rates):
            f.write(f"{i},{r:.4f}\n")
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"[done] {len(rates)} samples -> {OUT}", flush=True)
    print(f"[sha256] {h}", flush=True)
    print(f"[stats] min={min(rates):.1f} max={max(rates):.1f} "
          f"mean={sum(rates)/len(rates):.1f} tx/s", flush=True)


if __name__ == "__main__":
    main()
