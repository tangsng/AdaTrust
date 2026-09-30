"""Protocol performance models for PBFT / HotStuff / Tendermint.

The protocol pool is restricted to Byzantine-fault-tolerant protocols only
(revision R1, reviewer comment #3): a CFT protocol such as Raft has no valid
operating point under a Byzantine adversary and is therefore removed.

Each protocol exposes a uniform interface used by ConsensusEnv:
  - quorum(k): votes needed to decide a block
  - round_latency(k, mean_delay, sig_verify_s): expected normal-case latency (s)
  - msgs_per_round(k): authenticated messages per decided block
  - byz_tolerance: max Byzantine ratio the protocol tolerates

Latency model (paper Sec. 3.1/5.3) — the pool spans three distinct niches:
  PBFT       : 3 message delays + CPU cost of O(k) signature verifications.
               Niche: lowest latency at small/moderate committees.
  HotStuff   : 4 delays (3-chain pipelined) + O(1) aggregated verification.
               Niche: large committees — constant CPU and linear messages.
  Tendermint : 3 message delays + O(k) vote verifications, gossip keeps
               messages at O(k log k); block-part gossip overlaps proposal
               dissemination with voting, yielding 1.5x effective block
               capacity, but store-and-forward gossip of large blocks adds
               a dissemination delay proportional to block size.
               Niche: throughput under high offered load.
CPU verification is what makes O(k^2)-message protocols degrade with k.
"""
import math
from dataclasses import dataclass

PBFT, HOTSTUFF, TENDERMINT = 0, 1, 2
PROTOCOL_NAMES = {PBFT: "PBFT", HOTSTUFF: "HotStuff", TENDERMINT: "Tendermint"}


@dataclass(frozen=True)
class Protocol:
    pid: int
    phases: float          # number of sequential network delays in normal case
    msg_complexity: str    # 'linear' | 'quadratic' | 'gossip'
    byz_tolerance: float   # max tolerated Byzantine ratio within committee
    finality_blocks: int   # blocks until a decision is irreversible
    capacity_mult: float = 1.0   # effective block-capacity multiplier

    def quorum(self, k: int) -> int:
        return (2 * k) // 3 + 1                    # all-BFT pool: 2f+1 of 3f+1

    def round_latency(self, k: int, mean_delay: float, sig_verify_s: float,
                      block_kb: float = 0.0) -> float:
        net = self.phases * mean_delay
        if self.pid == HOTSTUFF:
            cpu = 4 * sig_verify_s                 # aggregated/threshold signatures
        else:
            cpu = 2 * k * sig_verify_s             # PBFT: prepare+commit; TM: prevote+precommit
        if self.pid == TENDERMINT:
            net += 0.5 * mean_delay * (block_kb / 1024.0)  # gossip store-forward of block parts
        return net + cpu

    def msgs_per_round(self, k: int) -> int:
        if self.pid == HOTSTUFF:
            return 3 * k
        if self.pid == TENDERMINT:
            return 3 * k * max(1, math.ceil(math.log2(max(k, 2))))   # gossip fan-out
        return k + 2 * k * (k - 1)                # leader fan-out + all-to-all x2


PROTOCOLS = {
    # R1 calibration: the PBFT niche keeps its low-latency edge at small
    # committees but its effective throughput niche is much lower than
    # Tendermint's (Besu QBFT measured ~82 TPS vs CometBFT ~1000 TPS, the
    # gap partly from EVM execution overhead — discussed honestly in paper)
    PBFT: Protocol(PBFT, phases=3.0, msg_complexity="quadratic",
                   byz_tolerance=1 / 3, finality_blocks=1,
                   capacity_mult=0.15),
    HOTSTUFF: Protocol(HOTSTUFF, phases=4.0, msg_complexity="linear",
                       byz_tolerance=1 / 3, finality_blocks=3,
                       capacity_mult=1.0),
    TENDERMINT: Protocol(TENDERMINT, phases=3.0, msg_complexity="gossip",
                         byz_tolerance=1 / 3, finality_blocks=1,
                         capacity_mult=1.5),
}

# Discrete action grids (paper Sec. 4.3, factorized heads)
# R1 calibration: with 48B kvstore txs and a 1s block period, these sizes map
# to real-chain service rates — TM niche: ~170/~680/~2.7k/~10.9k tx per block
# (x1.5 capacity_mult), PBFT niche x0.15 — so the 150<->1200 bursty load
# straddles the niches exactly as on the measured testbed.
BLOCK_SIZES_KB = [8, 32, 128, 512]
TIMEOUTS_S = [0.5, 1.0, 2.0, 4.0]
COMMITTEE_HINTS = [4, 8, 16, 32]
