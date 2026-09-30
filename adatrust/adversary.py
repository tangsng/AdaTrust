"""Adversary model: corruption schedules and attack classes A1-A4 (paper Sec. 3.2)."""
import numpy as np


class Adversary:
    """Controls which validators are Byzantine and how they behave each epoch.

    schedule: 'static' (fixed at genesis) | 'adaptive' (re-targets every retarget_every
              epochs, preferring high-trust honest validators)
    attack:   'none' | 'A1' equivocation | 'A2' on-off silence | 'A3' collusion
              | 'A4' switching-aware (intensity doubles during handover)
    ratio:    f/n
    """

    def __init__(self, n, ratio, attack="A1", schedule="static", seed=0,
                 retarget_every=25):
        self.n = n
        self.ratio = ratio
        self.f = int(round(ratio * n))
        self.attack = attack
        self.schedule = schedule
        self.retarget_every = retarget_every
        self.rng = np.random.default_rng(seed)
        self.byz = set(self.rng.choice(n, size=self.f, replace=False).tolist())

    def maybe_retarget(self, epoch, trust_scores):
        if self.schedule != "adaptive" or epoch == 0 or epoch % self.retarget_every != 0:
            return
        honest = [i for i in range(self.n) if i not in self.byz]
        honest.sort(key=lambda i: -trust_scores[i])     # attack the most trusted
        new = set(honest[: self.f])
        self.byz = new

    def epoch_behavior(self, epoch, in_handover):
        """Return per-Byzantine behavior probabilities for this epoch:
        p_silent, p_equiv. On-off attack modulates by epoch cycle; A4 boosts
        intensity inside the handover window."""
        p_silent, p_equiv = 0.0, 0.0
        if self.attack == "A1":
            p_equiv = 0.5
        elif self.attack == "A2":
            off = (epoch % 25) >= 20            # 20 honest epochs / 5 silent
            p_silent = 1.0 if off else 0.0
        elif self.attack == "A3":
            p_silent = 0.8                      # coordinated silence at pivotal rounds
        elif self.attack == "A4":
            p_silent, p_equiv = 0.3, 0.3
        if self.attack == "A4" and in_handover:
            p_silent = min(1.0, 2 * p_silent)
            p_equiv = min(1.0, 2 * p_equiv)
        return p_silent, p_equiv
