"""Bayesian trust layer and BayesElect committee election (paper Sec. 3.3, 4.2)."""
import math
import numpy as np


class TrustLayer:
    def __init__(self, n, lam=0.98, kappa=50.0, eta=2.0, theta_det=0.35, rng=None):
        self.n = n
        self.lam = lam            # decay factor lambda_d  (Eq. 2)
        self.kappa = kappa        # slash magnitude        (Sec. 4.2)
        self.eta = eta            # weight sharpness eta
        self.theta_det = theta_det
        self.rng = rng or np.random.default_rng(0)
        self.alpha = np.ones(n)
        self.beta = np.ones(n)

    @property
    def scores(self):
        return self.alpha / (self.alpha + self.beta)          # Eq. (1)

    @property
    def weights(self):
        return np.power(self.scores, self.eta)

    def update(self, good, bad, slashes):
        """good/bad/slashes: per-validator counts over the epoch (Eq. 2 + slash)."""
        self.alpha += good
        self.beta += bad + self.kappa * slashes
        self.alpha = 1.0 + self.lam * (self.alpha - 1.0)
        self.beta = 1.0 + self.lam * (self.beta - 1.0)

    def adv_mass(self, true_byz=None):
        """Estimated effective adversarial weight f-hat (Sec. 4.2).
        If true_byz indices are given (simulator only), also return the TRUE mass."""
        w = self.weights
        detected = w[self.scores < self.theta_det].sum()
        f_hat = detected / max(w.sum(), 1e-12)
        if true_byz is None:
            return f_hat, None
        f_true = w[list(true_byz)].sum() / max(w.sum(), 1e-12)
        return f_hat, f_true

    @staticmethod
    def committee_size(f_eff, k_min=4, k_max=64, eps=1e-3):
        """Smallest k whose binomial corruption tail <= eps (Eq. 9)."""
        f_eff = min(max(f_eff, 1e-4), 0.49)
        for k in range(k_min, k_max + 1):
            thr = math.ceil(k / 3)
            # exact binomial tail Pr[X >= thr], X~Bin(k, f_eff)
            tail = 0.0
            log_p, log_q = math.log(f_eff), math.log(1 - f_eff)
            for j in range(thr, k + 1):
                tail += math.exp(math.lgamma(k + 1) - math.lgamma(j + 1)
                                 - math.lgamma(k - j + 1) + j * log_p + (k - j) * log_q)
                if tail > eps:
                    break
            if tail <= eps:
                return k
        return k_max

    def elect(self, k, rng):
        """Trust-weighted sampling without replacement (Alg. 1, line 9)."""
        w = self.weights / self.weights.sum()
        return rng.choice(self.n, size=min(k, self.n), replace=False, p=w)

    def detection_stats(self, byz_set):
        """Detection rate & false-positive rate against ground truth."""
        det = self.scores < self.theta_det
        byz = np.zeros(self.n, dtype=bool)
        if byz_set:
            byz[list(byz_set)] = True
        tp = int((det & byz).sum())
        fp = int((det & ~byz).sum())
        fn = int((~det & byz).sum())
        tn = int((~det & ~byz).sum())
        dr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return dr, fpr
