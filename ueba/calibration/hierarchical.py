import numpy as np
from .base import anomaly_score


class HierarchicalCalibration:
    """M4 and prespecified ablations; no automatic conformal guarantee is claimed."""
    def __init__(self, k_user=30, k_role=200, use_user=True, use_role=True, fixed_weights=False):
        if k_user <= 0 or k_role <= 0:
            raise ValueError("Positive shrinkage constants required")
        self.k_user, self.k_role = k_user, k_role
        self.use_user, self.use_role = use_user, use_role
        self.fixed_weights = fixed_weights

    def probability(self, c):
        nu, nr = c["n_user"], c["n_role"]
        if self.fixed_weights:
            wu, wr = .5 * (nu > 0), .5 * (nr > 0)
        else:
            wu, wr = nu / (nu+self.k_user), nr / (nr+self.k_role)
        if not self.use_user:
            wu = np.zeros_like(wu)
        if not self.use_role:
            wr = np.zeros_like(wr)
        return wu*c["p_user"] + (1-wu)*(wr*c["p_role"] + (1-wr)*c["p_global"])

    def transform(self, raw_scores, components):
        return anomaly_score(self.probability(components))
