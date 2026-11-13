import numpy as np
from .base import anomaly_score


class RoleCalibration:
    """M1: role empirical tail, with explicit minimum-support fallback."""
    def __init__(self, min_history=30):
        self.min_history = min_history

    def transform(self, raw_scores, components):
        c = components
        return anomaly_score(np.where(c["n_role"] >= self.min_history, c["p_role"], c["p_global"]))
