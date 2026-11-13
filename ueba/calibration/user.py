import numpy as np
from .base import anomaly_score


class UserCalibration:
    """M2: user empirical tail, with explicit minimum-support fallback."""
    def __init__(self, min_history=5):
        self.min_history = min_history

    def transform(self, raw_scores, components):
        c = components
        return anomaly_score(np.where(c["n_user"] >= self.min_history, c["p_user"], c["p_global"]))
