"""Smoothed empirical upper-tail probabilities, including ties."""
import numpy as np
import pandas as pd


def upper_tail(reference, scores):
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("Nonfinite anomaly scores")
    return (1 + len(reference) - np.searchsorted(reference, values, side="left")) / (len(reference) + 1)


class Profiles:
    def __init__(self, max_user_history=None):
        if max_user_history is not None and max_user_history < 1:
            raise ValueError("History cap must be positive")
        self.max_user_history = max_user_history

    def fit(self, calibration):
        required = {"score", "user", "role", "date"}
        if not required <= set(calibration) or calibration.empty:
            raise ValueError("Nonempty dated calibration frame required")
        if calibration[["user", "date"]].duplicated().any():
            raise ValueError("Duplicate calibration user-day")
        self.last_date = calibration.date.max()
        self.global_scores = np.sort(calibration.score.to_numpy(float))
        self.roles = {str(k): np.sort(v.score.to_numpy(float)) for k, v in calibration.groupby("role")}
        user_history = calibration.sort_values("date")
        if self.max_user_history is not None:
            user_history = user_history.groupby("user", group_keys=False).tail(self.max_user_history)
        self.users = {str(k): np.sort(v.score.to_numpy(float)) for k, v in user_history.groupby("user")}
        if not np.isfinite(self.global_scores).all():
            raise ValueError("Nonfinite calibration")
        return self

    def components(self, targets):
        if len(targets) and targets.date.min() <= self.last_date:
            raise ValueError("Targets must be strictly after the frozen calibration block")
        scores = targets.score.to_numpy(float)
        pg = upper_tail(self.global_scores, scores)
        pr, pu = pg.copy(), pg.copy()
        nr, nu = np.zeros(len(scores), int), np.zeros(len(scores), int)
        for column, history, probabilities, counts in [
            ("role", self.roles, pr, nr), ("user", self.users, pu, nu)
        ]:
            for key, indices in targets.groupby(column, sort=False).indices.items():
                reference = history.get(str(key))
                if reference is not None:
                    probabilities[indices] = upper_tail(reference, scores[indices])
                    counts[indices] = len(reference)
        return dict(p_global=pg, p_role=pr, p_user=pu, n_role=nr, n_user=nu)


def anomaly_score(probability):
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1):
        raise ValueError("Tail probability must be in (0, 1]")
    return -np.log(p)
