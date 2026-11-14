"""Threshold and ranking metrics with explicit undefined-class handling."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_rate(scores, rate=.01):
    scores = np.asarray(scores, float)
    if len(scores) == 0 or not np.isfinite(scores).all() or not 0 < rate < 1:
        raise ValueError("Finite nonempty scores and valid rate required")
    return float(np.quantile(scores, 1-rate, method="higher"))


def stable_ties(frame):
    return np.array([hashlib.sha256(f"{u}|{pd.Timestamp(d).date()}".encode()).hexdigest()
                     for u, d in zip(frame.user, frame.date)])


def ranking_order(scores, tie_keys):
    return np.lexsort((tie_keys, -np.asarray(scores, float)))


def evaluate(frame, scores, threshold, tie_keys):
    y = frame.label.to_numpy(bool)
    s = np.asarray(scores, float)
    predicted = s > threshold
    tp = int((predicted & y).sum())
    fp = int((predicted & ~y).sum())
    positive, negative = int(y.sum()), int((~y).sum())
    precision = tp / (tp+fp) if tp+fp else 0.
    recall = tp / positive if positive else np.nan
    result = dict(n=len(y), positive=positive, negative=negative,
                  alerts=int(predicted.sum()), tp=tp, fp=fp,
                  fpr=fp/negative if negative else np.nan, recall=recall,
                  precision=precision, f1=2*tp/(positive+tp+fp) if positive+tp+fp else np.nan,
                  alert_rate=float(predicted.mean()), false_alarms_per_1000_days=1000*fp/len(y),
                  pr_auc=float(average_precision_score(y, s)) if positive else np.nan,
                  roc_auc=float(roc_auc_score(y, s)) if positive and negative else np.nan)
    order = ranking_order(s, tie_keys)
    for rate, suffix in [(.005, "0.5pct"), (.01, "1pct"), (.02, "2pct")]:
        k = max(1, int(np.ceil(rate*len(y))))
        hits = int(y[order[:k]].sum())
        result[f"k_{suffix}"] = k
        result[f"precision_at_{suffix}"] = hits/k
        result[f"recall_at_{suffix}"] = hits/positive if positive else np.nan
    role = frame[["role", "label"]].copy()
    role["fp"] = predicted & ~y
    role["negative"] = ~y
    role["tp"] = predicted & y
    grouped = role.groupby("role").agg(negative=("negative", "sum"), fp=("fp", "sum"),
                                        positive=("label", "sum"), tp=("tp", "sum"))
    grouped["fpr"] = grouped.fp / grouped.negative.replace(0, np.nan)
    grouped["recall"] = grouped.tp / grouped.positive.replace(0, np.nan)
    result["max_role_fpr"] = float(grouped.fpr.max())
    result["max_role_fpr_n100"] = float(grouped.loc[grouped.negative >= 100, "fpr"].max())
    result["role_fpr_std"] = float(grouped.fpr.std())
    return result, grouped.reset_index()
