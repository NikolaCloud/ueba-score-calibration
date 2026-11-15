"""Paired user-cluster bootstrap, averaging seeds within each resampled cohort."""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .metrics import ranking_order, stable_ties

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["fpr", "recall", "precision", "f1", "false_alarms_per_1000_days",
           "max_role_fpr", "pr_auc", "roc_auc", "precision_at_0.5pct", "recall_at_0.5pct",
           "precision_at_1pct", "recall_at_1pct", "precision_at_2pct", "recall_at_2pct"]


def safe_ratio(a, b, zero=np.nan):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), zero, dtype=float),
                     where=np.asarray(b) != 0)


def cluster_role_index(user_index, role_index, n_roles):
    # pandas categorical codes may be int16: cast before multiplication, not after.
    return np.asarray(user_index, dtype=np.int64)*int(n_roles)+np.asarray(role_index, dtype=np.int64)


def weighted_rank_curves(y, sorted_scores, weights, rates=(.005, .01, .02)):
    """Metrics on a score-sorted cohort with integer bootstrap multiplicities.

    Ranking uses stable row order for ties. AP and ROC integrate score ties as groups,
    matching sklearn's definitions and preserving valid weights for repeated clusters.
    """
    y = np.asarray(y, bool)
    w = np.asarray(weights, float)
    positive = float(w @ y)
    negative = float(w @ ~y)
    cumulative = np.cumsum(w)
    tp = np.cumsum(w * y)
    output = {}
    total = cumulative[-1]
    for rate, suffix in zip(rates, ["0.5pct", "1pct", "2pct"]):
        k = max(1, int(np.ceil(rate * total)))
        idx = min(np.searchsorted(cumulative, k, side="left"), len(w)-1)
        hits = tp[idx] - (cumulative[idx]-k)*y[idx]
        output[f"precision_at_{suffix}"] = hits/k
        output[f"recall_at_{suffix}"] = hits/positive if positive else np.nan
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y)-1]
    group_tp, group_total = tp[endpoints], cumulative[endpoints]
    delta_tp = np.diff(np.r_[0., group_tp])
    output["pr_auc"] = float(np.sum(delta_tp * safe_ratio(group_tp, group_total, zero=0))/positive) if positive else np.nan
    if positive and negative:
        fp = group_total-group_tp
        output["roc_auc"] = float(np.trapezoid(np.r_[0., group_tp/positive], np.r_[0., fp/negative]))
    else:
        output["roc_auc"] = np.nan
    return output


def bootstrap_protocol(directory, n_bootstrap=1000):
    directory = Path(directory)
    seeds = sorted(directory.glob("seed_*/test_predictions.parquet"))
    if len(seeds) != 5:
        raise ValueError("All five seeds required before inference")
    first = pd.read_parquet(seeds[0])
    users = sorted(first.user.unique())
    user_index = pd.Categorical(first.user, categories=users).codes.astype(np.int64)
    role_index, role_names = pd.factorize(first.role, sort=True)
    n_users, n_roles = len(users), len(role_names)
    y = first.label.to_numpy(bool)
    rng = np.random.default_rng(20261102)
    weights = rng.multinomial(n_users, np.full(n_users, 1/n_users), size=n_bootstrap).astype(float)
    # Last row is the original cohort, using exactly the same calculation path.
    weights = np.vstack([weights, np.ones(n_users)])
    positive_user = np.bincount(user_index, weights=y, minlength=n_users)
    negative_user = np.bincount(user_index, weights=~y, minlength=n_users)
    positive, negative = weights @ positive_user, weights @ negative_user
    total = positive+negative
    pair_index = cluster_role_index(user_index, role_index, n_roles)
    negative_role = np.bincount(pair_index, weights=~y, minlength=n_users*n_roles).reshape(n_users, n_roles)
    role_denominator = weights @ negative_role
    ties = stable_ties(first)
    results = {m: {k: [] for k in METRICS} for m in ["M0", "M1", "M2", "M3", "M4"]}
    role_draws = {m: [] for m in results}
    for path in seeds:
        frame = pd.read_parquet(path)
        if not first[["user", "date", "role", "label"]].equals(frame[["user", "date", "role", "label"]]):
            raise ValueError("Nonpaired observations across seeds")
        for method in results:
            alert = frame[f"alert_{method}"].to_numpy(bool)
            tp_user = np.bincount(user_index, weights=alert & y, minlength=n_users)
            fp_user = np.bincount(user_index, weights=alert & ~y, minlength=n_users)
            tp, fp = weights @ tp_user, weights @ fp_user
            estimates = dict(fpr=safe_ratio(fp, negative), recall=safe_ratio(tp, positive),
                             precision=safe_ratio(tp, tp+fp, zero=0),
                             f1=safe_ratio(2*tp, positive+tp+fp),
                             false_alarms_per_1000_days=1000*fp/total)
            role_fp = np.bincount(pair_index, weights=alert & ~y,
                                  minlength=n_users*n_roles).reshape(n_users, n_roles)
            role_rates = safe_ratio(weights @ role_fp, role_denominator)
            estimates["max_role_fpr"] = np.nanmax(role_rates, axis=1)
            role_draws[method].append(role_rates)
            scores = frame[f"score_{method}"].to_numpy(float)
            order = ranking_order(scores, ties)
            rank_values = []
            for multiplicity in weights:
                rank_values.append(weighted_rank_curves(y[order], scores[order], multiplicity[user_index[order]]))
            ranked = pd.DataFrame(rank_values)
            for metric in METRICS:
                values = estimates[metric] if metric in estimates else ranked[metric].to_numpy()
                results[method][metric].append(values)
        print(f"Bootstrap completed {directory.name}/{path.parent.name}", flush=True)
    means = {method: {} for method in results}
    cis, paired, role_rows = [], [], []
    for method, metrics in results.items():
        for metric, arrays in metrics.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                values = np.nanmean(arrays, axis=0)
            means[method][metric] = values
            finite = values[:-1][np.isfinite(values[:-1])]
            lower, upper = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            cis.append(dict(method=method, metric=metric, estimate=values[-1], lower=lower, upper=upper,
                            finite_replicates=len(finite), total_replicates=n_bootstrap))
        role_values = np.nanmean(role_draws[method], axis=0)
        for i, role in enumerate(role_names):
            finite = role_values[:-1, i][np.isfinite(role_values[:-1, i])]
            lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            role_rows.append(dict(method=method, role=role, fpr=role_values[-1, i], lower=lo, upper=hi,
                                  finite_replicates=len(finite)))
    family = {"max_role_fpr", "recall", "recall_at_1pct"}
    for comparator in ["M0", "M3"]:
        for metric in METRICS:
            delta = means["M4"][metric] - means[comparator][metric]
            finite = delta[:-1][np.isfinite(delta[:-1])]
            lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            observed = delta[-1]
            p = (1+np.sum(np.abs(finite-observed) >= abs(observed))) / (len(finite)+1) if len(finite) else np.nan
            paired.append(dict(comparison=f"M4-{comparator}", metric=metric, difference=observed,
                               lower=lo, upper=hi, p_centered_bootstrap=p,
                               primary_family=metric in family, finite_replicates=len(finite)))
    paired = pd.DataFrame(paired)
    selected = paired.primary_family & paired.p_centered_bootstrap.notna()
    paired.loc[selected, "p_holm"] = multipletests(paired.loc[selected, "p_centered_bootstrap"], method="holm")[1]
    # Practical joint criterion, descriptive probability under the bootstrap, not
    # a posterior probability or a second uncorrected significance test.
    reduction = 1-safe_ratio(means["M4"]["max_role_fpr"], means["M0"]["max_role_fpr"])
    recall_change = means["M4"]["recall"]-means["M0"]["recall"]
    practical = dict(max_role_fpr_relative_reduction=float(reduction[-1]),
                     recall_change=float(recall_change[-1]),
                     point_criterion_met=bool(reduction[-1] >= .2 and recall_change[-1] >= -.02),
                     bootstrap_joint_fraction=float(np.mean((reduction[:-1] >= .2) & (recall_change[:-1] >= -.02))),
                     note="Bootstrap fraction is descriptive, not probability the hypothesis is true")
    pd.DataFrame(cis).to_csv(directory / "confidence_intervals.csv", index=False)
    paired.to_csv(directory / "paired_comparisons.csv", index=False)
    pd.DataFrame(role_rows).to_csv(directory / "role_confidence_intervals.csv", index=False)
    (directory / "practical_criterion.json").write_text(json.dumps(practical, indent=2), encoding="utf-8")
    diagnostics = dict(replicates=n_bootstrap, seed=20261102, users=n_users,
                       positive_users=int(np.sum(positive_user > 0)),
                       replicates_without_positive=int(np.sum(positive[:-1] == 0)),
                       algorithm="paired multinomial user clusters; mean across five seeds",
                       uncertainty_scope="test cohort conditional on trained models, frozen history and selected thresholds",
                       family="6 two-sided tests: M4-M0 and M4-M3 on max-role FPR, recall, Recall@1%; Holm")
    (directory / "bootstrap_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(practical, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    for directory in sorted((ROOT / "results").iterdir()):
        if directory.is_dir() and (directory / "metrics.csv").exists():
            bootstrap_protocol(directory, args.replicates)


if __name__ == "__main__":
    main()
