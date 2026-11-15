import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ueba.metrics import evaluate, stable_ties, threshold_at_rate
from ueba.statistics import weighted_rank_curves, cluster_role_index


def test_prospective_threshold_ties_and_undefined_recall():
    scores = np.array([1.]*98 + [2., 2.])
    assert threshold_at_rate(scores) == 2.
    assert (scores > threshold_at_rate(scores)).sum() == 0
    frame = pd.DataFrame(dict(user=["a", "b"], date=pd.to_datetime(["2020-01-01"]*2),
                               role=["r", "s"], label=[0, 0]))
    result, _ = evaluate(frame, [1, 2], 1.5, stable_ties(frame))
    assert np.isnan(result["recall"])
    assert np.isnan(result["pr_auc"])
    assert result["fpr"] == .5


def test_weighted_bootstrap_curves_match_explicit_cluster_repetition():
    scores = np.array([4., 3., 3., 2., 1.])
    y = np.array([1, 0, 1, 0, 1])
    weights = np.array([2, 1, 3, 0, 2])
    expanded_y, expanded_s = np.repeat(y, weights), np.repeat(scores, weights)
    result = weighted_rank_curves(y, scores, weights)
    np.testing.assert_allclose(result["pr_auc"], average_precision_score(expanded_y, expanded_s))
    np.testing.assert_allclose(result["roc_auc"], roc_auc_score(expanded_y, expanded_s))
    assert result["recall_at_1pct"] == 1/7


def test_zero_weight_leading_tie_and_empty_class():
    out = weighted_rank_curves([0, 1, 0], np.array([3., 3., 1.]), [0, 1, 1])
    assert out["pr_auc"] == 1.
    assert out["roc_auc"] == 1.
    assert np.isnan(weighted_rank_curves([0, 0], np.array([2., 1.]), [1, 1])["recall_at_1pct"])


def test_cluster_role_index_does_not_overflow_categorical_int16():
    index = cluster_role_index(np.array([0, 867, 999], dtype=np.int16), [0, 41, 41], 42)
    np.testing.assert_array_equal(index, [0, 36455, 41999])
    assert index.dtype == np.int64
