import numpy as np
import pandas as pd
import pytest

from ueba.calibration import (GlobalScore, RoleCalibration, UserCalibration,
                              EqualCalibration, HierarchicalCalibration)
from ueba.calibration.base import Profiles, upper_tail


def test_tail_ties_and_extremes():
    np.testing.assert_allclose(upper_tail([1, 2, 2, 4], [0, 2, 5]), [1, .8, .2])
    np.testing.assert_allclose(upper_tail([], [0, 1]), [1, 1])
    with pytest.raises(ValueError):
        upper_tail([float("nan")], [1])


def test_frozen_profiles_fallback_and_order():
    cal = pd.DataFrame(dict(user=["A", "A", "B"], role=["R", "R", "S"],
                            date=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
                            score=[1., 2., 4.]))
    profile = Profiles().fit(cal)
    target = pd.DataFrame(dict(user=["C", "A"], role=["T", "R"],
                               date=pd.to_datetime(["2020-01-03"]*2), score=[3., 3.]), index=[8, 2])
    c = profile.components(target)
    np.testing.assert_allclose(c["p_global"], [.5, .5])
    np.testing.assert_allclose(c["p_user"], [.5, 1/3])
    np.testing.assert_array_equal(c["n_user"], [0, 2])
    np.testing.assert_allclose(HierarchicalCalibration().probability(c)[0], .5)
    np.testing.assert_allclose(RoleCalibration().transform(target.score, c), -np.log([.5, .5]))
    np.testing.assert_allclose(UserCalibration().transform(target.score, c), -np.log([.5, .5]))
    with pytest.raises(ValueError, match="strictly after"):
        profile.components(cal)


def test_hierarchy_arithmetic_and_ablation():
    c = {"p_user": np.array([.1]), "p_role": np.array([.4]), "p_global": np.array([.8]),
         "n_user": np.array([30]), "n_role": np.array([200])}
    np.testing.assert_allclose(HierarchicalCalibration().probability(c), [.35])
    np.testing.assert_allclose(HierarchicalCalibration(use_user=False).probability(c), [.6])
    np.testing.assert_allclose(HierarchicalCalibration(use_role=False).probability(c), [.45])
    np.testing.assert_allclose(EqualCalibration().transform([9], c), -np.log([.25]))
    np.testing.assert_allclose(GlobalScore().transform([9], c), [9])
