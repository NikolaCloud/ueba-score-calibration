import numpy as np
import pandas as pd
import pytest

from ueba.development.contextual import ContinuousContext,geometric_score


def reference():
    return pd.DataFrame(dict(user=["a"]*10+["b"]*10,role=["r"]*20,
        date=list(pd.date_range("2020-01-01",periods=10))*2,
        score=np.r_[np.linspace(.1,.11,10),np.linspace(.6,.61,10)]))


def test_continuous_score_recovers_order_beyond_local_empirical_tail():
    model=ContinuousContext().fit(reference())
    targets=pd.DataFrame(dict(user=["a","a"],role=["r","r"],date=pd.to_datetime(["2020-02-01"]*2),score=[.2,.4]))
    values=model.transform(targets)
    assert np.isfinite(values).all() and values[1]>values[0]
    with pytest.raises(ValueError,match="follow"):
        model.transform(reference())


def test_unseen_role_uses_fallback_instead_of_old_person_profile():
    model=ContinuousContext().fit(reference())
    targets=pd.DataFrame(dict(user=["a","new","new"],role=["newrole","newrole","r"],
        date=pd.to_datetime(["2020-02-01"]*3),score=[.2,.2,.2]))
    values=model.transform(targets)
    assert np.isfinite(values).all() and values[0]==values[1]


def test_constant_history_is_finite_and_does_not_mutate():
    cal=reference().assign(score=.5)
    original=cal.copy(deep=True)
    model=ContinuousContext().fit(cal)
    target=cal.iloc[:2].copy().assign(date=pd.Timestamp("2020-02-01"),score=[.5,.7])
    assert np.isfinite(model.transform(target)).all()
    pd.testing.assert_frame_equal(cal,original)


def test_geometric_pooling_has_expected_surprisal_and_missing_profile_weights():
    components=dict(p_user=np.array([.01,.4]),p_role=np.array([.5,.4]),p_global=np.array([.8,.4]),
                    n_user=np.array([30,0]),n_role=np.array([200,0]))
    out=geometric_score(components)
    np.testing.assert_allclose(out[0],-.5*np.log(.01)-.25*np.log(.5)-.25*np.log(.8))
    np.testing.assert_allclose(out[1],-np.log(.4))
