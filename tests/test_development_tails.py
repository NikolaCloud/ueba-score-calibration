import numpy as np
import pandas as pd
import pytest

from ueba.development.tails import ExtrapolatedProfiles,causal_mean,combine


def test_tail_matches_reference_knots_and_extrapolates():
    reference=np.array([1.,2.,2.,4.])
    values=np.array([1.,2.,4.,5.,6.])
    actual=ExtrapolatedProfiles.log_tail(reference,values,2.)
    np.testing.assert_allclose(actual[:3],np.log([1.,.8,.4]))
    np.testing.assert_allclose(np.diff(actual[2:]),[-.5,-.5])
    assert np.isfinite(actual).all()


def test_missing_role_and_history_date_guard():
    cal=pd.DataFrame(dict(user=['u']*4,role=['a']*4,date=pd.date_range('2020-01-01',periods=4),score=[1.,1.,1.,1.]))
    model=ExtrapolatedProfiles().fit(cal)
    target=pd.DataFrame(dict(user=['u','new'],role=['b','b'],date=[pd.Timestamp('2020-01-06')]*2,score=[2.,2.]))
    c=model.components(target)
    np.testing.assert_array_equal(c['n_user'],[0,0])
    np.testing.assert_allclose(combine(c),-c['log_global'])
    assert np.isfinite(combine(c)).all()
    with pytest.raises(ValueError,match='follow frozen'):
        model.components(cal)


def test_accumulation_is_causal_calendar_based_and_user_separated():
    frame=pd.DataFrame(dict(user=['a','a','b','a','a'],date=pd.to_datetime(['2020-01-01','2020-01-03','2020-01-03','2020-01-08','2020-01-09'])))
    scores=np.array([1.,3.,100.,8.,9.])
    result=causal_mean(frame,scores)
    np.testing.assert_allclose(result,[1.,2.,100.,5.5,20/3])
    changed=scores.copy();changed[-1]=10000
    np.testing.assert_allclose(causal_mean(frame,changed)[:-1],result[:-1])
    np.testing.assert_allclose(causal_mean(frame.iloc[:4],scores[:4]),result[:4])
