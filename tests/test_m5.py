import numpy as np
import pandas as pd
import pytest

from ueba.m5.model import AdaptiveContext
from ueba.m5.threshold import apply_guard,fit_guard,select_threshold


def toy():
    cal=pd.DataFrame([dict(user=u,role='r',date=d,score=float(s)) for u in ['u','v']
        for d,s in zip(pd.date_range('2020-01-01',periods=30),np.linspace(0,1,30))])
    val=pd.DataFrame([dict(user=u,role='r',date=d,score=float(s)) for d,s in
        zip(pd.date_range('2020-02-01',periods=12),[.3,.8,1.5,.4,1.1,1.,.8,.7,1.5,2.,1.,.3]) for u in ['u','v']])
    return cal,val


def test_m5_causal_prefix_batching_and_label_independence():
    cal,val=toy()
    all_scores,trace=AdaptiveContext().fit(cal).transform(val)
    prefix=val.loc[val.date.lt(pd.Timestamp('2020-02-07'))].copy()
    suffix=val.loc[val.date.ge(pd.Timestamp('2020-02-07'))].copy()
    model=AdaptiveContext().fit(cal)
    a,_=model.transform(prefix);b,_=model.transform(suffix)
    for name in a:
        np.testing.assert_allclose(np.r_[a[name],b[name]],all_scores[name],rtol=1e-12)
    altered=val.copy();altered['label']=np.arange(len(val))%2
    altered.loc[altered.date.ge('2020-02-07'),'score']=10000.
    c,_=AdaptiveContext().fit(cal).transform(altered)
    for name in a:np.testing.assert_allclose(c[name][:len(prefix)],a[name])
    assert np.all(all_scores['M5']>=trace.instant_evidence)
    assert np.all((trace.weight_user>=0)&(trace.weight_user<=1))
    with pytest.raises(ValueError,match='strictly'):
        model.transform(suffix)


def test_new_role_resets_history_and_current_score_is_not_in_reference():
    cal,val=toy()
    val=val.iloc[:2].copy();val['role']='unseen';val['score']=100.
    _,trace=AdaptiveContext().fit(cal).transform(val)
    np.testing.assert_array_equal(trace.history_count,[0,0])
    np.testing.assert_array_equal(trace.weight_user,[0,0])


def test_guard_is_frozen_and_small_roles_back_off():
    frame=pd.DataFrame(dict(user=['u']*200+['v']*200+['w']*5,role=['low']*200+['high']*200+['small']*5,label=0))
    scores=np.r_[np.linspace(0,1,200),np.linspace(10,11,200),np.repeat(100.,5)]
    shifts=fit_guard(frame,scores)
    assert shifts['small']==0
    assert all(v>=0 for v in shifts.values())
    mapped=apply_guard(pd.DataFrame({'role':['new','high']}),[1000.,1000.],shifts)
    assert mapped[0]==1000
    assert mapped[1]==1000-shifts['high']


def test_fixed_threshold_respects_validation_role_cap():
    frame=pd.DataFrame(dict(user=np.repeat(np.arange(100),10),role=np.tile(['r','s'],500),label=0))
    scores=np.arange(1000,dtype=float)
    threshold,diagnostics=select_threshold(frame,scores,replicates=99)
    assert np.isfinite(threshold)
    assert (scores>threshold).mean()<=.01
    assert diagnostics
