import numpy as np
import pandas as pd

from ueba.replication.prepare import selected_events
from ueba.replication.run import frozen_models
from ueba.replication.statistics import top_hits,bootstrap_period,correct_families
from ueba.experiment import run_protocol
from ueba.metrics import stable_ties,ranking_order


def test_comparable_events_exclude_receipts_and_local_file_reads():
    file = pd.DataFrame(dict(activity=["File Open","File Open","File Write","File Copy"],
        to_removable_media=["False","False","True","True"],from_removable_media=["True","False","False","False"]))
    assert selected_events(file,"file").tolist()==[False,False,True,True]
    email = pd.DataFrame(dict(activity=["Send","Receive"]))
    assert selected_events(email,"email").tolist()==[True,False]


def test_frozen_m4_parameters():
    model = frozen_models()["M4"]
    assert model.k_user==30 and model.k_role==200
    assert list(frozen_models())==["M0","M1","M2","M3","M4"]


def test_global_budget_scenario_recall_matches_explicit_repeated_clusters():
    rng = np.random.default_rng(132)
    uid = rng.integers(0,20,size=1000)
    labels = rng.random(1000)<.2
    scenario = rng.integers(0,5,size=1000)
    flags = np.column_stack([labels&(scenario==i) for i in range(5)]).astype(float)
    for _ in range(10):
        multiplicities = rng.multinomial(20,np.full(20,.05))
        repeated = np.repeat(np.arange(len(uid)),multiplicities[uid])
        for k in [1,int(np.ceil(.01*len(repeated))),int(.8*len(repeated))]:
            actual,by_scenario = top_hits(uid,labels,flags,multiplicities,k)
            assert actual==labels[repeated[:k]].sum()
            np.testing.assert_array_equal(by_scenario,flags[repeated[:k]].sum(axis=0))


def test_global_budget_prefix_expands_when_leading_users_have_zero_weight():
    uid = np.r_[np.zeros(256,dtype=int),np.ones(20,dtype=int)]
    labels = np.ones(len(uid),bool)
    flags = np.column_stack([labels,*[np.zeros(len(uid),bool) for _ in range(4)]]).astype(float)
    hits,scenario_hits = top_hits(uid,labels,flags,np.array([0,2]),3)
    assert hits==3 and scenario_hits[0]==3


def test_r5_bootstrap_matches_saved_point_metrics(tmp_path,monkeypatch):
    from ueba import experiment
    rows=[]
    rng=np.random.default_rng(71)
    for date in pd.date_range("2020-01-01",periods=100):
        for user in range(8):
            label=int(date.dayofyear%7==0 and user%2==0)
            scenario=4 if label else 0
            rows.append(dict(user=f"u{user}",date=date,role=f"r{user%2}",label=label,scenario=scenario,
                activity=int(rng.poisson(6)),incident_ids="fixture" if label else "",
                **{f"scenario_{i}":int(scenario==i) for i in range(1,6)},
                logon_events=2,device_events=1,file_events=0,email_events=3))
    frame=pd.DataFrame(rows)
    monkeypatch.setattr(experiment,"ROOT",tmp_path)
    run_protocol(frame,["activity"],"primary",[.5,.2,.15,.15],model_map=frozen_models(),history_caps=[],scenario_ids=range(1,6))
    directory=tmp_path/"results/primary"
    pairs=bootstrap_period(directory,replicates=20)
    corrected=correct_families([pairs],tmp_path/"results")
    assert corrected.p_holm_primary.notna().sum()==2
    ci=pd.read_csv(directory/"confidence_intervals.csv")
    metrics=pd.read_csv(directory/"metrics.csv")
    means=metrics.query("operating_point=='prospective'").groupby("method").mean(numeric_only=True)
    for method in frozen_models():
        for metric in ["recall_at_1pct","recall","max_role_fpr","fpr","precision_at_1pct"]:
            point=ci.loc[ci.method.eq(method)&ci.metric.eq(metric),"estimate"].iloc[0]
            np.testing.assert_allclose(point,means.loc[method,metric],atol=1e-12)
        per_seed=[]
        for path in sorted(directory.glob("seed_*/test_predictions.parquet")):
            test=pd.read_parquet(path)
            order=ranking_order(test[f"score_{method}"],stable_ties(test))
            top=order[:int(np.ceil(.01*len(test)))]; support=test.scenario_4.sum()
            per_seed.append(test.scenario_4.iloc[top].sum()/support)
        point=ci.loc[ci.method.eq(method)&ci.metric.eq("scenario_4_recall_at_1pct"),"estimate"].iloc[0]
        np.testing.assert_allclose(point,np.mean(per_seed),atol=1e-12)
    assert ci.loc[ci.metric.eq("scenario_5_recall_at_1pct"),"estimate"].isna().all()
