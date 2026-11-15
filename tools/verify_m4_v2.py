"""Check primary prototype metrics independently from saved row-level predictions."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ueba.replication.run import lock_methods


def main():
    out=ROOT/'results/m4_v2_development'
    choice=json.loads((out/'selection_round2.json').read_text())
    selected=choice['selected']
    metrics=pd.read_csv(out/'audit_metrics_round2.csv')
    controls=pd.read_csv(out/'temporal_control_metrics.csv')
    intervals=pd.read_csv(out/'confidence_intervals.csv')
    locks=lock_methods()
    assert locks['matched_original_r4_2_bundle']
    checked=0;baseline_arrays=0
    for release,original in [('r4.2','results/primary'),('r5.2','results/r5_2/primary')]:
        for seed in range(42,47):
            frozen=pd.read_parquet(ROOT/original/f'seed_{seed}/validation_predictions.parquet')
            midpoint=frozen.date.min()+pd.Timedelta(days=((frozen.date.max()+pd.Timedelta(days=1)-frozen.date.min()).days//2))
            expected=frozen.loc[frozen.date.ge(midpoint)].reset_index(drop=True)
            for filename,summary in [('audit_predictions_round2.parquet',metrics),('audit_predictions_temporal_controls.parquet',controls)]:
                frame=pd.read_parquet(out/release/f'seed_{seed}'/filename)
                assert frame[['user','date','role','label']].equals(expected[['user','date','role','label']])
                assert not frame[['user','date']].duplicated().any()
                keys=[hashlib.sha256(f'{u}|{pd.Timestamp(d).date()}'.encode()).hexdigest() for u,d in zip(frame.user,frame.date)]
                y=frame.label.to_numpy(bool)
                for method in summary.method.unique():
                    row=summary.loc[summary.release.eq(release)&summary.seed.eq(seed)&summary.method.eq(method)].iloc[0]
                    score=frame[f'score_{method}'].to_numpy(float)
                    alert=frame[f'alert_{method}'].to_numpy(bool)
                    assert np.isfinite(score).all()
                    np.testing.assert_array_equal(alert,score>row.threshold)
                    ranking=pd.DataFrame({'score':score,'tie':keys,'label':y}).sort_values(['score','tie'],ascending=[False,True])
                    k=int(np.ceil(.01*len(frame)))
                    np.testing.assert_allclose(row.recall_at_1pct,ranking.label.iloc[:k].sum()/y.sum())
                    np.testing.assert_allclose(row.alert_rate,alert.mean())
                    np.testing.assert_allclose(row.recall,(alert&y).sum()/y.sum())
                    np.testing.assert_allclose(row.fpr,(alert&~y).sum()/(~y).sum())
                    role=frame[['role']].copy();role['neg']=~y;role['fp']=alert&~y
                    groups=role.groupby('role')[['neg','fp']].sum()
                    np.testing.assert_allclose(row.max_role_fpr,(groups.fp/groups.neg.replace(0,np.nan)).max())
                    if method in ['M0','M1','M2','M3','M4']:
                        np.testing.assert_array_equal(score,expected[f'score_{method}'])
                        baseline_arrays+=1
                    checked+=1
    for row in intervals.itertuples():
        expected=metrics.loc[metrics.release.eq(row.release)&metrics.method.eq(row.method),row.metric].mean()
        np.testing.assert_allclose(row.estimate,expected,atol=1e-12)
    plans=[]
    for filename,protocol in [('development_plan.json','protocol_m4_v2.md'),('development_plan_round2.json','protocol_m4_v2_round2.md')]:
        plan=json.loads((out/filename).read_text())
        assert plan['accessed_blocks']==['calibration','validation']
        assert plan['protocol_sha256']==hashlib.sha256((ROOT/'docs'/protocol).read_bytes()).hexdigest()
        # These are development implementations; later report/statistics modules are separate.
        algorithm_files=['contextual.py','run.py'] if filename=='development_plan.json' else ['contextual.py','run.py','tails.py','round2.py']
        for name in algorithm_files:
            assert plan['source_sha256'][name]==hashlib.sha256((ROOT/'ueba/development'/name).read_bytes()).hexdigest(),name
        plans.append(filename)
    family=json.loads((out/'multiplicity_family.json').read_text())
    assert family['tests']==28
    assert len(pd.read_csv(out/'temporal_control_comparisons.csv'))==12
    result=dict(passed=True,prototype='M4v2-'+selected,seed_method_metric_rows=checked,
        original_baseline_arrays_identical=baseline_arrays,original_calibration=locks,
        fixed_plans=plans,independent_confirmation=False,bootstrap_replicates=9999)
    (ROOT/'reports/artifact_validation_m4_v2.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
