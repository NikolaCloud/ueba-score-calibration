"""Independent recomputation of prospective metrics and fixed-plan integrity."""
import hashlib
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ueba.m5.run import BASELINES,METHODS,OUT,SOURCES,freeze


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default=str(OUT))
    parser.add_argument('--skip-prior-contextual-cache',action='store_true')
    args=parser.parse_args()
    OUT=Path(args.results).resolve()
    import ueba.m5.run as runner
    runner.OUT=OUT
    freeze()
    releases=[p.name for p in sorted(OUT.glob('r*')) if (p/'metrics.csv').exists()]
    checked=0;original_arrays=0;contextual_arrays=0;threshold_rows=0;failures=[]
    for release in releases:
        folder=OUT/release
        metrics=pd.read_csv(folder/'metrics.csv',float_precision='round_trip')
        thresholds=pd.read_csv(folder/'thresholds.csv',float_precision='round_trip')
        candidates=pd.read_csv(folder/'threshold_candidates.csv',float_precision='round_trip')
        val_metrics=pd.read_csv(folder/'validation_metrics.csv')
        means=val_metrics.loc[val_metrics.method.isin(BASELINES)].groupby('method')[['recall','fpr']].mean()
        chosen=sorted(BASELINES,key=lambda m:(-means.loc[m,'recall'] if np.isfinite(means.loc[m,'recall']) else np.inf,
            means.loc[m,'fpr'],BASELINES.index(m)))[0]
        assert chosen==json.loads((folder/'comparator.json').read_text())['comparator']
        for seed in range(42,47):
            pred=pd.read_parquet(folder/f'seed_{seed}/test_predictions.parquet')
            val=pd.read_parquet(folder/f'seed_{seed}/validation_predictions.parquet')
            if release in ['r4.2','r5.2'] and not args.skip_prior_contextual_cache:
                prior=ROOT/'results/m4_v2_development'/release/f'seed_{seed}'
                audit=pd.read_parquet(prior/'audit_predictions_round2.parquet')
                aligned=val.merge(audit[['user','date']],on=['user','date'],how='inner',validate='one_to_one')
                assert aligned[['user','date']].equals(audit[['user','date']])
                for name in ['EU','TU','T30']:
                    np.testing.assert_allclose(aligned[f'score_{name}'],audit[f'score_{name}'],rtol=0,atol=1e-12)
                    contextual_arrays+=1
                temporal=pd.read_parquet(prior/'audit_predictions_temporal_controls.parquet')
                np.testing.assert_allclose(aligned.score_TM4,temporal.score_TM4,rtol=0,atol=1e-12)
                contextual_arrays+=1
            state=joblib.load(folder/f'seed_{seed}/stream_state.joblib')
            assert state['model'].last_transform==val.date.max()
            assert pred.date.min()>val.date.max()
            assert not pred[['user','date']].duplicated().any()
            raw=pd.read_parquet(SOURCES[release]/f'seed_{seed}/raw_scores.parquet',filters=[('block','==','test')]).reset_index(drop=True)
            assert pred[['user','date','role','label']].equals(raw[['user','date','role','label']])
            original=pd.read_parquet(SOURCES[release]/f'seed_{seed}/test_predictions.parquet')
            keys=[hashlib.sha256(f'{u}|{pd.Timestamp(d).date()}'.encode()).hexdigest() for u,d in zip(pred.user,pred.date)]
            y=pred.label.to_numpy(bool)
            expected_weight=pred.m5_n_eff/(pred.m5_n_eff+30)/(1+pred.m5_instability**2)
            np.testing.assert_allclose(pred.m5_weight_user,expected_weight,rtol=1e-12)
            z=np.maximum(pred.m5_cusum,pred.m5_instant_evidence)
            expected_m5=z-pred.role.astype(str).map(state['shifts']['M5']).fillna(0).to_numpy()
            np.testing.assert_allclose(pred.score_M5,expected_m5,rtol=1e-12)
            np.testing.assert_allclose(pred.score_A_no_role_guard,z,rtol=1e-12)
            midpoint=val.date.min()+pd.Timedelta(days=((val.date.max()+pd.Timedelta(days=1)-val.date.min()).days//2))
            valid=val.loc[val.date.ge(midpoint)].reset_index(drop=True)
            for method in METHODS:
                row=metrics.loc[metrics.seed.eq(seed)&metrics.method.eq(method)].iloc[0]
                th=thresholds.loc[thresholds.seed.eq(seed)&thresholds.method.eq(method)].iloc[0]
                assert row.threshold==th.threshold==state['thresholds'][method]
                score=pred[f'score_{method}'].to_numpy(float)
                alert=pred[f'alert_{method}'].to_numpy(bool)
                assert np.isfinite(score).all()
                np.testing.assert_array_equal(alert,score>row.threshold)
                np.testing.assert_allclose(row.recall,(alert&y).sum()/y.sum() if y.sum() else np.nan)
                np.testing.assert_allclose(row.fpr,(alert&~y).sum()/(~y).sum())
                np.testing.assert_allclose(row.precision,(alert&y).sum()/alert.sum() if alert.sum() else 0.)
                ranking=pd.DataFrame({'score':score,'key':keys,'y':y}).sort_values(['score','key'],ascending=[False,True])
                k=int(np.ceil(.01*len(pred)))
                np.testing.assert_allclose(row.recall_at_1pct,ranking.y.iloc[:k].sum()/y.sum() if y.sum() else np.nan)
                r=pred[['role']].copy();r['negative']=~y;r['fp']=alert&~y
                groups=r.groupby('role')[['negative','fp']].sum()
                rates=groups.fp/groups.negative.replace(0,np.nan)
                np.testing.assert_allclose(row.max_role_fpr,rates.max())
                np.testing.assert_allclose(row.max_role_fpr_n100,rates.loc[groups.negative.ge(100)].max())
                neg=valid.label.eq(0).to_numpy();va=valid[f'score_{method}'].to_numpy()>row.threshold
                np.testing.assert_allclose(th.validation_fpr,(va&neg).sum()/neg.sum())
                if np.isfinite(row.threshold):
                    choices=candidates.loc[candidates.seed.eq(seed)&candidates.method.eq(method)&candidates.eligible]
                    np.testing.assert_allclose(row.threshold,choices.threshold.min())
                    assert choices.loc[np.isclose(choices.threshold,row.threshold),'fpr_upper_95'].max()<=.01+1e-12
                else:
                    assert not candidates.loc[candidates.seed.eq(seed)&candidates.method.eq(method),'eligible'].any()
                if method in ['M0','M1','M2','M3','M4']:
                    np.testing.assert_allclose(score,original[f'score_{method}'],rtol=0,atol=1e-14)
                    original_arrays+=1
                checked+=1;threshold_rows+=1
        ci=pd.read_csv(folder/'confidence_intervals.csv')
        for row in ci.itertuples():
            np.testing.assert_allclose(row.estimate,metrics.loc[metrics.method.eq(row.method),row.metric].mean(),atol=1e-12)
    r6_training_checked=False
    if 'r6.2' in releases:
        from ueba.splits import calendar_split
        data=pd.read_parquet(ROOT/'data/processed/r6.2/user_days.parquet')
        blocks,_=calendar_split(data,[.5,.2,.15,.15])
        features=json.loads((ROOT/'data/processed/r6.2/data_audit.json').read_text())['features']
        assert len(features)==17
        train=blocks['train'][features].to_numpy(float)
        medians=np.nanmedian(train,axis=0)
        imputed=np.where(np.isnan(train),medians[None,:],train)
        for seed in range(42,47):
            bundle=joblib.load(SOURCES['r6.2']/f'seed_{seed}/detector.joblib')
            preprocessing=bundle['preprocessing'];detector=bundle['detector']
            assert bundle['features']==features
            np.testing.assert_allclose(preprocessing[0].statistics_,medians)
            np.testing.assert_allclose(preprocessing[1].mean_,imputed.mean(axis=0))
            np.testing.assert_allclose(preprocessing[1].var_,imputed.var(axis=0),rtol=1e-10)
            assert detector.random_state==seed and detector.n_estimators==200 and detector.max_samples_==2048
        metadata=json.loads((ROOT/'data/metadata/r6.2/cert_metadata.json').read_text())
        official=next(x for x in metadata['files'] if x['name']=='r6.2.tar.bz2')
        proof=json.loads((ROOT/'data/raw/r6.2.tar.bz2.provenance.json').read_text())
        assert proof['md5']==official['computed_md5']
        r6_training_checked=True
    result=dict(passed=True,releases=releases,independently_recomputed_metric_rows=checked,
        frozen_original_baseline_arrays=original_arrays,prior_EU_TU_TM4_T30_arrays=contextual_arrays,
        validation_thresholds_checked=threshold_rows,
        comparator_selection_checked=True,causal_formula_from_traces_checked=True,all_frozen_source_hashes_match=True,
        r6_train_only_preprocessing_checked=r6_training_checked,
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','numba','llvmlite','threadpoolctl','pyarrow']},failures=failures)
    destination=ROOT/'reports/artifact_validation_m5.json' if OUT==ROOT/'results/m5' else OUT/'artifact_validation.json'
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
