"""Freeze, select thresholds on validation, then process a chronologically later test."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..calibration.base import Profiles
from ..development.contextual import geometric_score
from ..development.tails import ExtrapolatedProfiles,causal_mean,combine
from ..metrics import evaluate,stable_ties
from ..replication.run import frozen_models,lock_methods
from .model import AdaptiveContext
from .threshold import apply_guard,fit_guard,select_threshold

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/m5'
SOURCES={'r4.2':ROOT/'results/primary','r5.2':ROOT/'results/r5_2/primary','r6.2':ROOT/'results/m5_r6_detector'}
BASELINES=['M0','M1','M2','M3','M4','EU','TU','TM4','T30','AB']
ABLATIONS=['A_no_stability','A_no_role','A_no_temporal','A_no_spike','A_frozen_history','A_no_role_guard']
METHODS=BASELINES+['M5']+ABLATIONS
GUARDED=['M5']+ABLATIONS[:-1]


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'docs/protocol_m5.md',ROOT/'docs/m5_r6_label_audit_plan.md']
    files += [ROOT/'ueba/m5'/name for name in ['model.py','threshold.py','run.py','data.py']]
    files += list((ROOT/'ueba/calibration').glob('*.py'))
    files += [ROOT/p for p in ['ueba/development/tails.py','ueba/development/contextual.py','ueba/metrics.py','ueba/splits.py','ueba/experiment.py','ueba/replication/prepare.py','ueba/replication/run.py']]
    plan=dict(files={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        seeds=list(range(42,47)),methods=METHODS,baselines=BASELINES,ablations=ABLATIONS,
        fractions=[.5,.2,.15,.15],threshold_replicates=999,bootstrap_replicates=9999,
        original_methods=lock_methods(),r6_test_seen_at_freeze=False,development_selection='One fixed M5, no test-dependent variant search')
    path=OUT/'frozen_plan.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Frozen scoring/protocol implementation changed')
    else:path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return plan


def classic(calibration,targets):
    c=Profiles().fit(calibration).components(targets)
    scores={name:model.transform(targets.score,c) for name,model in frozen_models().items()}
    e=ExtrapolatedProfiles().fit(calibration).components(targets)
    scores['EU']=-e['log_user']
    scores['TU']=causal_mean(targets,scores['EU'])
    scores['TM4']=causal_mean(targets,scores['M4'])
    scores['T30']=causal_mean(targets,combine(e,30))
    return scores


def metadata(frame,release):
    columns=['user','date','role','label','scenario']
    result=frame[columns].copy()
    for sid in range(1,6):
        key=f'scenario_{sid}'
        result[key]=frame[key].astype(int) if key in frame else frame.scenario.eq(sid).astype(int)
    if 'incident_ids' in frame:
        result['incident_ids']=frame.incident_ids
    else:
        events=pd.read_parquet(ROOT/'data/processed/answer_events.parquet')
        mapping=events.groupby(['user','date'])['key'].agg(lambda v:';'.join(sorted({Path(s).name for s in v}))).rename('incident_ids')
        result=result.merge(mapping,on=['user','date'],how='left',validate='one_to_one')
        result['incident_ids']=result.incident_ids.fillna('')
    if result.loc[result.label.eq(1),'incident_ids'].eq('').any():
        raise ValueError('Positive day lacks an official incident reference')
    return result


def incident_events(release):
    path=ROOT/('data/processed/answer_events.parquet' if release=='r4.2' else f'data/processed/{release}/answer_events.parquet')
    events=pd.read_parquet(path)
    events['incident_id']=events['key'].map(lambda s:Path(s).name) if release=='r4.2' else events.incident_id
    return events


def delay_rows(frame,alert,events,release,seed,method):
    rows=[];start=frame.date.min();end=frame.date.max()
    for incident,ground in events.groupby('incident_id',sort=False):
        flag=frame.incident_ids.map(lambda s:incident in s.split(';')).to_numpy()&frame.label.eq(1).to_numpy()
        if not flag.any():continue
        onset=ground.date.min();hits=frame.loc[flag&alert,'date']
        detected=bool(len(hits));delay=float((hits.min()-onset).days) if detected else np.nan
        left=onset<start;followup=int((end-onset).days)
        rows.append(dict(release=release,seed=seed,method=method,incident_id=incident,
            onset=str(onset.date()),left_censored=left,observed_positive_days=int(flag.sum()),
            detected=detected,delay_days=delay,
            delay_from_test_entry=float((hits.min()-max(onset,start)).days) if detected else np.nan,
            eligible_7day=bool(not left and followup>=7),
            restricted_delay_7day=min(delay,7.) if not left and followup>=7 and detected else
                7. if not left and followup>=7 else np.nan))
    return rows


def validation(release):
    folder=OUT/release;folder.mkdir(parents=True,exist_ok=True)
    val_metrics=[];threshold_rows=[];candidate_rows=[];partition=[]
    for seed in range(42,47):
        raw=pd.read_parquet(SOURCES[release]/f'seed_{seed}/raw_scores.parquet',filters=[('block','in',['calibration','validation'])])
        if not set(raw.block)<={'calibration','validation'}:raise ValueError('Unexpected test access during selection')
        cal=raw.loc[raw.block.eq('calibration')].reset_index(drop=True)
        val=raw.loc[raw.block.eq('validation')].reset_index(drop=True)
        midpoint=val.date.min()+pd.Timedelta(days=((val.date.max()+pd.Timedelta(days=1)-val.date.min()).days//2))
        guard_mask=val.date.lt(midpoint).to_numpy();th_mask=~guard_mask
        guard=val.loc[guard_mask].reset_index(drop=True);threshold_frame=val.loc[th_mask].reset_index(drop=True)
        scores=classic(cal,val)
        model=AdaptiveContext().fit(cal);adaptive,trace=model.transform(val);scores.update(adaptive)
        frozen=AdaptiveContext(update=False).fit(cal);frozen_values,_=frozen.transform(val)
        scores['A_frozen_history']=frozen_values['M5']
        scores['A_no_role_guard']=scores['M5'].copy()
        shifts={}
        for name in GUARDED:
            shifts[name]=fit_guard(guard,scores[name][guard_mask])
            scores[name]=apply_guard(val,scores[name],shifts[name])
        thresholds={};output=metadata(val,release)
        for name in METHODS:
            th,diagnostics=select_threshold(threshold_frame,scores[name][th_mask])
            thresholds[name]=th
            result,_=evaluate(threshold_frame,scores[name][th_mask],th,stable_ties(threshold_frame))
            val_metrics.append(dict(result,release=release,seed=seed,method=name))
            threshold_rows.append(dict(release=release,seed=seed,method=name,threshold=th,
                validation_fpr=result['fpr'],validation_recall=result['recall'],validation_max_role_fpr_n100=result['max_role_fpr_n100']))
            candidate_rows.extend(dict(d,release=release,seed=seed,method=name) for d in diagnostics)
            output[f'score_{name}']=scores[name];output[f'alert_{name}']=scores[name]>th
        output=pd.concat([output,trace.add_prefix('m5_')],axis=1)
        seed_folder=folder/f'seed_{seed}';seed_folder.mkdir(exist_ok=True)
        output.to_parquet(seed_folder/'validation_predictions.parquet',index=False)
        joblib.dump(dict(model=model,frozen=frozen,shifts=shifts,thresholds=thresholds),seed_folder/'stream_state.joblib')
        if seed==42:
            for name,part in [('calibration',cal),('guard',guard),('threshold',threshold_frame)]:
                partition.append(dict(release=release,block=name,start=str(part.date.min().date()),end=str(part.date.max().date()),
                    days=len(part),positives=int(part.label.sum()),positive_accounts=int(part.loc[part.label.eq(1),'user'].nunique())))
        print(f'M5 validation {release}/{seed} complete',flush=True)
    metrics=pd.DataFrame(val_metrics)
    metrics.to_csv(folder/'validation_metrics.csv',index=False)
    pd.DataFrame(threshold_rows).to_csv(folder/'thresholds.csv',index=False)
    pd.DataFrame(candidate_rows).to_csv(folder/'threshold_candidates.csv',index=False)
    pd.DataFrame(partition).to_csv(folder/'partitions.csv',index=False)
    means=metrics.loc[metrics.method.isin(BASELINES)].groupby('method')[['recall','fpr']].mean()
    chosen=sorted(BASELINES,key=lambda name:(-float(means.loc[name,'recall']) if np.isfinite(means.loc[name,'recall']) else np.inf,
        float(means.loc[name,'fpr']),BASELINES.index(name)))[0]
    selection=dict(release=release,comparator=chosen,selected_on='Threshold-validation only, before M5 test',
        validation_mean_recall=float(means.loc[chosen,'recall']),validation_mean_fpr=float(means.loc[chosen,'fpr']))
    locked=folder/'comparator.json'
    if locked.exists() and json.loads(locked.read_text())!=selection:raise ValueError('Comparator changed')
    locked.write_text(json.dumps(selection,indent=2),encoding='utf-8')
    print(json.dumps(selection),flush=True)


def test(release):
    folder=OUT/release
    if not (folder/'comparator.json').exists():raise ValueError('Lock validation comparator first')
    rows=[];roles=[];scenarios=[];histories=[];delays=[]
    events=incident_events(release)
    for seed in range(42,47):
        raw=pd.read_parquet(SOURCES[release]/f'seed_{seed}/raw_scores.parquet')
        cal=raw.loc[raw.block.eq('calibration')].reset_index(drop=True)
        val=raw.loc[raw.block.eq('validation')].reset_index(drop=True)
        frame=raw.loc[raw.block.eq('test')].reset_index(drop=True)
        if frame.date.min()<=val.date.max():raise ValueError('Overlapping validation/test')
        all_targets=pd.concat([val,frame],ignore_index=True)
        scores={name:values[len(val):] for name,values in classic(cal,all_targets).items()}
        state=joblib.load(folder/f'seed_{seed}/stream_state.joblib')
        adaptive,trace=state['model'].transform(frame);scores.update(adaptive)
        frozen_values,_=state['frozen'].transform(frame)
        scores['A_frozen_history']=frozen_values['M5'];scores['A_no_role_guard']=scores['M5'].copy()
        for name in GUARDED:scores[name]=apply_guard(frame,scores[name],state['shifts'][name])
        output=metadata(frame,release);ties=stable_ties(frame)
        bins=pd.cut(trace.n_eff,bins=[-np.inf,10,30,60,np.inf],labels=['0-9','10-29','30-59','60+'],right=False)
        for name in METHODS:
            values=scores[name];threshold=state['thresholds'][name]
            result,role=evaluate(frame,values,threshold,ties)
            rows.append(dict(result,release=release,seed=seed,method=name,threshold=threshold,
                fpr_exceeds_1pct=bool(result['fpr']>.01),role_fpr_n100_exceeds_5pct=bool(result['max_role_fpr_n100']>.05)))
            roles.extend(role.assign(release=release,seed=seed,method=name).to_dict('records'))
            alert=values>threshold
            output[f'score_{name}']=values;output[f'alert_{name}']=alert
            order=np.lexsort((ties,-values));top=order[:int(np.ceil(.01*len(frame)))]
            for sid in range(1,6):
                flag=output[f'scenario_{sid}'].to_numpy(bool);support=int(flag.sum())
                scenarios.append(dict(release=release,seed=seed,method=name,scenario=sid,positive_days=support,
                    positive_accounts=int(frame.loc[flag,'user'].nunique()),recall=float((alert&flag).sum()/support) if support else np.nan,
                    recall_at_1pct=float(flag[top].sum()/support) if support else np.nan))
            for history in bins.cat.categories:
                mask=bins.eq(history).to_numpy();y=frame.label.to_numpy(bool)
                histories.append(dict(release=release,seed=seed,method=name,history_bin=str(history),days=int(mask.sum()),
                    positives=int((mask&y).sum()),negative=int((mask&~y).sum()),
                    recall=float((mask&y&alert).sum()/(mask&y).sum()) if (mask&y).any() else np.nan,
                    fpr=float((mask&~y&alert).sum()/(mask&~y).sum()) if (mask&~y).any() else np.nan))
            delays.extend(delay_rows(output,alert,events,release,seed,name))
        output=pd.concat([output,trace.add_prefix('m5_')],axis=1)
        output.to_parquet(folder/f'seed_{seed}/test_predictions.parquet',index=False)
        print(f'M5 fixed-threshold test {release}/{seed} complete',flush=True)
    for filename,records in [('metrics.csv',rows),('role_metrics.csv',roles),('scenario_metrics.csv',scenarios),
        ('history_metrics.csv',histories),('attack_delays.csv',delays)]:
        pd.DataFrame(records).to_csv(folder/filename,index=False)


def r6_detector():
    from ..experiment import run_protocol
    frame=pd.read_parquet(ROOT/'data/processed/r6.2/user_days.parquet')
    audit=json.loads((ROOT/'data/processed/r6.2/data_audit.json').read_text())
    run_protocol(frame,audit['features'],'m5_r6_detector',[.5,.2,.15,.15],model_map=frozen_models(),history_caps=[],scenario_ids=range(1,6))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',choices=list(SOURCES),required=True)
    parser.add_argument('--stage',choices=['validation','test','all','detector'],default='all')
    args=parser.parse_args();freeze()
    with threadpool_limits(limits=1):
        if args.stage=='detector':
            if args.release!='r6.2':raise ValueError('Reuse existing detectors for r4.2/r5.2')
            r6_detector()
        if args.stage in ['validation','all']:validation(args.release)
        if args.stage in ['test','all']:test(args.release)


if __name__=='__main__':main()
