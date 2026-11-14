"""Bounded second development round, with a persisted selection before audit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..calibration.base import Profiles
from ..metrics import ranking_order,stable_ties,threshold_at_rate
from ..replication.run import frozen_models
from .contextual import ContinuousContext,geometric_score
from .run import BASELINES,OUT,RELEASES,metrics
from .tails import ExtrapolatedProfiles,causal_mean,combine

CANDIDATES=['E30','E5','EU','T30','T5','TU','TG30']


def main():
    protocol=Path(__file__).resolve().parents[2]/'docs/protocol_m4_v2_round2.md'
    plan=dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),candidates=CANDIDATES,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        seeds=list(range(42,47)),accessed_blocks=['calibration','validation'])
    plan_path=OUT/'development_plan_round2.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Round-two implementation changed; no silent reruns')
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    rows=[];cache={}
    for release,directory in RELEASES.items():
        for seed in plan['seeds']:
            raw=pd.read_parquet(directory/f'seed_{seed}/raw_scores.parquet',filters=[('block','in',['calibration','validation'])])
            if not set(raw.block)<={'calibration','validation'}:
                raise ValueError('Test block accessed')
            cal=raw.loc[raw.block.eq('calibration')].reset_index(drop=True)
            val=raw.loc[raw.block.eq('validation')].reset_index(drop=True)
            midpoint=val.date.min()+pd.Timedelta(days=((val.date.max()+pd.Timedelta(days=1)-val.date.min()).days//2))
            selection=val.date.lt(midpoint).to_numpy()
            c=Profiles().fit(cal).components(val)
            scores={name:model.transform(val.score,c) for name,model in frozen_models().items()}
            saved=pd.read_parquet(directory/f'seed_{seed}/validation_predictions.parquet')
            for name in BASELINES:
                np.testing.assert_allclose(scores[name],saved[f'score_{name}'])
            e=ExtrapolatedProfiles().fit(cal).components(val)
            scores.update(E30=combine(e,30),E5=combine(e,5),EU=-e['log_user'])
            for target,source in [('T30','E30'),('T5','E5'),('TU','EU')]:
                scores[target]=causal_mean(val,scores[source])
            scores['TG30']=causal_mean(val,geometric_score(c))
            scores['EB30']=ContinuousContext(mode='eb').fit(cal).transform(val)
            scores['UZ']=ContinuousContext(mode='user').fit(cal).transform(val)
            thresholds={name:threshold_at_rate(s[selection]) for name,s in scores.items()}
            for name,s in scores.items():
                rows.append(dict(metrics(val.loc[selection].reset_index(drop=True),s[selection],thresholds[name]),release=release,seed=seed,method=name))
            cache[release,seed]=(val,~selection,scores,thresholds)
            print(f'Round 2 selection prepared {release}/{seed}',flush=True)
    selection_metrics=pd.DataFrame(rows)
    selection_metrics.to_csv(OUT/'selection_metrics_round2.csv',index=False)
    means=selection_metrics.groupby(['release','method'])[['recall_at_1pct','max_role_fpr']].mean()
    criteria=[]
    for order,name in enumerate(CANDIDATES):
        details={};ok=True
        for release in RELEASES:
            m=means.loc[release];candidate=m.loc[name];best=m.loc[BASELINES].recall_at_1pct.max()
            good=bool(candidate.recall_at_1pct>=best-1e-12 and candidate.max_role_fpr<=1.1*m.loc['M3','max_role_fpr']+1e-12)
            ok &= good
            details[release]=dict(recall_at_1pct=float(candidate.recall_at_1pct),best_baseline_recall=float(best),
                max_role_fpr=float(candidate.max_role_fpr),qualified=good)
        criteria.append(dict(candidate=name,qualified=ok,order=order,per_release=details,
            mean_recall=float(means.xs(name,level='method').recall_at_1pct.mean()),
            mean_max_role_fpr=float(means.xs(name,level='method').max_role_fpr.mean())))
    qualified=[c for c in criteria if c['qualified']]
    chosen=sorted(qualified,key=lambda c:(-c['mean_recall'],c['mean_max_role_fpr'],c['order']))[0]['candidate'] if qualified else 'E30'
    choice=dict(selected=chosen,qualified=bool(qualified),candidates=criteria,exploratory=True)
    locked=OUT/'selection_round2.json'
    if locked.exists() and json.loads(locked.read_text())!=choice:
        raise ValueError('Locked selection changed')
    locked.write_text(json.dumps(choice,indent=2),encoding='utf-8')
    print(f'Locked round-two selection: {chosen}, qualified={bool(qualified)}',flush=True)
    rows=[];scenarios=[]
    compared=list(dict.fromkeys(BASELINES+['EB30','UZ',chosen,'EU','TU']))
    for (release,seed),(val,audit,scores,thresholds) in cache.items():
        part=val.loc[audit].reset_index(drop=True)
        output=part[['user','date','role','label','scenario']].copy()
        for i in range(1,6):
            key=f'scenario_{i}'
            output[key]=part[key].astype(int) if key in part else part.scenario.eq(i).astype(int)
        for name in compared:
            s=scores[name][audit]
            rows.append(dict(metrics(part,s,thresholds[name]),release=release,seed=seed,method=name,threshold=thresholds[name]))
            output[f'score_{name}']=s;output[f'alert_{name}']=s>thresholds[name]
            top=ranking_order(s,stable_ties(part))[:int(np.ceil(.01*len(part)))]
            for i in range(1,6):
                flags=output[f'scenario_{i}'].to_numpy(bool);n=int(flags.sum())
                scenarios.append(dict(release=release,seed=seed,method=name,scenario=i,positive_days=n,
                    positive_accounts=int(part.loc[flags,'user'].nunique()),
                    recall_at_1pct=float(flags[top].sum()/n) if n else np.nan))
        folder=OUT/release/f'seed_{seed}';folder.mkdir(exist_ok=True,parents=True)
        output.to_parquet(folder/'audit_predictions_round2.parquet',index=False)
    pd.DataFrame(rows).to_csv(OUT/'audit_metrics_round2.csv',index=False)
    pd.DataFrame(scenarios).to_csv(OUT/'audit_scenario_metrics_round2.csv',index=False)


if __name__=='__main__':
    main()
