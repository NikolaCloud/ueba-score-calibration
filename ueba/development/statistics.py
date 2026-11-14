"""Conditional, exploratory paired user-cluster inference for the frozen prototype."""
import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from ..metrics import ranking_order,stable_ties
from ..replication.statistics import interval,mean_without_warnings,top_hits
from ..statistics import cluster_role_index,safe_ratio
from .run import BASELINES,OUT,RELEASES

REPLICATES=9999
ENDPOINTS=['recall_at_1pct','max_role_fpr']


def bootstrap_audit(release,selected,prediction_file='audit_predictions_round2.parquet',comparators=None,diagnostic_suffix=''):
    paths=sorted((OUT/release).glob('seed_*/'+prediction_file))
    if len(paths)!=5:
        raise ValueError('Five paired seeds required')
    first=pd.read_parquet(paths[0])
    methods=[c.removeprefix('score_') for c in first if c.startswith('score_')]
    users=sorted(first.user.unique());nu=len(users)
    uid=pd.Categorical(first.user,categories=users).codes.astype(np.int64)
    rid,roles=pd.factorize(first.role,sort=True);nr=len(roles)
    y=first.label.to_numpy(bool)
    rng=np.random.default_rng(20261102)
    weights=rng.multinomial(nu,np.full(nu,1/nu),size=REPLICATES).astype(float)
    weights=np.vstack([weights,np.ones(nu)])
    pu=np.bincount(uid,weights=y,minlength=nu)
    negative_user=np.bincount(uid,weights=~y,minlength=nu)
    pos=weights@pu;neg=weights@negative_user;total=pos+neg
    budget=np.maximum(1,np.ceil(.01*total).astype(int))
    index=cluster_role_index(uid,rid,nr)
    role_neg=np.bincount(index,weights=~y,minlength=nu*nr).reshape(nu,nr)
    role_den=weights@role_neg
    eligible=role_neg.sum(axis=0)>=100
    ties=stable_ties(first)
    names=ENDPOINTS+['max_role_fpr_n100','alert_rate','recall','fpr']
    draws={m:{n:[] for n in names} for m in methods}
    empty_flags=np.empty((len(first),0))
    for path in paths:
        frame=pd.read_parquet(path)
        if not frame[['user','date','role','label']].equals(first[['user','date','role','label']]):
            raise ValueError('Unpaired rows')
        for method in methods:
            alert=frame[f'alert_{method}'].to_numpy(bool)
            tp=weights@np.bincount(uid,weights=alert&y,minlength=nu)
            fp=weights@np.bincount(uid,weights=alert&~y,minlength=nu)
            role_fp=np.bincount(index,weights=alert&~y,minlength=nu*nr).reshape(nu,nr)
            rates=safe_ratio(weights@role_fp,role_den)
            order=ranking_order(frame[f'score_{method}'],ties)
            sorted_uid,sorted_y=uid[order],y[order]
            hits=np.zeros(len(weights))
            for b,w in enumerate(weights):
                hits[b],_=top_hits(sorted_uid,sorted_y,empty_flags,w,budget[b])
            values=dict(recall_at_1pct=safe_ratio(hits,pos),max_role_fpr=np.nanmax(rates,axis=1),
                max_role_fpr_n100=np.nanmax(rates[:,eligible],axis=1),
                alert_rate=(tp+fp)/total,recall=safe_ratio(tp,pos),fpr=safe_ratio(fp,neg))
            for name in names:
                draws[method][name].append(values[name])
        print(f'Exploratory bootstrap {release}/{path.parent.name}',flush=True)
    means={m:{name:mean_without_warnings(v) for name,v in metrics.items()} for m,metrics in draws.items()}
    cis=[];pairs=[]
    for method,metrics in means.items():
        for name,values in metrics.items():
            finite,lo,hi=interval(values)
            cis.append(dict(release=release,method=method,metric=name,estimate=values[-1],lower=lo,upper=hi,
                finite_replicates=len(finite),replicates=REPLICATES))
    for baseline in dict.fromkeys(BASELINES+['EU','TU'] if comparators is None else comparators):
        if baseline==selected:
            continue
        for name in ENDPOINTS:
            values=means[selected][name]-means[baseline][name]
            finite,lo,hi=interval(values)
            p=(1+np.sum(np.abs(finite-values[-1])>=abs(values[-1])))/(len(finite)+1) if len(finite) else np.nan
            pairs.append(dict(release=release,comparison=f'{selected}-{baseline}',metric=name,
                difference=values[-1],lower=lo,upper=hi,p_centered_bootstrap=p,finite_replicates=len(finite)))
    diagnostics=dict(release=release,replicates=REPLICATES,seed=20261102,users=nu,positive_accounts=int((pu>0).sum()),
        replicates_without_positive=int((pos[:-1]==0).sum()),conditional_on='Frozen models, profiles, selected method and selection thresholds',
        independent_confirmation=False)
    (OUT/release/f'bootstrap_diagnostics{diagnostic_suffix}.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    return cis,pairs


def main():
    selected=json.loads((OUT/'selection_round2.json').read_text())['selected']
    cis=[];pairs=[]
    with threadpool_limits(limits=1):
        for release in RELEASES:
            c,p=bootstrap_audit(release,selected);cis.extend(c);pairs.extend(p)
    pd.DataFrame(cis).to_csv(OUT/'confidence_intervals.csv',index=False)
    frame=pd.DataFrame(pairs)
    valid=frame.p_centered_bootstrap.notna()
    frame.loc[valid,'p_holm']=multipletests(frame.loc[valid,'p_centered_bootstrap'],method='holm')[1]
    frame.to_csv(OUT/'paired_comparisons.csv',index=False)
    (OUT/'multiplicity_family.json').write_text(json.dumps(dict(tests=int(valid.sum()),
        methods_compared_to=BASELINES+['EU','TU'],endpoints=ENDPOINTS,releases=list(RELEASES),
        replicates=REPLICATES,exploratory=True),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
