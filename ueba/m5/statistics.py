"""Paired user-cluster inference conditional on fixed validation choices."""
import argparse
import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from ..metrics import ranking_order,stable_ties
from ..replication.statistics import interval,mean_without_warnings,top_hits
from ..statistics import cluster_role_index,safe_ratio
from .run import METHODS,OUT

REPLICATES=9999


def bootstrap(release):
    folder=OUT/release
    comparator=json.loads((folder/'comparator.json').read_text())['comparator']
    paths=sorted(folder.glob('seed_*/test_predictions.parquet'))
    if len(paths)!=5:raise ValueError('All five seeds required')
    first=pd.read_parquet(paths[0]);users=sorted(first.user.unique());nu=len(users)
    uid=pd.Categorical(first.user,categories=users).codes.astype(np.int64)
    rid,roles=pd.factorize(first.role,sort=True);nr=len(roles)
    y=first.label.to_numpy(bool)
    weights=np.random.default_rng(20261102).multinomial(nu,np.full(nu,1/nu),size=REPLICATES).astype(float)
    weights=np.vstack([weights,np.ones(nu)])
    user_positive=np.bincount(uid,weights=y,minlength=nu)
    pos=weights@user_positive;neg=weights@np.bincount(uid,weights=~y,minlength=nu);total=pos+neg
    index=cluster_role_index(uid,rid,nr)
    role_neg=np.bincount(index,weights=~y,minlength=nu*nr).reshape(nu,nr)
    denominator=weights@role_neg;eligible=role_neg.sum(axis=0)>=100
    ties=stable_ties(first);budget=np.maximum(1,np.ceil(.01*total).astype(int))
    empty=np.empty((len(first),0))
    names=['recall','fpr','precision','recall_at_1pct','max_role_fpr','max_role_fpr_n100','alert_rate']
    draws={m:{n:[] for n in names} for m in METHODS}
    for path in paths:
        frame=pd.read_parquet(path)
        if not frame[['user','date','role','label']].equals(first[['user','date','role','label']]):raise ValueError('Nonpaired seed rows')
        for method in METHODS:
            alert=frame[f'alert_{method}'].to_numpy(bool)
            tp=weights@np.bincount(uid,weights=alert&y,minlength=nu)
            fp=weights@np.bincount(uid,weights=alert&~y,minlength=nu)
            counts=np.bincount(index,weights=alert&~y,minlength=nu*nr).reshape(nu,nr)
            rates=safe_ratio(weights@counts,denominator)
            order=ranking_order(frame[f'score_{method}'],ties)
            hit=np.zeros(len(weights));sorted_uid,sorted_y=uid[order],y[order]
            for b,w in enumerate(weights):hit[b],_=top_hits(sorted_uid,sorted_y,empty,w,budget[b])
            values=dict(recall=safe_ratio(tp,pos),fpr=safe_ratio(fp,neg),precision=safe_ratio(tp,tp+fp,zero=0.),
                recall_at_1pct=safe_ratio(hit,pos),max_role_fpr=np.nanmax(rates,axis=1),
                max_role_fpr_n100=np.nanmax(rates[:,eligible],axis=1),alert_rate=(tp+fp)/total)
            for name in names:draws[method][name].append(values[name])
        print(f'M5 bootstrap {release}/{path.parent.name}',flush=True)
    means={m:{name:mean_without_warnings(v) for name,v in metrics.items()} for m,metrics in draws.items()}
    ci=[];pairs=[]
    for method,metrics in means.items():
        for name,values in metrics.items():
            finite,lo,hi=interval(values)
            ci.append(dict(release=release,method=method,metric=name,estimate=values[-1],lower=lo,upper=hi,
                finite_replicates=len(finite),replicates=REPLICATES))
    for other in METHODS:
        if other=='M5':continue
        for name in ['recall','fpr','max_role_fpr']:
            values=means['M5'][name]-means[other][name]
            finite,lo,hi=interval(values)
            p=(1+np.sum(np.abs(finite-values[-1])>=abs(values[-1])))/(len(finite)+1) if len(finite) else np.nan
            pairs.append(dict(release=release,comparison='M5-'+other,comparator=other,metric=name,
                difference=values[-1],lower=lo,upper=hi,p_centered_bootstrap=p,
                finite_replicates=len(finite),primary=bool(other==comparator and name in ['recall','fpr'])))
    pd.DataFrame(ci).to_csv(folder/'confidence_intervals.csv',index=False)
    pd.DataFrame(pairs).to_csv(folder/'paired_comparisons_uncorrected.csv',index=False)
    diagnostics=dict(release=release,users=nu,positive_accounts=int((user_positive>0).sum()),
        positive_days=int(y.sum()),replicates=REPLICATES,seed=20261102,
        replicates_without_positive=int((pos[:-1]==0).sum()),
        conditional_on='Models, frozen M5 code, validation-selected thresholds and comparator',
        warning='Undefined recall draws omitted from recall CI; few positive accounts imply weak population inference')
    (folder/'bootstrap_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')


def correct_all():
    paths=sorted(OUT.glob('r*/paired_comparisons_uncorrected.csv'))
    if not paths:raise ValueError('No completed bootstrap results')
    frame=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    valid=frame.p_centered_bootstrap.notna()
    for field,mask in [('p_holm_primary',frame.primary.astype(bool)&valid),('p_holm_exploratory',valid)]:
        frame.loc[mask,field]=multipletests(frame.loc[mask,'p_centered_bootstrap'],method='holm')[1]
    frame.to_csv(OUT/'paired_comparisons.csv',index=False)
    (OUT/'multiplicity_families.json').write_text(json.dumps(dict(primary_tests=int((frame.primary&valid).sum()),
        exploratory_tests=int(valid.sum()),releases=sorted(frame.release.unique()),replicates=REPLICATES),indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',choices=['r4.2','r5.2','r6.2'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.release:bootstrap(args.release)
        correct_all()


if __name__=='__main__':main()
