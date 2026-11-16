"""Focused paired cluster inference with one global alert budget per scenario."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..metrics import ranking_order, stable_ties
from ..statistics import cluster_role_index, safe_ratio

METHODS = ["M0", "M1", "M2", "M3", "M4"]
PAIRS = [(m, "M0") for m in METHODS[1:]]+[("M4", "M3")]
ENDPOINTS = ["recall_at_1pct", "max_role_fpr"]
ROOT = Path(__file__).resolve().parents[2]
REPLICATES = 9999


def top_hits(uid, labels, flags, multiplicities, k):
    """Exact weighted top-K; expand its sorted prefix if cluster weights require it."""
    n = len(labels)
    end = min(n, max(64, 4*k))
    while True:
        weights = multiplicities[uid[:end]]
        cumulative = np.cumsum(weights)
        if cumulative[-1] >= k or end == n:
            break
        end = min(n, end*2)
    idx = min(int(np.searchsorted(cumulative,k,side="left")),end-1)
    excess = cumulative[idx]-k
    chosen = weights[:idx+1]
    hits = float(chosen @ labels[:idx+1]-excess*labels[idx])
    scenario_hits = chosen @ flags[:idx+1]-excess*flags[idx]
    return hits,np.asarray(scenario_hits,float)


def mean_without_warnings(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        return np.nanmean(values,axis=0)


def interval(values):
    finite = values[:-1][np.isfinite(values[:-1])]
    lo,hi = np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
    return finite,lo,hi


def bootstrap_period(directory,replicates=REPLICATES):
    directory = Path(directory)
    paths = sorted(directory.glob("seed_*/test_predictions.parquet"))
    if len(paths)!=5:
        raise ValueError("Five paired seeds required")
    first = pd.read_parquet(paths[0])
    users = sorted(first.user.unique())
    uid = pd.Categorical(first.user,categories=users).codes.astype(np.int64)
    rid,roles = pd.factorize(first.role,sort=True)
    nu,nr = len(users),len(roles)
    flags = first[[f"scenario_{i}" for i in range(1,6)]].to_numpy(float)
    y = first.label.to_numpy(bool)
    rng = np.random.default_rng(20261102)
    weights = rng.multinomial(nu,np.full(nu,1/nu),size=replicates).astype(float)
    weights = np.vstack([weights,np.ones(nu)])
    positive_user = np.bincount(uid,weights=y,minlength=nu)
    negative_user = np.bincount(uid,weights=~y,minlength=nu)
    pos,neg = weights@positive_user,weights@negative_user
    totals = pos+neg
    scenario_user = np.column_stack([np.bincount(uid,weights=flags[:,i],minlength=nu) for i in range(5)])
    scenario_denominator = weights@scenario_user
    index = cluster_role_index(uid,rid,nr)
    role_negative = np.bincount(index,weights=~y,minlength=nu*nr).reshape(nu,nr)
    role_denominator = weights@role_negative
    eligible = role_negative.sum(axis=0)>=100
    ties = stable_ties(first)
    names = ["recall_at_1pct","max_role_fpr","max_role_fpr_n100","recall","fpr","precision_at_1pct"]
    names += [f"scenario_{i}_{kind}" for i in range(1,6) for kind in ["recall_at_1pct","recall"]]
    draws = {m:{n:[] for n in names} for m in METHODS}
    role_draws = {m:[] for m in METHODS}
    budgets = np.maximum(1,np.ceil(.01*totals).astype(int))
    metadata = ["user","date","role","label",*[f"scenario_{i}" for i in range(1,6)]]
    for path in paths:
        frame = pd.read_parquet(path)
        if not first[metadata].equals(frame[metadata]):
            raise ValueError("Nonpaired test rows across seeds")
        for method in METHODS:
            alert = frame[f"alert_{method}"].to_numpy(bool)
            tp = weights@np.bincount(uid,weights=alert&y,minlength=nu)
            fp = weights@np.bincount(uid,weights=alert&~y,minlength=nu)
            role_fp = np.bincount(index,weights=alert&~y,minlength=nu*nr).reshape(nu,nr)
            rates = safe_ratio(weights@role_fp,role_denominator)
            role_draws[method].append(rates)
            values = dict(recall=safe_ratio(tp,pos),fpr=safe_ratio(fp,neg),max_role_fpr=np.nanmax(rates,axis=1),
                max_role_fpr_n100=np.nanmax(rates[:,eligible],axis=1) if eligible.any() else np.full(len(weights),np.nan))
            order = ranking_order(frame[f"score_{method}"],ties)
            sorted_uid,sorted_y,sorted_flags = uid[order],y[order],flags[order]
            hits,scenario_hits = np.zeros(len(weights)),np.zeros((len(weights),5))
            for b,multiplicities in enumerate(weights):
                hits[b],scenario_hits[b] = top_hits(sorted_uid,sorted_y,sorted_flags,multiplicities,budgets[b])
            values["recall_at_1pct"],values["precision_at_1pct"] = safe_ratio(hits,pos),hits/budgets
            for i in range(5):
                detected = np.bincount(uid,weights=alert*flags[:,i],minlength=nu)
                values[f"scenario_{i+1}_recall"] = safe_ratio(weights@detected,scenario_denominator[:,i])
                values[f"scenario_{i+1}_recall_at_1pct"] = safe_ratio(scenario_hits[:,i],scenario_denominator[:,i])
            for metric in names:
                draws[method][metric].append(values[metric])
        print(f"r5.2 bootstrap {directory.name}/{path.parent.name}: {replicates} replicates",flush=True)
    means = {m:{name:mean_without_warnings(values) for name,values in draws[m].items()} for m in METHODS}
    cis,pairs,role_cis = [],[],[]
    for method in METHODS:
        for metric,values in means[method].items():
            finite,lo,hi = interval(values)
            cis.append(dict(period=directory.name,method=method,metric=metric,estimate=values[-1],
                lower=lo,upper=hi,finite_replicates=len(finite),replicates=replicates))
        rates = mean_without_warnings(role_draws[method])
        for i,role in enumerate(roles):
            finite,lo,hi = interval(rates[:,i])
            role_cis.append(dict(period=directory.name,method=method,role=role,
                negative_days=int(role_negative[:,i].sum()),fpr=rates[-1,i],lower=lo,upper=hi,
                finite_replicates=len(finite)))
    for method,comparator in PAIRS:
        for metric in names:
            values = means[method][metric]-means[comparator][metric]
            finite,lo,hi = interval(values)
            p = ((1+np.sum(np.abs(finite-values[-1])>=abs(values[-1])))/(len(finite)+1)
                 if len(finite) and np.isfinite(values[-1]) else np.nan)
            pairs.append(dict(period=directory.name,comparison=f"{method}-{comparator}",metric=metric,
                difference=values[-1],lower=lo,upper=hi,p_centered_bootstrap=p,finite_replicates=len(finite)))
    pd.DataFrame(cis).to_csv(directory/"confidence_intervals.csv",index=False)
    pd.DataFrame(pairs).to_csv(directory/"paired_comparisons_uncorrected.csv",index=False)
    pd.DataFrame(role_cis).to_csv(directory/"role_confidence_intervals.csv",index=False)
    diagnostics = dict(replicates=replicates,bootstrap_seed=20261102,users=nu,
        positive_accounts=int((positive_user>0).sum()),replicates_without_positive=int((pos[:-1]==0).sum()),
        scenario_replicates_without_positive={str(i+1):int((scenario_denominator[:-1,i]==0).sum()) for i in range(5)},
        conditional_on="Trained models, frozen profiles, fixed validation thresholds",
        scenario_budget="Shared global top-1% list, without per-scenario re-ranking")
    (directory/"bootstrap_diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    return pd.DataFrame(pairs)


def correct_families(all_pairs,output):
    frame = pd.concat(all_pairs,ignore_index=True)
    valid = frame.p_centered_bootstrap.notna()
    primary = frame.period.eq("primary")&frame.comparison.eq("M4-M3")&frame.metric.isin(ENDPOINTS)&valid
    secondary = frame.metric.isin(ENDPOINTS)&valid
    scenario = frame.metric.str.match(r"scenario_\d+_recall(?:_at_1pct)?$")&valid
    masks = [("primary",primary),("secondary",secondary),("scenario",scenario)]
    for name,mask in masks:
        if mask.any():
            frame.loc[mask,f"p_holm_{name}"] = multipletests(frame.loc[mask,"p_centered_bootstrap"],method="holm")[1]
    frame.to_csv(Path(output)/"paired_comparisons.csv",index=False)
    families = {name:dict(tests=int(mask.sum()),minimum_attainable_raw_p=1/(REPLICATES+1)) for name,mask in masks}
    (Path(output)/"multiplicity_families.json").write_text(json.dumps(families,indent=2),encoding="utf-8")
    return frame
