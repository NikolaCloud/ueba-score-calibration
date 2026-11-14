"""Validation-only group guards and cluster-aware fixed threshold selection."""
import numpy as np
import pandas as pd


def fit_guard(frame,scores):
    scores=np.asarray(scores,float)
    negative=frame.label.eq(0).to_numpy()
    global_q=float(np.quantile(scores[negative],.95,method='higher'))
    shifts={}
    for role,idx in frame.groupby('role',sort=False).indices.items():
        idx=idx[negative[idx]]
        shifts[str(role)]=(max(0.,float(np.quantile(scores[idx],.95,method='higher'))-global_q) if len(idx)>=100 else 0.)
    return shifts


def apply_guard(frame,scores,shifts):
    return np.asarray(scores,float)-frame.role.astype(str).map(shifts).fillna(0).to_numpy(float)


def select_threshold(frame,scores,replicates=999):
    scores=np.asarray(scores,float)
    if len(scores)!=len(frame) or not np.isfinite(scores).all():
        raise ValueError('Finite paired validation scores required')
    y=frame.label.to_numpy(bool);negative=~y
    if not negative.any():raise ValueError('Negative validation days required')
    role_floor=-np.inf
    for _,idx in frame.groupby('role',sort=False).indices.items():
        idx=idx[negative[idx]]
        if len(idx)>=100:
            role_floor=max(role_floor,float(np.quantile(scores[idx],.95,method='higher')))
    candidates=sorted(set(max(role_floor,float(np.quantile(scores[negative],1-rate,method='higher')))
                          for rate in [.0025,.005,.0075,.01]))
    users=sorted(frame.user.unique());nu=len(users)
    uid=pd.Categorical(frame.user,categories=users).codes.astype(np.int64)
    weights=np.random.default_rng(20261102).multinomial(nu,np.full(nu,1/nu),size=replicates)
    den=weights@np.bincount(uid,weights=negative,minlength=nu)
    diagnostics=[];selected=np.inf
    for threshold in candidates:
        fp=negative&(scores>threshold)
        numerator=weights@np.bincount(uid,weights=fp,minlength=nu)
        draws=np.divide(numerator,den,out=np.zeros(len(den)),where=den>0)
        upper=float(np.quantile(draws,.95))
        valid=upper<=.01+1e-15
        diagnostics.append(dict(threshold=threshold,validation_fpr=float(fp.sum()/negative.sum()),
            fpr_upper_95=upper,eligible=valid))
        if valid and not np.isfinite(selected):selected=threshold
    return float(selected),diagnostics
