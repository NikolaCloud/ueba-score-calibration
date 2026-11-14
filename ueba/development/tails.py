"""Exploratory continuous tails and causal accumulation, separate from frozen M4."""
import numpy as np
import pandas as pd


def tail_scale(values):
    values=np.asarray(values,float)
    q=np.quantile(values,.75)
    return float(np.mean(values[values>=q]-q))


class ExtrapolatedProfiles:
    def fit(self,calibration):
        if calibration.empty or calibration[['user','date']].duplicated().any():
            raise ValueError('Nonempty unique calibration required')
        if not np.isfinite(calibration.score).all():
            raise ValueError('Nonfinite scores')
        self.last_date=calibration.date.max()
        raw_users={(str(u),str(r)):np.sort(p.score.to_numpy(float))
                   for (u,r),p in calibration.groupby(['user','role'],sort=False)}
        scales={key:tail_scale(ref) for key,ref in raw_users.items()}
        positive=[b for b in scales.values() if b>0]
        self.global_scale=float(np.median(positive)) if positive else max(float(calibration.score.std(ddof=0)),1e-6)
        self.floor=max(1e-9,.1*self.global_scale)
        self.global_ref=np.sort(calibration.score.to_numpy(float))
        self.roles={}
        for role,part in calibration.groupby('role',sort=False):
            role=str(role)
            local=[b for (u,r),b in scales.items() if r==role and b>0]
            b=float(np.median(local)) if local else self.global_scale
            w=len(part)/(len(part)+200)
            self.roles[role]=(np.sort(part.score.to_numpy(float)),max(w*b+(1-w)*self.global_scale,self.floor))
        self.users={}
        for (user,role),ref in raw_users.items():
            parent=self.roles[role][1]
            w=len(ref)/(len(ref)+30)
            self.users[(user,role)]=(ref,max(w*scales[(user,role)]+(1-w)*parent,self.floor))
        return self

    @staticmethod
    def log_tail(reference,values,scale):
        knots,first=np.unique(reference,return_index=True)
        probabilities=(1+len(reference)-first)/(len(reference)+1)
        p=np.interp(values,knots,probabilities,left=1.,right=probabilities[-1])
        logp=np.log(p)
        beyond=values>knots[-1]
        logp[beyond]-=(values[beyond]-knots[-1])/scale
        return logp

    def components(self,targets):
        if len(targets) and targets.date.min()<=self.last_date:
            raise ValueError('Targets must follow frozen calibration')
        scores=targets.score.to_numpy(float)
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite targets')
        pg=self.log_tail(self.global_ref,scores,self.global_scale)
        pr,pu=pg.copy(),pg.copy()
        nr,nu=np.zeros(len(targets)),np.zeros(len(targets))
        for role,idx in targets.groupby('role',sort=False).indices.items():
            profile=self.roles.get(str(role))
            if profile is not None:
                ref,b=profile
                pr[idx]=self.log_tail(ref,scores[idx],b);nr[idx]=len(ref)
        pu=pr.copy()
        for (user,role),idx in targets.groupby(['user','role'],sort=False).indices.items():
            profile=self.users.get((str(user),str(role)))
            if profile is not None:
                ref,b=profile
                pu[idx]=self.log_tail(ref,scores[idx],b);nu[idx]=len(ref)
        return dict(log_global=pg,log_role=pr,log_user=pu,n_user=nu,n_role=nr)


def combine(c,k_user=30):
    wu=c['n_user']/(c['n_user']+k_user)
    wr=c['n_role']/(c['n_role']+200)
    return -(wu*c['log_user']+(1-wu)*(wr*c['log_role']+(1-wr)*c['log_global']))


def causal_mean(frame,scores):
    work=frame[['user','date']].copy()
    work['value']=np.asarray(scores,float)
    work['position']=np.arange(len(work))
    output=np.empty(len(work))
    if work[['user','date']].duplicated().any():
        raise ValueError('Duplicate user-day')
    for _,part in work.groupby('user',sort=False):
        part=part.sort_values('date')
        output[part.position.to_numpy()]=part.set_index('date').value.rolling('7D',closed='right').mean().to_numpy()
    return output
