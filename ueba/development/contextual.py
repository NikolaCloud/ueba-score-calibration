"""Continuous robust contextual scores with empirical-Bayes partial pooling."""
import numpy as np
import pandas as pd


def robust_summary(values):
    values=np.asarray(values,float)
    center=float(np.median(values))
    scale=float(1.4826*np.median(np.abs(values-center)))
    return center,scale


class ContinuousContext:
    """Hierarchical conditional normalization; outputs a score, not a p-value."""
    def __init__(self,mode="eb",k_scale=30,k_user=30,k_role=200,scale_floor=.1):
        if mode not in {"eb","fixed","user"} or min(k_scale,k_user,k_role,scale_floor)<=0:
            raise ValueError("Invalid contextual normalization parameters")
        self.mode,self.k_scale,self.k_user,self.k_role,self.scale_floor=mode,k_scale,k_user,k_role,scale_floor

    def fit(self,calibration):
        if calibration.empty or not {"user","role","date","score"}<=set(calibration):
            raise ValueError("Nonempty dated calibration required")
        if not np.isfinite(calibration.score).all() or calibration[["user","date"]].duplicated().any():
            raise ValueError("Invalid calibration observations")
        self.last_date=calibration.date.max()
        rows=[]
        for (user,role),part in calibration.groupby(["user","role"],sort=False):
            center,scale=robust_summary(part.score)
            rows.append(dict(user=str(user),role=str(role),n=len(part),center=center,scale=scale))
        profiles=pd.DataFrame(rows)
        self.global_center,_=robust_summary(profiles.center)
        positive=profiles.loc[profiles.scale>0,"scale"].to_numpy()
        _,raw_global_scale=robust_summary(calibration.score)
        self.within_global=float(np.median(positive)) if len(positive) else max(raw_global_scale,1e-6)
        self.floor=max(1e-9,self.scale_floor*self.within_global)
        _,between_scale=robust_summary(profiles.center)
        noise=(np.pi/2)*np.maximum(profiles.scale.to_numpy(),self.floor)**2/profiles.n.to_numpy()
        self.between_global=max(between_scale**2-float(np.median(noise)),self.floor**2)
        role_rows=[]
        for role,part in profiles.groupby("role",sort=False):
            center,between=robust_summary(part.center)
            scales=part.loc[part.scale>0,"scale"].to_numpy()
            raw_within=float(np.median(scales)) if len(scales) else self.within_global
            n=int(part.n.sum())
            wr=n/(n+self.k_role)
            within=np.sqrt(wr*raw_within**2+(1-wr)*self.within_global**2)
            var_mean=(np.pi/2)*max(within,self.floor)**2/max(len(part),1)
            role_rows.append(dict(role=role,n=n,n_users=len(part),center=center,within=within,
                                  between=max(between**2-float(np.median((np.pi/2)*np.maximum(part.scale,self.floor)**2/part.n)),self.floor**2),
                                  noise=var_mean))
        role_table=pd.DataFrame(role_rows)
        _,role_between=robust_summary(role_table.center)
        self.role_signal=max(role_between**2-float(np.median(role_table.noise)),self.floor**2)
        self.roles={}
        for row in role_table.itertuples():
            wr=(row.n/(row.n+self.k_role) if self.mode=="fixed" else
                self.role_signal/(self.role_signal+row.noise))
            center=wr*row.center+(1-wr)*self.global_center
            self.roles[row.role]=dict(center=center,within=max(row.within,self.floor),
                between=row.between,n=row.n,center_uncertainty=(1-wr)*self.role_signal)
        self.users={}
        for row in profiles.itertuples():
            parent=self.roles[row.role]
            signal=parent["between"]
            noise=(np.pi/2)*max(row.scale,self.floor)**2/row.n
            wu=(row.n/(row.n+self.k_user) if self.mode=="fixed" else signal/(signal+noise))
            if self.mode=="user":
                wu=1.
            center=wu*row.center+(1-wu)*parent["center"]
            ws=row.n/(row.n+self.k_scale)
            variance=ws*row.scale**2+(1-ws)*parent["within"]**2
            variance=max(variance,self.floor**2)+(1-wu)*signal
            self.users[(row.user,row.role)]=dict(center=float(center),scale=float(np.sqrt(variance)),
                                                n=int(row.n),weight=float(wu))
        self.profile_table=profiles
        return self

    def transform(self,targets):
        if len(targets) and targets.date.min()<=self.last_date:
            raise ValueError("Targets must follow frozen calibration")
        if not np.isfinite(targets.score).all():
            raise ValueError("Nonfinite target score")
        scores=targets.score.to_numpy(float)
        output=np.empty(len(targets),float)
        for (user,role),indices in targets.groupby(["user","role"],sort=False).indices.items():
            person=self.users.get((str(user),str(role)))
            if person is not None:
                center,scale=person["center"],person["scale"]
            else:
                parent=self.roles.get(str(role))
                if parent is not None:
                    center=parent["center"]
                    scale=np.sqrt(parent["within"]**2+parent["between"]+parent["center_uncertainty"])
                else:
                    center=self.global_center
                    scale=np.sqrt(self.within_global**2+self.between_global)
            output[indices]=(scores[indices]-center)/max(scale,self.floor)
        return output


def geometric_score(components,k_user=30,k_role=200):
    c=components
    wu=c["n_user"]/(c["n_user"]+k_user)
    wr=c["n_role"]/(c["n_role"]+k_role)
    return -(wu*np.log(c["p_user"])+(1-wu)*(wr*np.log(c["p_role"])+(1-wr)*np.log(c["p_global"])))
