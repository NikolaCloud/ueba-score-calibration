"""One-round candidate development using original calibration/validation blocks only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..calibration.base import Profiles
from ..replication.run import frozen_models
from ..metrics import evaluate,ranking_order,stable_ties,threshold_at_rate
from .contextual import ContinuousContext,geometric_score

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/m4_v2_development"
RELEASES={"r4.2":ROOT/"results/primary","r5.2":ROOT/"results/r5_2/primary"}
CANDIDATES=["G30","Z30","EB30","EB90","UZ"]
BASELINES=["M0","M1","M2","M3","M4"]


def metrics(frame,scores,threshold):
    ties=stable_ties(frame)
    result,_=evaluate(frame,scores,threshold,ties)
    order=ranking_order(scores,ties)
    k=int(np.ceil(.01*len(frame)))
    alert=np.zeros(len(frame),bool);alert[order[:k]]=True
    roles=frame[["role","label"]].copy()
    roles["negative"]=roles.label.eq(0)
    roles["fp"]=alert&roles.negative.to_numpy()
    summary=roles.groupby("role")[["negative","fp"]].sum()
    result["max_role_fpr_at_1pct"]=float((summary.fp/summary.negative.replace(0,np.nan)).max())
    return result


def candidate_scores(calibration,validation,components):
    output={"G30":geometric_score(components)}
    for name,mode,ks in [("Z30","fixed",30),("EB30","eb",30),("EB90","eb",90),("UZ","user",30)]:
        output[name]=ContinuousContext(mode=mode,k_scale=ks).fit(calibration).transform(validation)
    return output


def choose_variant(selection_metrics):
    means=selection_metrics.groupby(["release","method"])[["recall_at_1pct","max_role_fpr"]].mean()
    criteria=[]
    for order,name in enumerate(CANDIDATES):
        gains=[];fprs=[];qualified=True;per_release={}
        for release in RELEASES:
            baseline=means.loc[release].loc[BASELINES]
            best_recall=float(baseline.recall_at_1pct.max())
            reference_fpr=float(baseline.loc["M3","max_role_fpr"])
            score=means.loc[(release,name)]
            gain=float(score.recall_at_1pct-best_recall)
            good=bool(gain>=-1e-12 and score.max_role_fpr<=1.1*reference_fpr+1e-12)
            qualified &= good;gains.append(float(score.recall_at_1pct));fprs.append(float(score.max_role_fpr))
            per_release[release]=dict(recall_at_1pct=float(score.recall_at_1pct),best_baseline_recall=best_recall,
                gain_against_best_baseline=gain,max_role_fpr=float(score.max_role_fpr),M3_max_role_fpr=reference_fpr,qualified=good)
        criteria.append(dict(candidate=name,qualified=qualified,mean_recall=float(np.mean(gains)),
                             mean_max_role_fpr=float(np.mean(fprs)),order=order,per_release=per_release))
    qualified=[row for row in criteria if row["qualified"]]
    chosen=(sorted(qualified,key=lambda row:(-row["mean_recall"],row["mean_max_role_fpr"],row["order"]))[0]["candidate"]
            if qualified else "EB30")
    return dict(selected=chosen,qualified=bool(qualified),candidates=criteria,
                rule="Both releases: Recall@1% >= best M0-M4 and max-role FPR <= 1.1*M3; mean recall tie-break",
                validation_status="Exploratory development on previously studied releases; not independent confirmation")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=ROOT/"docs/protocol_m4_v2.md"
    plan=dict(protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),candidates=CANDIDATES,
              source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
              accessed_blocks=["calibration","validation"],seeds=[42,43,44,45,46])
    plan_path=OUT/"development_plan.json"
    if plan_path.exists():
        prior=json.loads(plan_path.read_text())
        if prior["protocol_sha256"]!=plan["protocol_sha256"] or prior["candidates"]!=plan["candidates"]:
            raise ValueError("Development plan changed; start a new declared round instead")
    else:
        plan_path.write_text(json.dumps(plan,indent=2),encoding="utf-8")
    selection_rows=[];diagnostics=[];cache={};partitions=[]
    for release,directory in RELEASES.items():
        for seed in plan["seeds"]:
            raw=pd.read_parquet(directory/f"seed_{seed}/raw_scores.parquet",
                                filters=[("block","in",["calibration","validation"])])
            if not set(raw.block)<={"calibration","validation"}:
                raise ValueError("Test block was accessed")
            calibration=raw.loc[raw.block.eq("calibration")].reset_index(drop=True)
            validation=raw.loc[raw.block.eq("validation")].reset_index(drop=True)
            start=validation.date.min();stop=validation.date.max()+pd.Timedelta(days=1)
            midpoint=start+pd.Timedelta(days=(stop-start).days//2)
            selection=validation.date.lt(midpoint).to_numpy()
            audit=~selection
            if seed==42:
                for name,mask in [("selection",selection),("development_audit",audit)]:
                    part=validation.loc[mask]
                    partitions.append(dict(release=release,block=name,start=str(part.date.min().date()),
                        stop=str(part.date.max().date()),days=len(part),positives=int(part.label.sum()),
                        positive_accounts=int(part.loc[part.label.eq(1),"user"].nunique())))
            components=Profiles().fit(calibration).components(validation)
            values={name:model.transform(validation.score,components) for name,model in frozen_models().items()}
            values.update(candidate_scores(calibration,validation,components))
            saved=pd.read_parquet(directory/f"seed_{seed}/validation_predictions.parquet")
            for name in BASELINES:
                np.testing.assert_allclose(values[name],saved[f"score_{name}"])
            thresholds={name:threshold_at_rate(scores[selection]) for name,scores in values.items()}
            for name,scores in values.items():
                result=metrics(validation.loc[selection].reset_index(drop=True),scores[selection],thresholds[name])
                selection_rows.append(dict(result,release=release,seed=seed,method=name))
            floor=(components["n_user"]>0)&np.isclose(components["p_user"],1/(components["n_user"]+1),rtol=0,atol=1e-14)
            hierarchical=np.exp(-values["M4"])
            selected_floor=floor&selection
            diagnostics.append(dict(release=release,seed=seed,selection_days=int(selection.sum()),
                saturated_user_tail_days=int(selected_floor.sum()),
                saturated_positive_days=int((selected_floor&validation.label.eq(1).to_numpy()).sum()),
                median_hier_probability_inflation=float(np.median(hierarchical[selected_floor]/components["p_user"][selected_floor]))
                    if selected_floor.any() else np.nan))
            cache[(release,seed)]=(validation,audit,values,thresholds)
            print(f"Development scores prepared {release}/{seed}",flush=True)
    selection_metrics=pd.DataFrame(selection_rows)
    selection_metrics.to_csv(OUT/"selection_metrics.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"tail_diagnostics.csv",index=False)
    pd.DataFrame(partitions).to_csv(OUT/"partitions.csv",index=False)
    choice=choose_variant(selection_metrics)
    locked=OUT/"selection.json"
    if locked.exists() and json.loads(locked.read_text())["selected"]!=choice["selected"]:
        raise ValueError("Previously selected variant changed")
    locked.write_text(json.dumps(choice,indent=2),encoding="utf-8")
    print(json.dumps(choice,indent=2),flush=True)
    # The selection is persisted before audit labels are used in metrics.
    audit_rows=[];scenario_rows=[]
    compared=list(dict.fromkeys(BASELINES+[choice["selected"],"UZ"]))
    for (release,seed),(validation,audit,values,thresholds) in cache.items():
        part=validation.loc[audit].reset_index(drop=True)
        output=part[["user","date","role","label","scenario"]].copy()
        for scenario in range(1,6):
            key=f"scenario_{scenario}"
            output[key]=part[key].astype(int) if key in part else part.scenario.eq(scenario).astype(int)
        for name in compared:
            scores=values[name][audit]
            result=metrics(part,scores,thresholds[name])
            audit_rows.append(dict(result,release=release,seed=seed,method=name,threshold=thresholds[name]))
            output[f"score_{name}"]=scores
            output[f"alert_{name}"]=scores>thresholds[name]
            top=ranking_order(scores,stable_ties(part))[:int(np.ceil(.01*len(part)))]
            for scenario in range(1,6):
                flag=output[f"scenario_{scenario}"].to_numpy(bool)
                support=int(flag.sum())
                scenario_rows.append(dict(release=release,seed=seed,method=name,scenario=scenario,
                    positive_days=support,recall_at_1pct=float(flag[top].sum()/support) if support else np.nan))
        folder=OUT/release/f"seed_{seed}"
        folder.mkdir(parents=True,exist_ok=True)
        output.to_parquet(folder/"audit_predictions.parquet",index=False)
    pd.DataFrame(audit_rows).to_csv(OUT/"audit_metrics.csv",index=False)
    pd.DataFrame(scenario_rows).to_csv(OUT/"audit_scenario_metrics.csv",index=False)


if __name__=="__main__":
    main()
