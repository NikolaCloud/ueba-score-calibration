"""Independently check r5.2 predictions, metadata, model locks, and reported estimates."""
import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ueba.calibration.base import Profiles
from ueba.metrics import evaluate,ranking_order,stable_ties
from ueba.replication.run import PROTOCOLS,frozen_models,lock_methods
from ueba.splits import calendar_split


def main():
    output=ROOT/"results/r5_2"
    frame=pd.read_parquet(ROOT/"data/processed/r5.2/user_days.parquet")
    audit=json.loads((ROOT/"data/processed/r5.2/data_audit.json").read_text())
    plan=json.loads((output/"preflight_plan.json").read_text())
    assert plan["locked_methods"]==lock_methods()
    assert plan["data_sha256"]==hashlib.sha256((ROOT/"data/processed/r5.2/user_days.parquet").read_bytes()).hexdigest()
    assert not frame[["user","date"]].duplicated().any()
    assert int(frame.label.sum())==audit["positive_days"]
    labelled=pd.read_parquet(ROOT/"data/processed/r5.2/answer_days.parquet")
    official_keys=pd.MultiIndex.from_frame(labelled[["user","date"]])
    observed_keys=pd.MultiIndex.from_frame(frame[["user","date"]])
    np.testing.assert_array_equal(frame.label.to_numpy(),observed_keys.isin(official_keys).astype(int))
    dated=frame.snapshot_date.notna()
    assert (frame.loc[dated,"snapshot_date"]<=frame.loc[dated,"date"]).all()
    assert set(audit["features"])==set(json.loads((ROOT/"data/processed/data_audit.json").read_text())["features"])
    checks=[]
    for period,fractions in PROTOCOLS.items():
        directory=output/period
        blocks,_=calendar_split(frame,fractions)
        assert blocks["train"].date.max()<blocks["calibration"].date.min()
        assert blocks["calibration"].date.max()<blocks["validation"].date.min()
        assert blocks["validation"].date.max()<blocks["test"].date.min()
        config=json.loads((directory/"config.json").read_text())
        assert config["k_user"]==30 and config["k_role"]==200 and config["n_estimators"]==200
        assert config["seeds"]==[42,43,44,45,46] and config["methods"]==list(frozen_models())
        thresholds=pd.read_csv(directory/"thresholds.csv",float_precision="round_trip")
        metrics=pd.read_csv(directory/"metrics.csv",float_precision="round_trip")
        ci=pd.read_csv(directory/"confidence_intervals.csv")
        expected={method:{} for method in frozen_models()}
        fingerprints=[]
        for seed in config["seeds"]:
            folder=directory/f"seed_{seed}"
            test=pd.read_parquet(folder/"test_predictions.parquet")
            keys=["user","date","label",*[f"scenario_{i}" for i in range(1,6)]]
            assert test[keys].equals(blocks["test"][keys].reset_index(drop=True))
            ties=stable_ties(test)
            model=joblib.load(folder/"detector.joblib")
            parameters=model["detector"].get_params()
            assert parameters["n_estimators"]==200 and parameters["n_jobs"]==-1 and parameters["random_state"]==seed
            assert parameters["max_samples"]==min(2048,len(blocks["train"]))
            assert model["features"]==audit["features"]
            for method,calibrator in frozen_models().items():
                threshold=thresholds.loc[thresholds.seed.eq(seed)&thresholds.method.eq(method),"threshold"].iloc[0]
                np.testing.assert_array_equal(test[f"alert_{method}"],test[f"score_{method}"]>threshold)
                point,_=evaluate(test,test[f"score_{method}"],threshold,ties)
                saved=metrics.loc[metrics.seed.eq(seed)&metrics.method.eq(method)&metrics.operating_point.eq("prospective")].iloc[0]
                for metric,value in point.items():
                    np.testing.assert_allclose(value,saved[metric],atol=1e-12,equal_nan=True)
                for metric in ["recall_at_1pct","max_role_fpr","max_role_fpr_n100","recall","fpr","precision_at_1pct"]:
                    expected[method].setdefault(metric,[]).append(point[metric])
                order=ranking_order(test[f"score_{method}"],ties)
                top=order[:int(np.ceil(.01*len(test)))]
                for scenario in range(1,6):
                    membership=test[f"scenario_{scenario}"].to_numpy(bool)
                    n=membership.sum()
                    for kind,numerator in [("recall_at_1pct",membership[top].sum()),
                                           ("recall",(membership&test[f"alert_{method}"].to_numpy()).sum())]:
                        expected[method].setdefault(f"scenario_{scenario}_{kind}",[]).append(numerator/n if n else np.nan)
            fingerprints.append(hashlib.sha256(test.score_M0.to_numpy().tobytes()).hexdigest())
            if seed==42:
                train=blocks["train"][audit["features"]].to_numpy(float)
                imputer=model["preprocessing"].named_steps["simpleimputer"]
                scaler=model["preprocessing"].named_steps["standardscaler"]
                np.testing.assert_allclose(imputer.statistics_,np.nanmedian(train,axis=0))
                np.testing.assert_allclose(scaler.mean_,np.mean(imputer.transform(blocks["train"][audit["features"]]),axis=0))
                raw=pd.read_parquet(folder/"raw_scores.parquet")
                calibration=raw.loc[raw.block.eq("calibration")]
                raw_test=raw.loc[raw.block.eq("test")].reset_index(drop=True)
                components=Profiles().fit(calibration).components(raw_test)
                for method,calibrator in frozen_models().items():
                    np.testing.assert_allclose(calibrator.transform(raw_test.score,components),test[f"score_{method}"],atol=1e-12)
                selected=np.linspace(0,len(test)-1,min(100,len(test)),dtype=int)
                x=model["preprocessing"].transform(raw_test.iloc[selected][audit["features"]])
                np.testing.assert_allclose(-model["detector"].score_samples(x),raw_test.score.iloc[selected],atol=1e-12)
        assert len(set(fingerprints))==5
        for row in ci.itertuples():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",RuntimeWarning)
                expected_point=np.nanmean(expected[row.method][row.metric])
            np.testing.assert_allclose(row.estimate,expected_point,atol=1e-12,equal_nan=True)
        checks.append(dict(period=period,seeds=5,methods=5,test_days=len(blocks["test"]),passed=True))
    pairs=pd.read_csv(output/"paired_comparisons.csv")
    for column in ["p_holm_primary","p_holm_secondary","p_holm_scenario"]:
        selected=pairs[column].notna()
        assert (pairs.loc[selected,column]+1e-12>=pairs.loc[selected,"p_centered_bootstrap"]).all()
        assert pairs.loc[selected,column].between(0,1).all()
    assert pairs.p_holm_primary.notna().sum()==2
    result=dict(passed=True,release="r5.2",data_days=len(frame),positive_days=int(frame.label.sum()),
                scenarios_present=[i for i in range(1,6) if frame[f"scenario_{i}"].sum()],
                parameters_match_r4_2_bundle=plan["locked_methods"]["matched_original_r4_2_bundle"],periods=checks)
    (ROOT/"reports/artifact_validation_r5_2.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    inventory={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*")
               if p.is_file() and p.suffix in {".csv",".json",".sha256",".png",".svg"}}
    (ROOT/"reports/artifact_inventory_r5_2.json").write_text(json.dumps(inventory,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
