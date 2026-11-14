"""Run prespecified experiments, storing per-seed scores and decisions."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import (GlobalScore, RoleCalibration, UserCalibration,
                          EqualCalibration, HierarchicalCalibration)
from .calibration.base import Profiles
from .metrics import evaluate, stable_ties, threshold_at_rate
from .splits import calendar_split

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44, 45, 46]


def methods():
    result = dict(M0=GlobalScore(), M1=RoleCalibration(), M2=UserCalibration(),
                  M3=EqualCalibration(), M4=HierarchicalCalibration(),
                  A_no_user=HierarchicalCalibration(use_user=False),
                  A_no_role=HierarchicalCalibration(use_role=False),
                  A_fixed=HierarchicalCalibration(fixed_weights=True))
    for ku in [10, 30, 90]:
        for kr in [50, 200, 800]:
            if (ku, kr) != (30, 200):
                result[f"S_ku{ku}_kr{kr}"] = HierarchicalCalibration(ku, kr)
    return result


def run_protocol(frame, features, name, fractions, model_map=None, history_caps=None, scenario_ids=None):
    models = methods() if model_map is None else model_map
    caps = [1, 3, 5, 10, 20, 30, 60] if history_caps is None else history_caps
    scenarios = [1, 2, 3] if scenario_ids is None else list(scenario_ids)
    directory = ROOT / "results" / name
    directory.mkdir(parents=True, exist_ok=True)
    blocks, support = calendar_split(frame, fractions)
    pd.DataFrame(support).to_csv(directory / "split_support.csv", index=False)
    # Save support and the fixed config before any held-out model scores are computed.
    config = dict(protocol=name, fractions=fractions, seeds=SEEDS, features=features,
                  n_estimators=200, max_samples=min(2048, len(blocks["train"])),
                  k_user=30, k_role=200, history="frozen calibration only",
                  role_min_history=30, user_min_history=5, methods=list(models),
                  threshold_comparison=">", target_alert_rate=.01, bootstrap_replicates=1000,
                  timestamp_utc=pd.Timestamp.now(tz="UTC").isoformat())
    config_file = directory / "config.json"
    if not config_file.exists():
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    (directory / "config.sha256").write_text(config_hash + "\n", encoding="utf-8")
    records, roles, thresholds, scenario_rows, history_rows, activity_rows = [], [], [], [], [], []
    truncated_rows = []
    ties = {key: stable_ties(blocks[key]) for key in ("validation", "test")}
    train_users = set(blocks["train"].user)
    unseen = blocks["test"].loc[~blocks["test"].user.isin(train_users)]
    (directory / "unseen_users.json").write_text(json.dumps(dict(days=len(unseen),
        users=int(unseen.user.nunique()), positives=int(unseen.label.sum()),
        note="Natural unseen-user subset only; no retrospective artificial withholding"), indent=2), encoding="utf-8")
    print(name, json.dumps(support), flush=True)
    for seed in SEEDS:
        started = time.monotonic()
        seed_dir = directory / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        score_path = seed_dir / "raw_scores.parquet"
        if score_path.exists():
            cached = pd.read_parquet(score_path)
            current = pd.concat([blocks[k].assign(block=k) for k in ["calibration", "validation", "test"]])
            current = current.reset_index(drop=True)
            if not cached[["user", "date", "block"]].equals(current[["user", "date", "block"]]):
                raise ValueError("Stale score cache")
            scored = {key: cached.loc[cached.block.eq(key)].copy().reset_index(drop=True)
                      for key in ["calibration", "validation", "test"]}
        else:
            preprocessing = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
            xtrain = preprocessing.fit_transform(blocks["train"][features])
            detector = IsolationForest(n_estimators=200, max_samples=min(2048, len(xtrain)),
                                       random_state=seed, n_jobs=-1)
            detector.fit(xtrain)
            joblib.dump(dict(preprocessing=preprocessing, detector=detector, features=features,
                             config_sha256=config_hash), seed_dir / "detector.joblib")
            scored = {}
            for key in ["calibration", "validation", "test"]:
                scored[key] = blocks[key].copy().reset_index(drop=True)
                x = preprocessing.transform(scored[key][features])
                with parallel_backend("threading", n_jobs=6):
                    scored[key]["score"] = -detector.score_samples(x)
                scored[key]["block"] = key
            pd.concat(scored.values(), ignore_index=True).to_parquet(score_path, index=False)
        profile = Profiles().fit(scored["calibration"])
        comp = {key: profile.components(scored[key]) for key in ["validation", "test"]}
        metadata_columns = ["user", "date", "role", "label", "scenario"]
        metadata_columns += [c for c in scored["test"] if c.startswith("scenario_") or c == "incident_ids"]
        outputs = {key: scored[key][metadata_columns].copy()
                   for key in ["validation", "test"]}
        for key in outputs:
            outputs[key]["n_user"] = comp[key]["n_user"]
            outputs[key]["n_role"] = comp[key]["n_role"]
            outputs[key]["unseen_train"] = ~outputs[key].user.isin(train_users)
        for method, calibrator in models.items():
            values = {key: calibrator.transform(scored[key].score, comp[key]) for key in outputs}
            threshold = threshold_at_rate(values["validation"])
            negative_validation = scored["validation"].label.eq(0).to_numpy()
            fpr_threshold = threshold_at_rate(values["validation"][negative_validation])
            thresholds.append(dict(protocol=name, seed=seed, method=method, threshold=threshold,
                                   validation_alert_rate=float((values["validation"]>threshold).mean()),
                                   fpr_threshold=fpr_threshold,
                                   validation_fpr=float((values["validation"][negative_validation]>fpr_threshold).mean())))
            for key in outputs:
                outputs[key][f"score_{method}"] = values[key]
                outputs[key][f"alert_{method}"] = values[key] > threshold
            for label, th in [("prospective", threshold), ("fpr_constrained", fpr_threshold)]:
                metrics, group = evaluate(scored["test"], values["test"], th, ties["test"])
                metrics.update(protocol=name, seed=seed, method=method, operating_point=label,
                               threshold=th, test_fpr_exceeds_1pct=bool(metrics["fpr"] > .01))
                records.append(metrics)
                if label == "prospective":
                    roles.append(group.assign(protocol=name, seed=seed, method=method))
            if method in {"M0", "M1", "M2", "M3", "M4"}:
                test = scored["test"]
                alert = values["test"] > threshold
                for scenario in scenarios:
                    membership = f"scenario_{scenario}"
                    mask = (test[membership].eq(1) if membership in test else test.scenario.eq(scenario)).to_numpy()
                    scenario_rows.append(dict(seed=seed, method=method, scenario=scenario,
                        positive_days=int(mask.sum()), detected=int((alert & mask).sum()),
                        recall=float(alert[mask].mean()) if mask.any() else np.nan))
                history = pd.cut(comp["test"]["n_user"], [-1, 0, 4, 14, 29, np.inf],
                                 labels=["0", "1-4", "5-14", "15-29", "30+"])
                for bucket in history.categories:
                    mask = np.asarray(history == bucket)
                    y = test.label.to_numpy(bool)
                    neg, pos = mask & ~y, mask & y
                    history_rows.append(dict(seed=seed, method=method, history=str(bucket),
                        days=int(mask.sum()), negative=int(neg.sum()), positive=int(pos.sum()),
                        fp=int((neg & alert).sum()), tp=int((pos & alert).sum()),
                        fpr=float(alert[neg].mean()) if neg.any() else np.nan,
                        recall=float(alert[pos].mean()) if pos.any() else np.nan))
                for activity in ["logon", "device", "file", "email"]:
                    mask = (test[f"{activity}_events"].to_numpy() > 0) & test.label.eq(0).to_numpy()
                    activity_rows.append(dict(seed=seed, method=method, activity=activity,
                        negative_days=int(mask.sum()), fp=int((mask & alert).sum()),
                        fpr=float(alert[mask].mean()) if mask.any() else np.nan))
                if len(unseen):
                    mask = ~test.user.isin(train_users)
                    subset = test.loc[mask].reset_index(drop=True)
                    um, _ = evaluate(subset, values["test"][mask], threshold, stable_ties(subset))
                    records.append(dict(um, protocol=name, seed=seed, method=method,
                                        operating_point="natural_unseen", threshold=threshold))
        for key, output in outputs.items():
            output.to_parquet(seed_dir / f"{key}_predictions.parquet", index=False)
        for cap in caps:
            capped_profile = Profiles(max_user_history=cap).fit(scored["calibration"])
            capped = {key: capped_profile.components(scored[key]) for key in outputs}
            for method in ["M2", "M3", "M4"]:
                model = models[method]
                vs = model.transform(scored["validation"].score, capped["validation"])
                ts = model.transform(scored["test"].score, capped["test"])
                threshold = threshold_at_rate(vs)
                metrics, _ = evaluate(scored["test"], ts, threshold, ties["test"])
                truncated_rows.append(dict(metrics, seed=seed, method=method, history_cap=cap))
        print(f"{name}, seed {seed}: {time.monotonic()-started:.1f}s", flush=True)
    pd.DataFrame(records).to_csv(directory / "metrics.csv", index=False)
    pd.concat(roles, ignore_index=True).to_csv(directory / "role_metrics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(directory / "thresholds.csv", index=False)
    pd.DataFrame(scenario_rows).to_csv(directory / "scenario_metrics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(directory / "history_metrics.csv", index=False)
    pd.DataFrame(activity_rows).to_csv(directory / "activity_metrics.csv", index=False)
    pd.DataFrame(truncated_rows).to_csv(directory / "history_truncation.csv", index=False)
    return support[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    frame = pd.read_parquet(ROOT / "data/processed/user_days.parquet")
    audit = json.loads((ROOT / "data/processed/data_audit.json").read_text())
    features = audit["features"]
    forbidden = {"label", "scenario", "labelled_events", "user", "date", "role", "event_types"}
    if forbidden & set(features):
        raise ValueError("Target or metadata leakage into features")
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        versions={package: importlib.metadata.version(package) for package in
                  ["numpy", "pandas", "scikit-learn", "matplotlib", "scipy", "statsmodels", "pyarrow", "pytest"]})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    primary = run_protocol(frame, features, "primary", [.5, .2, .15, .15])
    if not primary["reliable_support"] and not args.primary_only:
        for i, fractions in enumerate([[.3, .15, .1, .15], [.4, .15, .1, .15], [.5, .15, .1, .15]], 1):
            run_protocol(frame, features, f"temporal_{i}", fractions)


if __name__ == "__main__":
    main()
