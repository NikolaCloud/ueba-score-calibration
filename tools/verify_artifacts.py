"""Independent consistency checks on real-data artifacts after computation."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from ueba.calibration.base import Profiles
    from ueba.experiment import methods
    from ueba.metrics import evaluate, stable_ties
    from ueba.splits import calendar_split
    frame = pd.read_parquet(ROOT / "data/processed/user_days.parquet")
    audit = json.loads((ROOT / "data/processed/data_audit.json").read_text())
    assert not frame[["user", "date"]].duplicated().any()
    assert frame.label.sum() == audit["positive_days"]
    checks = []
    for directory in sorted((ROOT / "results").iterdir()):
        if not directory.is_dir() or not (directory / "metrics.csv").exists():
            continue
        config = json.loads((directory / "config.json").read_text())
        blocks, _ = calendar_split(frame, config["fractions"])
        assert blocks["train"].date.max() < blocks["calibration"].date.min()
        assert blocks["calibration"].date.max() < blocks["validation"].date.min()
        assert blocks["validation"].date.max() < blocks["test"].date.min()
        assert len(set(config["features"]) & {"user", "role", "label", "scenario", "date"}) == 0
        saved_metrics = pd.read_csv(directory / "metrics.csv")
        thresholds = pd.read_csv(directory / "thresholds.csv", float_precision="round_trip")
        expected_ties = stable_ties(blocks["test"])
        seed_fingerprints = []
        for seed in config["seeds"]:
            folder = directory / f"seed_{seed}"
            raw = pd.read_parquet(folder / "raw_scores.parquet")
            test = pd.read_parquet(folder / "test_predictions.parquet")
            validation = pd.read_parquet(folder / "validation_predictions.parquet")
            assert len(test) == len(blocks["test"])
            assert len(validation) == len(blocks["validation"])
            assert test[["user", "date", "label"]].reset_index(drop=True).equals(
                blocks["test"][["user", "date", "label"]].reset_index(drop=True))
            for method in methods():
                row = thresholds.loc[thresholds.seed.eq(seed) & thresholds.method.eq(method)].iloc[0]
                np.testing.assert_array_equal(test[f"alert_{method}"], test[f"score_{method}"] > row.threshold)
                point, _ = evaluate(test, test[f"score_{method}"], row.threshold, expected_ties)
                saved = saved_metrics.loc[saved_metrics.seed.eq(seed) & saved_metrics.method.eq(method) &
                                          saved_metrics.operating_point.eq("prospective")].iloc[0]
                for metric, value in point.items():
                    np.testing.assert_allclose(saved[metric], value, equal_nan=True, atol=1e-12)
            seed_fingerprints.append(hashlib.sha256(test.score_M0.to_numpy().tobytes()).hexdigest())
            if seed == 42:
                calibration = raw.loc[raw.block.eq("calibration")]
                raw_test = raw.loc[raw.block.eq("test")].reset_index(drop=True)
                components = Profiles().fit(calibration).components(raw_test)
                for method, model in methods().items():
                    np.testing.assert_allclose(model.transform(raw_test.score, components), test[f"score_{method}"], atol=1e-12)
                model = joblib.load(folder / "detector.joblib")
                selected = np.linspace(0, len(raw_test)-1, min(100, len(raw_test)), dtype=int)
                x = model["preprocessing"].transform(raw_test.iloc[selected][config["features"]])
                np.testing.assert_allclose(-model["detector"].score_samples(x), raw_test.score.iloc[selected], atol=1e-12)
        assert len(set(seed_fingerprints)) == 5, "Seeds produced identical raw models"
        ci = pd.read_csv(directory / "confidence_intervals.csv")
        means = saved_metrics.loc[saved_metrics.operating_point.eq("prospective")].groupby("method").mean(numeric_only=True)
        for row in ci.itertuples():
            np.testing.assert_allclose(row.estimate, means.loc[row.method, row.metric], equal_nan=True, atol=1e-12)
        checks.append(dict(protocol=directory.name, seeds=5, methods=len(methods()),
                           test_days=len(blocks["test"]), all_checks_passed=True))
    reports = ROOT / "reports"
    inventory = {}
    for folder_name in ["ueba", "tools", "tests", "docs", "reports", "results", "data/metadata", "data/processed"]:
        for path in (ROOT / folder_name).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".csv", ".json", ".png", ".svg", ".parquet", ".joblib"}:
                if path.name in {"artifact_inventory.json", "artifact_validation.json"}:
                    continue
                inventory[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (reports / "artifact_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    result = dict(passed=True, data_days=len(frame), positive_days=int(frame.label.sum()),
                  checks=checks, artifact_files=len(inventory))
    (reports / "artifact_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
