"""Small deterministic integration fixture; outputs never enter research results."""
import numpy as np
import pandas as pd

from ueba import experiment
from ueba.statistics import bootstrap_protocol
from ueba.report import build_protocol


def test_real_pipeline_and_bootstrap_agree_on_point_estimates(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    rows = []
    for day in pd.date_range("2020-01-01", periods=100):
        for user in range(8):
            rows.append(dict(user=f"u{user}", date=day, role=f"r{user%2}",
                             label=int(day.dayofyear%9 == 0 and user%3 == 0),
                             scenario=int(day.dayofyear%9 == 0 and user%3 == 0),
                             activity=int(rng.poisson(8)+user),
                             logon_events=2, device_events=int(user%2), file_events=1, email_events=3))
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    experiment.run_protocol(frame, ["activity"], "primary", [.5, .2, .15, .15])
    directory = tmp_path / "results/primary"
    bootstrap_protocol(directory, n_bootstrap=20)
    metrics = pd.read_csv(directory / "metrics.csv")
    point = metrics.loc[metrics.operating_point.eq("prospective")].groupby("method").mean(numeric_only=True)
    ci = pd.read_csv(directory / "confidence_intervals.csv")
    for row in ci.itertuples():
        np.testing.assert_allclose(row.estimate, point.loc[row.method, row.metric], atol=1e-12)
    thresholds = pd.read_csv(directory / "thresholds.csv")
    assert (thresholds.validation_alert_rate <= .01+1/100).all()
    assert (directory / "seed_46/test_predictions.parquet").exists()
    main, _, _ = build_protocol(directory)
    assert main.index.tolist() == ["M0", "M1", "M2", "M3", "M4"]
    assert (directory / "figures/figure_3_precision_recall.svg").stat().st_size > 1000
