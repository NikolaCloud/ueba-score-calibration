import importlib.util
from pathlib import Path

import pandas as pd


def test_answer_days_include_impersonation_exclude_benign_replies():
    root = Path(__file__).resolve().parents[1]
    path = root / "data/processed/answer_events.parquet"
    if not path.exists():
        import pytest
        pytest.skip("Official answer audit must be run first")
    events = pd.read_parquet(path)
    impersonated = events.loc[events.user.ne(events.incident_insider)]
    assert len(impersonated) == 30
    assert set(impersonated.event_type) == {"email", "logon"}
    for _, group in events.loc[events.scenario.eq(3)].groupby("key"):
        transfer_day = group.loc[group.event_type.eq("file"), "date"].min()
        assert (group.loc[group.user.ne(group.incident_insider), "date"] > transfer_day).all()
    assert not events[["event_type", "event_id"]].duplicated().any()
