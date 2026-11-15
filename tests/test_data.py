import pandas as pd
import pytest

from ueba.data import attach_roles, external_recipients
from ueba.splits import calendar_split


def test_roles_never_come_from_future(tmp_path):
    pd.DataFrame(dict(user_id=["A"], role=["old"])).to_csv(tmp_path / "2020-02.csv", index=False)
    pd.DataFrame(dict(user_id=["A"], role=["new"])).to_csv(tmp_path / "2020-03.csv", index=False)
    days = pd.DataFrame(dict(user=["A"]*3, date=pd.to_datetime(["2020-01-15", "2020-02-15", "2020-03-15"])))
    result = attach_roles(days, tmp_path)
    assert result.role.tolist() == ["UNKNOWN", "old", "new"]


def test_external_domain_boundaries_and_duplicates():
    assert external_recipients("a@dtaa.com;b@evil.com;b@evil.com;c@notdtaa.com;d@DTAA.COM") == 2


def test_calendar_boundaries_and_disjoint_keys():
    frame = pd.DataFrame(dict(user=["A"]*100, date=pd.date_range("2020-01-01", periods=100), label=[0]*100))
    blocks, support = calendar_split(frame)
    assert [len(x) for x in blocks.values()] == [50, 20, 15, 15]
    assert blocks["train"].date.max() < blocks["calibration"].date.min()
    assert len(pd.concat(blocks.values())) == 100
    with pytest.raises(ValueError):
        calendar_split(pd.concat([frame, frame.iloc[:1]]))
