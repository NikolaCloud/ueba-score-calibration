"""Reconcile r5.2 master incidents with event-level observables."""
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def build_answers():
    output = ROOT / "data/processed/r5.2"
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ROOT / "data/raw/answers.tar.bz2", "r:bz2") as archive:
        master = pd.read_csv(io.BytesIO(archive.extractfile("answers/insiders.csv").read()), dtype=str)
        master = master.loc[master.dataset.eq("5.2")].copy()
        if master.details.duplicated().any():
            raise ValueError("Duplicated master incident key")
        mapping = master.set_index("details")
        seen_files, records = set(), []
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or "r5.2-" not in name:
                continue
            if name not in mapping.index:
                raise ValueError(f"Unlisted official incident key: {name}")
            incident = mapping.loc[name]
            seen_files.add(name)
            rows = csv.reader(archive.extractfile(member).read().decode("utf-8").splitlines())
            for row in rows:
                if not row:
                    continue
                if len(row) < 5:
                    raise ValueError("Malformed official observable")
                kind, event_id, stamp, actor, pc = row[:5]
                records.append(dict(event_type=kind, event_id=event_id, timestamp=stamp,
                    user=actor, pc=pc, scenario=int(incident.scenario), incident_id=name,
                    incident_insider=incident.user, master_start=incident.start, master_end=incident.end,
                    key_owner_matches_account=actor == incident.user))
    if seen_files != set(master.details):
        raise ValueError("Master incidents and actual keys do not match")
    references = pd.DataFrame(records)
    references["date"] = pd.to_datetime(references.timestamp, format="%m/%d/%Y %H:%M:%S").dt.normalize()
    reconciled = []
    for _, group in references.groupby(["event_type", "event_id"], sort=False):
        if len(group[["user", "pc", "timestamp", "scenario"]].drop_duplicates()) != 1:
            raise ValueError("Contradictory observable metadata in official keys")
        preferred = group.loc[group.key_owner_matches_account]
        row = (preferred if len(preferred) else group).iloc[0].to_dict()
        row["key_references"] = ";".join(sorted(group.incident_id.unique()))
        reconciled.append(row)
    events = pd.DataFrame(reconciled)
    events.to_parquet(output / "answer_events.parquet", index=False)
    references.to_parquet(output / "answer_references.parquet", index=False)
    days = events.groupby(["user", "date"]).agg(labelled_events=("event_id", "size"),
        scenario_ids=("scenario", lambda x: ";".join(map(str, sorted(set(x))))),
        event_types=("event_type", lambda x: ";".join(sorted(set(x)))),
        incident_ids=("incident_id", lambda x: ";".join(sorted(set(x))))).reset_index()
    for scenario in range(1, 6):
        days[f"scenario_{scenario}"] = days.scenario_ids.map(lambda value: str(scenario) in value.split(";")).astype(int)
    days["scenario"] = days.scenario_ids.map(lambda x: int(x) if ";" not in x else 0)
    days["label"] = 1
    days.to_parquet(output / "answer_days.parquet", index=False)
    master.to_csv(output / "official_incidents.csv", index=False)
    audit = dict(release="r5.2", official_incidents=len(master), official_key_files=len(seen_files),
        incidents_by_scenario={str(k):int(v) for k,v in master.groupby("scenario").size().items()},
        raw_observable_references=len(references), unique_events=len(events),
        repeated_references_removed=len(references)-len(events),
        other_account_references=references.loc[~references.key_owner_matches_account].to_dict("records"),
        positive_days=len(days), positive_accounts=int(days.user.nunique()),
        days_by_scenario={str(i):int(days[f"scenario_{i}"].sum()) for i in range(1, 6)},
        multi_scenario_days=int(days.scenario_ids.str.contains(";").sum()),
        earliest=str(days.date.min().date()), latest=str(days.date.max().date()),
        interpretation="Official incident-observable account-days, including precursor/context events; not all days of an insider.",
        cross_reference_resolution="Two scenario-4 observables also occur in their actual account owner's incident key; deduplicated by (event_type,id), preserving all key references.",
        absent_scenarios=[i for i in range(1,6) if str(i) not in set(master.scenario)])
    # Timestamps in the diagnostic records are stringified for portable JSON.
    (output / "answer_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k:v for k,v in audit.items() if k != "other_account_references"}, indent=2), flush=True)


if __name__ == "__main__":
    build_answers()
