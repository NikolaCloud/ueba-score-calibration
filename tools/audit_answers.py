"""Build event-level labels from incident keys; exclude other actors in the key."""
import csv
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_labels():
    destination = ROOT / "data/processed"
    destination.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    events = []
    with tarfile.open(ROOT / "data/raw/answers.tar.bz2", "r:bz2") as archive:
        for member in archive:
            if not member.isfile():
                continue
            if member.name.endswith(("readme.txt", "scenarios.txt")):
                text = archive.extractfile(member).read().decode("utf-8")
                print(member.name, text, flush=True)
                (ROOT / "data/metadata" / Path(member.name).name).write_text(text, encoding="utf-8")
            match = re.search(r"r4\.2-([123])-([A-Z]{3}[0-9]{4})\.csv$", member.name)
            if not match:
                continue
            scenario, insider = match.groups()
            text = archive.extractfile(member).read().decode("utf-8")
            rows = [r for r in csv.reader(text.splitlines()) if r]
            transfer_dates = [pd.to_datetime(r[2], format="%m/%d/%Y %H:%M:%S").normalize()
                              for r in rows if r[0] == "file"]
            if scenario == "3" and not transfer_dates:
                raise ValueError("Scenario 3 requires a verified keylogger-transfer day")
            for row in rows:
                if not row:
                    continue
                if len(row) < 5:
                    raise ValueError((member.name, row))
                kind, event_id, timestamp, actor, pc = row[:5]
                counts[f"scenario_{scenario}_{kind}_all"] += 1
                if actor != insider:
                    # The key includes benign supervisor replies on the setup day and
                    # malicious activity under stolen supervisor credentials the next day.
                    day = pd.to_datetime(timestamp, format="%m/%d/%Y %H:%M:%S").normalize()
                    impersonation = (scenario == "3" and day > min(transfer_dates)
                                     and kind in {"email", "logon"})
                    if not impersonation:
                        counts["benign_other_actor_observables_excluded"] += 1
                        continue
                    counts["impersonated_account_attack_events_included"] += 1
                events.append(dict(event_type=kind, event_id=event_id, timestamp=timestamp,
                                   user=actor, incident_insider=insider, pc=pc,
                                   scenario=int(scenario), key=member.name))
    frame = pd.DataFrame(events)
    if frame[["event_type", "event_id"]].duplicated().any():
        # An event can occur in two keys; require invariant metadata before combining.
        for _, group in frame[frame[["event_type", "event_id"]].duplicated(False)].groupby(["event_type", "event_id"]):
            if group[["event_type", "timestamp", "user", "scenario"]].drop_duplicates().shape[0] > 1:
                raise ValueError("Contradicting event labels")
        frame = frame.drop_duplicates(["event_type", "event_id"])
    frame["date"] = pd.to_datetime(frame.timestamp, format="%m/%d/%Y %H:%M:%S").dt.normalize()
    frame.to_parquet(destination / "answer_events.parquet", index=False)
    days = frame.groupby(["user", "date"]).agg(
        labelled_events=("event_id", "size"), scenario=("scenario", "max"),
        event_types=("event_type", lambda x: ";".join(sorted(set(x))))).reset_index()
    days["label"] = 1
    days.to_parquet(destination / "answer_days.parquet", index=False)
    summary = {"definition": "incident-observable day of insider or verified stolen-account activity; benign supervisor replies excluded",
               "note": "Official observables include preparatory/precursor activity; they do not independently prove malicious intent of every event.",
               "counts": dict(counts), "events": len(frame), "days": len(days),
               "users": frame.user.nunique(), "min": str(frame.date.min()), "max": str(frame.date.max()),
               "by_scenario": days.groupby("scenario").size().to_dict(),
               "by_type": frame.groupby("event_type").size().to_dict()}
    (destination / "answer_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    build_labels()
