"""Causal user-day aggregation of the four selected CERT logs."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


def external_recipients(value):
    addresses = {address.strip().lower() for address in str(value).split(";") if "@" in address}
    return sum(address.rsplit("@", 1)[1] != "dtaa.com" for address in addresses)


def attach_roles(days, ldap_directory):
    """Use only snapshots dated no later than the target day; no future backfill."""
    snapshots = []
    for path in sorted(Path(ldap_directory).glob("*.csv")):
        match = re.search(r"(20\d{2})[-_](\d{2})", path.stem)
        if not match:
            raise ValueError(f"Unknown LDAP snapshot date: {path}")
        frame = pd.read_csv(path, dtype=str).fillna("")
        frame.columns = frame.columns.str.strip().str.lower()
        if "user_id" not in frame or "role" not in frame:
            raise ValueError(f"LDAP requires user_id and role: {frame.columns.tolist()}")
        if frame.user_id.duplicated().any():
            raise ValueError("Duplicate LDAP user")
        frame = frame.rename(columns={"user_id": "user"})
        frame["snapshot_date"] = pd.Timestamp(f"{match[1]}-{match[2]}-01")
        snapshots.append(frame[["user", "role", "snapshot_date"]])
    if not snapshots:
        raise ValueError("LDAP is required for the primary role experiment")
    roles = pd.concat(snapshots).sort_values("snapshot_date")
    roles["snapshot_date"] = roles.snapshot_date.astype("datetime64[ns]")
    days = days.copy()
    days["date"] = days.date.astype("datetime64[ns]")
    result = pd.merge_asof(days.sort_values("date"), roles, left_on="date",
                           right_on="snapshot_date", by="user", direction="backward")
    if (result.snapshot_date.dropna() > result.loc[result.snapshot_date.notna(), "date"]).any():
        raise AssertionError("Future LDAP information")
    result["role"] = result.role.fillna("UNKNOWN").replace("", "UNKNOWN")
    return result.sort_values(["date", "user"]).reset_index(drop=True)


def prepare():
    raw = ROOT / "data/raw/r4.2"
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/metadata/extracted_manifest.json").read_text())
    for entry in manifest:
        path = ROOT / "data/raw" / entry["archive_member"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Extracted source changed: {path}")
    answers = pd.read_parquet(processed / "answer_events.parquet")
    found = set()
    parts, station_parts, audit = [], [], {}
    for kind in ["logon", "device", "file", "email"]:
        # CERT's errata explicitly say IDs are unique within each log, not globally.
        kind_answers = answers.loc[answers.event_type.eq(kind)]
        keys = set(kind_answers.event_id)
        expected = kind_answers.set_index("event_id")
        source = raw / f"{kind}.csv"
        header = pd.read_csv(source, nrows=0).columns.tolist()
        wanted = ["id", "date", "user", "pc"]
        if kind in {"logon", "device"}:
            wanted.append("activity")
        if kind == "email":
            wanted.extend(["to", "cc", "bcc"])
        if not set(wanted) <= set(header):
            raise ValueError(f"Unexpected columns for {kind}: {header}")
        seen = {}
        rows = duplicate = missing = 0
        activity_counts = {}
        print(f"Aggregating {kind}", flush=True)
        for chunk in pd.read_csv(source, usecols=wanted, dtype=str, chunksize=150_000,
                                 keep_default_na=False):
            rows += len(chunk)
            missing += int(chunk[["id", "date", "user", "pc"]].eq("").any(axis=1).sum())
            if missing:
                raise ValueError(f"Missing essential event fields: {kind}")
            # Hash fields used in the experiment to distinguish exact repeated rows
            # from event-ID collisions. Text bodies are deliberately never model features.
            hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
            keep = np.ones(len(chunk), dtype=bool)
            for i, (event_id, fingerprint) in enumerate(zip(chunk.id, hashes)):
                prior = seen.get(event_id)
                if prior is not None:
                    if prior != int(fingerprint):
                        raise ValueError(f"Conflicting event ID: {event_id}")
                    duplicate += 1
                    keep[i] = False
                else:
                    seen[event_id] = int(fingerprint)
            chunk = chunk.loc[keep].copy()
            stamp = pd.to_datetime(chunk.date, format=DATE_FORMAT, errors="raise")
            chunk["date"] = stamp.dt.normalize()
            chunk["total_events"] = 1
            chunk["afterhours_events"] = ((stamp.dt.hour < 8) | (stamp.dt.hour >= 18)).astype(int)
            chunk["weekend_events"] = (stamp.dt.dayofweek >= 5).astype(int)
            columns = ["total_events", "afterhours_events", "weekend_events"]
            chunk[f"{kind}_events"] = 1
            chunk[f"{kind}_afterhours"] = chunk.afterhours_events
            columns.extend([f"{kind}_events", f"{kind}_afterhours"])
            if kind in {"logon", "device"}:
                activity_counts = pd.concat([pd.Series(activity_counts), chunk.activity.value_counts()],
                                             axis=1).fillna(0).sum(axis=1).astype(int).to_dict()
                values = {"logon": {"logons": "Logon", "logoffs": "Logoff"},
                          "device": {"usb_connects": "Connect", "usb_disconnects": "Disconnect"}}[kind]
                if not set(chunk.activity.unique()) <= set(values.values()):
                    raise ValueError(f"Unknown {kind} activity")
                for name, value in values.items():
                    chunk[name] = chunk.activity.eq(value).astype(int)
                    columns.append(name)
            if kind == "email":
                recipients = chunk["to"] + ";" + chunk["cc"] + ";" + chunk["bcc"]
                chunk["external_recipients"] = recipients.map(external_recipients)
                columns.append("external_recipients")
            matching = chunk[chunk.id.isin(keys)]
            for row in matching.itertuples():
                labelled = expected.loc[row.id]
                if labelled.user != row.user or labelled.event_type != kind or labelled.date != row.date:
                    raise ValueError(f"Answer metadata mismatch: {row.id}")
            found.update((kind, value) for value in matching.id)
            parts.append(chunk.groupby(["user", "date"])[columns].sum())
            station_parts.append(chunk[["user", "date", "pc"]].drop_duplicates())
        audit[kind] = dict(rows=rows, duplicate_rows_removed=duplicate,
                           missing_essential_fields=missing, activities=activity_counts,
                           unique_events=len(seen))
        print(f"{kind}: {audit[kind]}", flush=True)
    needed = set(answers.loc[answers.event_type.ne("http"), ["event_type", "event_id"]].itertuples(index=False, name=None))
    if needed - found:
        raise ValueError(f"{len(needed-found)} selected answer events missing from source logs")
    features = pd.concat(parts).fillna(0).groupby(level=[0, 1]).sum().reset_index()
    stations = pd.concat(station_parts).drop_duplicates().groupby(["user", "date"]).size()
    features = features.merge(stations.rename("unique_pcs"), on=["user", "date"], validate="one_to_one")
    numeric = features.columns.difference(["user", "date"])
    features[numeric] = features[numeric].astype("int64")
    labels = pd.read_parquet(processed / "answer_days.parquet")
    features = features.merge(labels, on=["user", "date"], how="left", validate="one_to_one")
    features["label"] = features.label.fillna(0).astype(int)
    features["scenario"] = features.scenario.fillna(0).astype(int)
    features["labelled_events"] = features.labelled_events.fillna(0).astype(int)
    features["event_types"] = features.event_types.fillna("")
    features = attach_roles(features, raw / "LDAP")
    if features[["user", "date"]].duplicated().any():
        raise ValueError("Duplicate user-day")
    if features[numeric].isna().any().any() or (features[numeric] < 0).any().any():
        raise ValueError("Invalid feature values")
    features.to_parquet(processed / "user_days.parquet", index=False)
    summary = dict(users=int(features.user.nunique()), active_days=len(features),
                   positive_days=int(features.label.sum()), negative_days=int((features.label == 0).sum()),
                   positive_users=int(features.loc[features.label.eq(1), "user"].nunique()),
                   start=str(features.date.min().date()), end=str(features.date.max().date()),
                   roles=int(features.role.nunique()), unknown_role_days=int(features.role.eq("UNKNOWN").sum()),
                   missing_label_days=len(labels)-int(features.label.sum()),
                   confirmed_non_http_events=len(found), features=list(numeric),
                   missing_feature_values=int(features[numeric].isna().sum().sum()),
                   http_used=False, source_audit=audit)
    (processed / "data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    features.groupby("role").agg(days=("user", "size"), users=("user", "nunique"),
                                  positive_days=("label", "sum")).to_csv(processed / "role_distribution.csv")
    features.groupby([features.date.dt.to_period("M").astype(str), "scenario"]).size().rename(
        "user_days").to_csv(processed / "scenario_timeline.csv")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
