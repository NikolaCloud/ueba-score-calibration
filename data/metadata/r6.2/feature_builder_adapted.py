"""Restore r4.2-compatible user-day features from r6.2 raw logs."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from ..data import attach_roles, external_recipients
ROOT = Path(__file__).resolve().parents[2]
FORMAT = '%m/%d/%Y %H:%M:%S'

def selected_events(frame, kind):
    if kind == 'email':
        return frame.activity.eq('Send')
    if kind == 'file':
        destination = frame.to_removable_media.str.lower().eq('true')
        valid_actions = frame.activity.isin(['File Write', 'File Copy'])
        if (destination & ~valid_actions).any():
            raise ValueError('Unexpected action marked as writing to removable media')
        return destination & valid_actions
    return pd.Series(True, index=frame.index)

def prepare():
    raw = ROOT / 'data/raw/r6.2'
    output = ROOT / 'data/processed/r6.2'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'data/metadata/r6.2/extracted_manifest.json').read_text())
    for entry in manifest:
        path = ROOT / 'data/raw' / entry['archive_member']
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != entry['bytes'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('Source checksum changed after extraction')
    answers = pd.read_parquet(output / 'answer_events.parquet')
    pieces, stations, source_audit, coverage = ([], [], {}, [])
    found = set()
    for kind in ['logon', 'device', 'file', 'email']:
        source = raw / f'{kind}.csv'
        columns = pd.read_csv(source, nrows=0).columns.tolist()
        wanted = ['id', 'date', 'user', 'pc', 'activity']
        if kind == 'file':
            wanted += ['to_removable_media', 'from_removable_media']
        if kind == 'email':
            wanted += ['to', 'cc', 'bcc']
        if not set(wanted) <= set(columns):
            raise ValueError(f'Incompatible schema: {kind}: {columns}')
        expected = answers.loc[answers.event_type.eq(kind)].set_index('event_id')
        seen, counts, actions = ({}, Counter(), Counter())
        print(f'r6.2 aggregating {kind}', flush=True)
        for chunk in pd.read_csv(source, usecols=wanted, dtype=str, keep_default_na=False, chunksize=150000):
            counts['raw_rows'] += len(chunk)
            if chunk[['id', 'date', 'user', 'pc']].eq('').any().any():
                raise ValueError('Missing essential event metadata')
            actions.update(chunk.activity)
            matching = chunk.loc[chunk.id.isin(expected.index)]
            keep = selected_events(chunk, kind)
            for idx, row in matching.iterrows():
                key = expected.loc[row.id]
                stamp = pd.to_datetime(row.date, format=FORMAT)
                if key.user != row.user or key.timestamp != row.date or key.pc != row.pc:
                    raise ValueError(f'Official key mismatches raw metadata: {kind}/{row.id}')
                found.add((kind, row.id))
                coverage.append(dict(event_type=kind, event_id=row.id, user=row.user, date=stamp.normalize(), scenario=int(key.scenario), selected_feature_event=bool(keep.loc[idx]), activity=row.activity))
            chunk = chunk.loc[keep].copy()
            counts['selected_rows_before_dedup'] += len(chunk)
            fingerprints = pd.util.hash_pandas_object(chunk, index=False).to_numpy()
            unique = np.ones(len(chunk), bool)
            for i, (event_id, fingerprint) in enumerate(zip(chunk.id, fingerprints)):
                if event_id in seen:
                    if seen[event_id] != int(fingerprint):
                        raise ValueError('Conflicting selected event ID')
                    unique[i] = False
                    counts['selected_duplicates_removed'] += 1
                else:
                    seen[event_id] = int(fingerprint)
            chunk = chunk.loc[unique].copy()
            if chunk.empty:
                continue
            timestamp = pd.to_datetime(chunk.date, format=FORMAT)
            chunk['date'] = timestamp.dt.normalize()
            chunk['total_events'] = 1
            chunk['afterhours_events'] = ((timestamp.dt.hour < 8) | (timestamp.dt.hour >= 18)).astype(int)
            chunk['weekend_events'] = timestamp.dt.dayofweek.ge(5).astype(int)
            chunk[f'{kind}_events'] = 1
            chunk[f'{kind}_afterhours'] = chunk.afterhours_events
            numeric = ['total_events', 'afterhours_events', 'weekend_events', f'{kind}_events', f'{kind}_afterhours']
            if kind in {'logon', 'device'}:
                mapping = {'logon': {'logons': 'Logon', 'logoffs': 'Logoff'}, 'device': {'usb_connects': 'Connect', 'usb_disconnects': 'Disconnect'}}[kind]
                if not set(chunk.activity) <= set(mapping.values()):
                    raise ValueError('Unknown activity')
                for name, activity in mapping.items():
                    chunk[name] = chunk.activity.eq(activity).astype(int)
                    numeric.append(name)
            if kind == 'email':
                chunk['external_recipients'] = (chunk['to'] + ';' + chunk.cc + ';' + chunk.bcc).map(external_recipients)
                numeric.append('external_recipients')
            pieces.append(chunk.groupby(['user', 'date'])[numeric].sum())
            stations.append(chunk[['user', 'date', 'pc']].drop_duplicates())
        source_audit[kind] = dict(columns=columns, counts=dict(counts), raw_activity_counts=dict(actions), unique_selected_events=len(seen), deduplication_scope='Selected comparable events')
        print(kind, source_audit[kind], flush=True)
    wanted_keys = set(answers.loc[answers.event_type.ne('http'), ['event_type', 'event_id']].itertuples(index=False, name=None))
    if wanted_keys - found:
        raise ValueError(f'{len(wanted_keys - found)} official non-HTTP events not found')
    features = pd.concat(pieces).fillna(0).groupby(level=[0, 1]).sum().reset_index()
    pc_counts = pd.concat(stations).drop_duplicates().groupby(['user', 'date']).size().rename('unique_pcs')
    features = features.merge(pc_counts, on=['user', 'date'], validate='one_to_one')
    original_audit = json.loads((ROOT / 'data/processed/data_audit.json').read_text())
    numeric = original_audit['features']
    if set(features) - {'user', 'date'} != set(numeric):
        raise ValueError('Feature names differ from frozen r4.2 feature set')
    features[numeric] = features[numeric].astype('int64')
    labels = pd.read_parquet(output / 'answer_days.parquet')
    features = features.merge(labels, on=['user', 'date'], how='left', validate='one_to_one')
    for column in ['label', 'scenario', 'labelled_events', *[f'scenario_{i}' for i in range(1, 6)]]:
        features[column] = features[column].fillna(0).astype(int)
    for column in ['scenario_ids', 'incident_ids', 'event_types']:
        features[column] = features[column].fillna('')
    features = attach_roles(features, raw / 'LDAP')
    if features[['user', 'date']].duplicated().any() or features[numeric].isna().any().any():
        raise ValueError('Invalid aggregated user-days')
    missing = labels.merge(features[['user', 'date']], on=['user', 'date'], how='left', indicator=True)
    missing = missing.loc[missing._merge.eq('left_only')]
    missing.to_csv(output / 'unobserved_label_days.csv', index=False)
    features.to_parquet(output / 'user_days.parquet', index=False)
    covered = pd.DataFrame(coverage)
    covered.to_parquet(output / 'key_event_coverage.parquet', index=False)
    covered.groupby(['scenario', 'event_type', 'selected_feature_event']).size().rename('events').to_csv(output / 'key_event_coverage.csv')
    day_coverage = covered.groupby(['user', 'date']).selected_feature_event.sum().rename('selected_key_events').reset_index()
    label_coverage = labels.merge(day_coverage, on=['user', 'date'], how='left')
    label_coverage['selected_key_events'] = label_coverage.selected_key_events.fillna(0).astype(int)
    label_coverage.to_csv(output / 'label_day_feature_coverage.csv', index=False)
    mapping = []
    for feature in numeric:
        note = 'r4.2 removable-media copy; r6.2 File Write/Copy with to_removable_media=True' if feature.startswith('file_') else 'Outgoing Send events only; Receive omitted' if feature.startswith('email_') or feature == 'external_recipients' else 'Union/sum over selected comparable logon/device/file-write/email-send events' if feature in {'total_events', 'afterhours_events', 'weekend_events', 'unique_pcs'} else 'Same event and activity definitions as r4.2'
        mapping.append(dict(feature=feature, definition_mapping=note, model_input=True))
    pd.DataFrame(mapping).to_csv(output / 'feature_comparability.csv', index=False)
    features.groupby('role').agg(days=('user', 'size'), users=('user', 'nunique'), positive_days=('label', 'sum')).to_csv(output / 'role_distribution.csv')
    audit = dict(release='r6.2', users=int(features.user.nunique()), active_days=len(features), positive_days=int(features.label.sum()), negative_days=int(features.label.eq(0).sum()), positive_accounts=int(features.loc[features.label.eq(1), 'user'].nunique()), roles=int(features.role.nunique()), unknown_role_days=int(features.role.eq('UNKNOWN').sum()), missing_label_days=len(missing), start=str(features.date.min().date()), end=str(features.date.max().date()), features=numeric, http_used=False, confirmed_non_http_events=len(found), source_audit=source_audit, feature_semantics=dict(file='File Write/Copy AND to_removable_media=True; r4.2 contains removable-media copies only, so write-vs-copy granularity is a mapping limitation', email='Send only; Receive excluded', total_events='Only comparable selected events'))
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'source_audit'}, indent=2), flush=True)
if __name__ == '__main__':
    prepare()