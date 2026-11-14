"""Audited r6.2 labels plus a parameterized copy of the verified r5.2 feature builder."""
import ast
import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def build_r6_answers():
    output=ROOT/'data/processed/r6.2';output.mkdir(parents=True,exist_ok=True)
    records=[]
    with tarfile.open(ROOT/'data/raw/answers.tar.bz2','r:bz2') as archive:
        master=pd.read_csv(io.BytesIO(archive.extractfile('answers/insiders.csv').read()),dtype=str)
        master=master.loc[master.dataset.eq('6.2')].copy()
        for item in ['scenarios.txt','readme.txt']:
            (ROOT/'data/metadata/r6.2'/('answers_'+item)).write_bytes(archive.extractfile('answers/'+item).read())
        for incident in master.itertuples(index=False):
            rows=[r for r in csv.reader(archive.extractfile('answers/'+incident.details).read().decode('utf-8').splitlines()) if r]
            transfer=[pd.to_datetime(r[2],format='%m/%d/%Y %H:%M:%S').normalize()
                for r in rows if r[0]=='file' and r[3]==incident.user and 'keylogger' in r[5].lower()]
            stolen=set()
            day=None
            if incident.scenario=='3':
                if not transfer:raise ValueError('No verified keylogger transfer')
                day=min(transfer)+pd.Timedelta(days=1)
                logons={(r[3],r[4]) for r in rows if r[0]=='logon' and r[3]!=incident.user
                    and pd.to_datetime(r[2],format='%m/%d/%Y %H:%M:%S').normalize()==day}
                sends={(r[3],r[4]) for r in rows if r[0]=='email' and len(r)>9 and r[9]=='Send'
                    and pd.to_datetime(r[2],format='%m/%d/%Y %H:%M:%S').normalize()==day}
                stolen=logons&sends
                if not stolen:raise ValueError('No corroborated stolen-account Logon and Send')
            for row in rows:
                kind,event_id,stamp,actor,pc=row[:5]
                date=pd.to_datetime(stamp,format='%m/%d/%Y %H:%M:%S').normalize()
                owner=actor==incident.user
                impersonation=(incident.scenario=='3' and (actor,pc) in stolen and date==day
                    and (kind=='logon' or (kind=='email' and len(row)>9 and row[9]=='Send')))
                include=owner or impersonation
                records.append(dict(event_type=kind,event_id=event_id,timestamp=stamp,user=actor,pc=pc,date=date,
                    scenario=int(incident.scenario),incident_id=incident.details,incident_insider=incident.user,
                    included=include,reason='owner_observable' if owner else 'verified_impersonation' if impersonation else 'other_actor_context'))
    refs=pd.DataFrame(records)
    included=refs.loc[refs.included].copy()
    for _,part in included.groupby(['event_type','event_id']):
        if len(part[['user','timestamp','pc','scenario']].drop_duplicates())!=1:
            raise ValueError('Conflicting official observable references')
    events=included.drop_duplicates(['event_type','event_id']).copy()
    events['key_references']=events.incident_id
    refs.to_parquet(output/'answer_references.parquet',index=False)
    events.to_parquet(output/'answer_events.parquet',index=False)
    refs.loc[~refs.included].to_csv(output/'excluded_other_actor_observables.csv',index=False)
    days=events.groupby(['user','date']).agg(labelled_events=('event_id','size'),
        scenario_ids=('scenario',lambda v:';'.join(map(str,sorted(set(v))))),
        event_types=('event_type',lambda v:';'.join(sorted(set(v)))),
        incident_ids=('incident_id',lambda v:';'.join(sorted(set(v))))).reset_index()
    for sid in range(1,6):days[f'scenario_{sid}']=days.scenario_ids.map(lambda s:str(sid) in s.split(';')).astype(int)
    days['scenario']=days.scenario_ids.map(lambda s:int(s) if ';' not in s else 0)
    days['label']=1;days.to_parquet(output/'answer_days.parquet',index=False)
    master.to_csv(output/'official_incidents.csv',index=False)
    audit=dict(release='r6.2',official_incidents=len(master),all_reference_events=len(refs),
        included_events=len(events),excluded_other_actor_events=int((~refs.included).sum()),
        verified_impersonation_events=int(refs.reason.eq('verified_impersonation').sum()),
        positive_days=len(days),positive_accounts=int(days.user.nunique()),
        days_by_scenario={str(i):int(days[f'scenario_{i}'].sum()) for i in range(1,6)},
        master_scenario4_start_typo=True,definition='Owner incident-observable days plus corroborated stolen-account activity; context reactions excluded')
    (output/'answer_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)


class ReleaseAdapter(ast.NodeTransformer):
    """Change release literals, without editing frozen r5.2 source or feature logic."""
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            value=node.value.replace('r5.2','r6.2')
            return ast.copy_location(ast.Constant(value=value),node)
        return node


def prepare_r6():
    source=ROOT/'ueba/replication/prepare.py'
    original=source.read_text(encoding='utf-8')
    tree=ReleaseAdapter().visit(ast.parse(original));ast.fix_missing_locations(tree)
    transformed=ast.unparse(tree)
    folder=ROOT/'data/metadata/r6.2'
    (folder/'feature_builder_adapted.py').write_text(transformed,encoding='utf-8')
    (folder/'feature_builder_provenance.json').write_text(json.dumps(dict(
        original_source='ueba/replication/prepare.py',original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        adapter='Replace literal r5.2 with r6.2 only, preserving schema assertions and all feature/filter logic',
        adapted_sha256=hashlib.sha256(transformed.encode('utf-8')).hexdigest()),indent=2),encoding='utf-8')
    namespace={'__name__':'ueba.replication.adapted_r6_features','__package__':'ueba.replication','__file__':str(source)}
    exec(compile(tree,str(source),mode='exec'),namespace)
    namespace['prepare']()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['answers','features','all'],default='all')
    args=parser.parse_args()
    if args.stage in ['answers','all']:build_r6_answers()
    if args.stage in ['features','all']:prepare_r6()
