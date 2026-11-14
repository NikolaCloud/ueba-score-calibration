"""Post-selection diagnostic: give original M2/M3/M4 the same temporal averaging."""
import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

from ..metrics import threshold_at_rate
from .run import OUT,RELEASES,metrics
from .statistics import bootstrap_audit
from .tails import causal_mean


def main():
    selected=json.loads((OUT/'selection_round2.json').read_text())['selected']
    rows=[]
    for release,directory in RELEASES.items():
        for seed in range(42,47):
            frame=pd.read_parquet(directory/f'seed_{seed}/validation_predictions.parquet')
            midpoint=frame.date.min()+pd.Timedelta(days=((frame.date.max()+pd.Timedelta(days=1)-frame.date.min()).days//2))
            selection=frame.date.lt(midpoint).to_numpy();audit=~selection
            source=OUT/release/f'seed_{seed}/audit_predictions_round2.parquet'
            frozen=pd.read_parquet(source)
            part=frame.loc[audit].reset_index(drop=True)
            if not part[['user','date','role','label']].equals(frozen[['user','date','role','label']]):
                raise ValueError('Nonpaired control rows')
            output=frozen[[c for c in frozen if not c.startswith(('score_','alert_'))]+[f'score_{selected}',f'alert_{selected}']].copy()
            for original in ['M2','M3','M4']:
                name='T'+original
                smoothed=causal_mean(frame,frame[f'score_{original}'])
                threshold=threshold_at_rate(smoothed[selection]);scores=smoothed[audit]
                rows.append(dict(metrics(part,scores,threshold),release=release,seed=seed,method=name,threshold=threshold))
                output[f'score_{name}']=scores;output[f'alert_{name}']=scores>threshold
            output.to_parquet(source.with_name('audit_predictions_temporal_controls.parquet'),index=False)
    pd.DataFrame(rows).to_csv(OUT/'temporal_control_metrics.csv',index=False)
    cis=[];pairs=[]
    with threadpool_limits(limits=1):
        for release in RELEASES:
            c,p=bootstrap_audit(release,selected,'audit_predictions_temporal_controls.parquet',['TM2','TM3','TM4'],'_temporal_controls')
            cis.extend(c);pairs.extend(p)
    pd.DataFrame(cis).to_csv(OUT/'temporal_control_confidence_intervals.csv',index=False)
    frame=pd.DataFrame(pairs);valid=frame.p_centered_bootstrap.notna()
    frame.loc[valid,'p_holm']=multipletests(frame.loc[valid,'p_centered_bootstrap'],method='holm')[1]
    frame.to_csv(OUT/'temporal_control_comparisons.csv',index=False)


if __name__=='__main__':
    main()
