"""Descriptive mechanism diagnostics; never used to alter M5 or its threshold."""
import json

import pandas as pd

from .run import OUT


def main():
    rows=[];disagreements=[]
    for folder in sorted(OUT.glob('r*')):
        if not (folder/'metrics.csv').exists():continue
        comparator=json.loads((folder/'comparator.json').read_text())['comparator']
        for seed in range(42,47):
            frame=pd.read_parquet(folder/f'seed_{seed}/test_predictions.parquet')
            for label,part in frame.groupby('label'):
                rows.append(dict(release=folder.name,seed=seed,label=int(label),days=len(part),
                    mean_user_weight=float(part.m5_weight_user.mean()),mean_instability=float(part.m5_instability.mean()),
                    fraction_memory_path=float(part.m5_cusum.gt(part.m5_instant_evidence).mean()),
                    fraction_alerts=float(part.alert_M5.mean())))
            positive=frame.label.eq(1)
            for method in dict.fromkeys([comparator,'EU','AB']):
                baseline=frame[f'alert_{method}']
                disagreements.append(dict(release=folder.name,seed=seed,comparator=method,positive_days=int(positive.sum()),
                    baseline_only=int((positive&baseline&~frame.alert_M5).sum()),
                    M5_only=int((positive&~baseline&frame.alert_M5).sum()),
                    both_detected=int((positive&baseline&frame.alert_M5).sum()),
                    both_missed=int((positive&~baseline&~frame.alert_M5).sum())))
    pd.DataFrame(rows).to_csv(OUT/'mechanism_diagnostics.csv',index=False)
    pd.DataFrame(disagreements).to_csv(OUT/'positive_day_disagreements.csv',index=False)


if __name__=='__main__':main()
