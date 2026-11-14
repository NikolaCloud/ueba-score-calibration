"""Declared post-selection capacity sensitivity, never used to reselect M5."""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..metrics import evaluate,stable_ties
from .run import OUT,ROOT,SOURCES,freeze
from .threshold import fit_guard,apply_guard,select_threshold

CAPACITIES=[10,30,60,90,120]


def load_model(capacity):
    class Capacity(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value in [119,120]:
                return ast.copy_location(ast.Constant(value=capacity-(node.value==119)),node)
            return node
    source=ROOT/'ueba/m5/model.py'
    tree=Capacity().visit(ast.parse(source.read_text(encoding='utf-8')));ast.fix_missing_locations(tree)
    text=ast.unparse(tree)
    generated=OUT/'history_sources';generated.mkdir(exist_ok=True)
    path=generated/f'capacity_{capacity}.py'
    if not path.exists() or path.read_text(encoding='utf-8')!=text:
        path.write_text(text,encoding='utf-8')
    name=f'ueba.m5._capacity_{capacity}'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.AdaptiveContext,hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',choices=list(SOURCES),required=True)
    args=parser.parse_args();freeze()
    release=args.release;rows=[];proofs={}
    with threadpool_limits(limits=1):
        for capacity in CAPACITIES:
            Model,digest=load_model(capacity);proofs[str(capacity)]=digest
            for seed in range(42,47):
                raw=pd.read_parquet(SOURCES[release]/f'seed_{seed}/raw_scores.parquet')
                cal=raw.loc[raw.block.eq('calibration')].reset_index(drop=True)
                val=raw.loc[raw.block.eq('validation')].reset_index(drop=True)
                test=raw.loc[raw.block.eq('test')].reset_index(drop=True)
                midpoint=val.date.min()+pd.Timedelta(days=((val.date.max()+pd.Timedelta(days=1)-val.date.min()).days//2))
                guard_mask=val.date.lt(midpoint).to_numpy();th_mask=~guard_mask
                model=Model().fit(cal)
                scores,_=model.transform(val);shifts=fit_guard(val.loc[guard_mask].reset_index(drop=True),scores['M5'][guard_mask])
                validation=apply_guard(val,scores['M5'],shifts)
                threshold,_=select_threshold(val.loc[th_mask].reset_index(drop=True),validation[th_mask])
                values,trace=model.transform(test);values=apply_guard(test,values['M5'],shifts)
                if capacity==120:
                    original=pd.read_parquet(OUT/release/f'seed_{seed}/test_predictions.parquet')
                    np.testing.assert_allclose(values,original.score_M5,rtol=0,atol=1e-12)
                    baseline=pd.read_csv(OUT/release/'thresholds.csv',float_precision='round_trip')
                    expected=baseline.loc[baseline.seed.eq(seed)&baseline.method.eq('M5'),'threshold'].iloc[0]
                    np.testing.assert_allclose(threshold,expected,rtol=0,atol=1e-12)
                metrics,_=evaluate(test,values,threshold,stable_ties(test))
                rows.append(dict(metrics,release=release,seed=seed,capacity=capacity,threshold=threshold,
                    mean_n_eff=float(trace.n_eff.mean()),fpr_exceeds_1pct=bool(metrics['fpr']>.01)))
                print(f'M5 history sensitivity {release}/capacity={capacity}/seed={seed}',flush=True)
    pd.DataFrame(rows).to_csv(OUT/release/'history_capacity_metrics.csv',index=False)
    (OUT/release/'history_capacity_provenance.json').write_text(json.dumps(dict(
        capacity_grid=CAPACITIES,post_selection_diagnostic=True,not_used_to_select_M5=True,
        base_model_sha256=hashlib.sha256((ROOT/'ueba/m5/model.py').read_bytes()).hexdigest(),
        generated_sources_sha256=proofs,capacity120_matches_primary=True),indent=2),encoding='utf-8')


if __name__=='__main__':main()
