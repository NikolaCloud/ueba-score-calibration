"""Recalculate in a fresh directory, including from the small review bundle."""
import argparse
import hashlib
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from . import run,statistics,history


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/m5_reproduction')
    parser.add_argument('--releases',nargs='+',choices=list(run.SOURCES),default=['r4.2','r5.2','r6.2'])
    parser.add_argument('--history',action='store_true')
    parser.add_argument('--r6-detector',action='store_true')
    args=parser.parse_args()
    target=Path(args.output).resolve()
    if target.exists():raise ValueError('Use a new output directory; existing plans/results are preserved')
    reference=json.loads((run.ROOT/'results/m5/frozen_plan.json').read_text())
    for relative,digest in reference['files'].items():
        if hashlib.sha256((run.ROOT/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Scoring/protocol source differs from the reviewed experiment: '+relative)
    target.mkdir(parents=True)
    for module in [run,statistics,history]:module.OUT=target
    # A fresh plan does not require the historical review ZIP: its recorded calibration
    # SHA256 above proves formula identity even when matched_original_bundle is false.
    run.freeze()
    with threadpool_limits(limits=1):
        if args.r6_detector:run.r6_detector()
        for release in args.releases:
            run.validation(release);run.test(release);statistics.bootstrap(release)
        statistics.correct_all()
    if args.history:
        import sys
        original=sys.argv[:]
        try:
            for release in args.releases:
                sys.argv=['history','--release',release];history.main()
        finally:sys.argv=original
    print('Reproduction completed: '+str(target),flush=True)


if __name__=='__main__':main()
