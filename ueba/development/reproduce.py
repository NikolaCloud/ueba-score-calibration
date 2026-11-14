"""Recalculate in a new directory without changing the sealed development artifacts."""
import argparse
from pathlib import Path

from . import controls,round2,run,statistics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/m4_v2_reproduction')
    args=parser.parse_args()
    target=Path(args.output).resolve()
    if target.exists():
        raise ValueError('Choose a new output directory; existing research artifacts are preserved')
    target.mkdir(parents=True)
    for module in [run,round2,statistics,controls]:
        module.OUT=target
    run.main()
    round2.main()
    statistics.main()
    controls.main()


if __name__=='__main__':
    main()
