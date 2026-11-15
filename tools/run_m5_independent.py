"""Continue the frozen independent experiment once verified raw extraction finishes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=ROOT/'data/metadata/r6.2/extracted_manifest.json'
    previous=time.monotonic()-60
    while not manifest.exists():
        if time.monotonic()-previous>=60:
            print('Waiting for publisher-verified r6.2 extraction; M5 configuration is frozen',flush=True)
            previous=time.monotonic()
        time.sleep(3)
    env=os.environ.copy();env['PYTHONUTF8']='1'
    commands=[['-m','ueba.m5.data','--stage','features'],
        ['-m','ueba.m5.run','--release','r6.2','--stage','detector'],
        ['-m','ueba.m5.run','--release','r6.2','--stage','all'],
        ['-m','ueba.m5.statistics','--release','r6.2']]
    for command in commands:
        print('Running: python '+' '.join(command),flush=True)
        subprocess.run([sys.executable,'-u',*command],cwd=ROOT,env=env,check=True)
    print('Frozen independent r6.2 experiment completed',flush=True)


if __name__=='__main__':main()
