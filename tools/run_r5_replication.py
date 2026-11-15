"""Run logged replication stages, optionally waiting for the official extraction."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-data",action="store_true")
    parser.add_argument("--from-stage",choices=["prepare","experiments","statistics","report","verify"],default="prepare")
    args = parser.parse_args()
    marker = ROOT/"data/metadata/r5.2/extracted_manifest.json"
    if args.wait_for_data:
        while not marker.exists():
            time.sleep(5)
    logs = ROOT/"logs/r5_2"
    logs.mkdir(parents=True,exist_ok=True)
    commands = [("prepare",["-m","ueba.replication.prepare"]),
                ("experiments",["-m","ueba.replication.run","--stage","experiments"]),
                ("statistics",["-m","ueba.replication.run","--stage","statistics"]),
                ("report",["-m","ueba.replication.report"]),
                ("verify",["tools/verify_r5_artifacts.py"])]
    start = [name for name,_ in commands].index(args.from_stage)
    for name,arguments in commands[start:]:
        print(f"r5.2 stage: {name}",flush=True)
        log_path = logs/(name+".log")
        with log_path.open("a",encoding="utf-8") as log:
            log.write(f"\nSTART {time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())} UTC\n")
            process = subprocess.Popen([sys.executable,"-u",*arguments],cwd=ROOT,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",env={**os.environ,"PYTHONUTF8":"1"})
            for line in process.stdout:
                print(line,end="",flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise SystemExit(f"Stage {name} failed ({code}); see {log_path.relative_to(ROOT)}")
    print("r5.2 replication completed",flush=True)


if __name__ == "__main__":
    main()
