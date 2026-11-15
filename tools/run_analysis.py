"""Execute analysis stages, persisting logs and stopping immediately on failure."""
import subprocess
import sys
import time
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    source_files = list((ROOT / "ueba").rglob("*.py")) + list((ROOT / "tools").glob("*.py"))
    source_files.extend([ROOT / "docs/protocol.md", ROOT / "requirements-lock.txt"])
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (logs / "source_snapshot.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
    commands = [("prepare", ["-m", "ueba.data"]), ("tests", ["-m", "pytest", "-q"]),
                ("experiment", ["-m", "ueba.experiment"]),
                ("statistics", ["-m", "ueba.statistics", "--replicates", "1000"]),
                ("report", ["-m", "ueba.report"]),
                ("verification", ["tools/verify_artifacts.py"])]
    for name, arguments in commands:
        print(f"Starting {name}", flush=True)
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace",
                                       env={**__import__("os").environ, "PYTHONUTF8": "1"})
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise SystemExit(f"Stage {name} failed with exit code {code}; see logs/{name}.log")
    print("All analysis stages completed", flush=True)


if __name__ == "__main__":
    main()
