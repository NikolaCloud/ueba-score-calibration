"""Inspect the execution environment without changing it."""
import importlib.util
import json
import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
print(json.dumps({"python": sys.executable, "platform": platform.platform(),
                  "cpu": os.cpu_count(), "disk": shutil.disk_usage(root)._asdict(),
                  "packages": {x: bool(importlib.util.find_spec(x)) for x in
                               ["numpy", "pandas", "polars", "sklearn", "scipy", "statsmodels",
                                "matplotlib", "requests", "pytest", "pyarrow"]}}, indent=2))
for name, url in {
    "cert_metadata": "https://api.figshare.com/v2/articles/12841247",
    "zenodo_metadata": "https://zenodo.org/api/records/19435851",
}.items():
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "UEBA-research/0.1"})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.load(response)
        directory = root / "data" / "metadata"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (name + ".json")).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(name, json.dumps(data.get("files", []), indent=2))
    except Exception as exc:
        print(name, type(exc).__name__, str(exc))
