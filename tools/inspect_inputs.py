"""Inspect official answer keys and derivative matrices before modelling."""
import json
import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / "data/raw/answers.tar.bz2"
with tarfile.open(archive, "r:bz2") as handle:
    members = handle.getmembers()
    print("Answer archive files:", len(members))
    print("\n".join(x.name for x in members[:25]))
    selected = [x for x in members if x.isfile() and
                ("r4.2" in x.name or "readme" in x.name.lower() or x.name.endswith("insiders.csv")
                 or x.name.endswith("scenarios.txt"))]
    for item in selected:
        relative = PurePosixPath(item.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(item.name)
        target = ROOT / "data/answers" / Path(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = handle.extractfile(item).read()
        target.write_bytes(payload)
        if "readme" in item.name.lower() or item.name.endswith("insiders.csv") or item == selected[0]:
            print(item.name, item.size, payload[:4000].decode("utf-8", errors="replace"))
    print("r4.2 answer examples:")
    for item in [x for x in selected if "r4.2" in x.name][:4]:
        print(item.name, handle.extractfile(item).read(1200).decode("utf-8", errors="replace"))
for path in (ROOT / "data/external").glob("*.parquet"):
    frame = pd.read_parquet(path)
    print(path.name, frame.shape)
    print(frame.dtypes.to_string())
    print(frame.head(3).to_string())
    print("missing", frame.isna().sum().to_dict())
