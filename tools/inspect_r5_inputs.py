"""Inspect official r5.2 incident files before choosing schema adapters."""
import csv
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "data/metadata/r5.2"
folder.mkdir(parents=True, exist_ok=True)
response = requests.get("https://api.figshare.com/v2/articles/12841247", timeout=60)
response.raise_for_status()
metadata = response.json()
(folder / "cert_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(next(x for x in metadata["files"] if x["name"] == "r5.2.tar.bz2"), indent=2))
with tarfile.open(ROOT / "data/raw/answers.tar.bz2", "r:bz2") as archive:
    master = pd.read_csv(io.BytesIO(archive.extractfile("answers/insiders.csv").read()), dtype=str)
    selected = master.loc[master.dataset.eq("5.2")].copy()
    selected.to_csv(folder / "official_incidents.csv", index=False)
    print("Official incidents by scenario:", selected.groupby("scenario").size().to_dict())
    summaries = []
    for member in archive:
        if not member.isfile() or "r5.2" not in member.name:
            continue
        rows = [r for r in csv.reader(archive.extractfile(member).read().decode("utf-8").splitlines()) if r]
        kinds = Counter(r[0] for r in rows)
        actors = Counter(r[3] for r in rows)
        summary = dict(file=member.name, rows=len(rows), kinds=dict(kinds), actors=dict(actors),
                       first_rows=[r[:min(12, len(r)-1)] for r in rows[:2]],
                       last_rows=[r[:min(12, len(r)-1)] for r in rows[-3:]])
        summaries.append(summary)
    (folder / "answer_key_structure.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("Incident files:", len(summaries))
    print("Actors other than the incident owner:",
          [dict(file=s["file"], actors=s["actors"]) for s in summaries
           if len(s["actors"]) > 1])
