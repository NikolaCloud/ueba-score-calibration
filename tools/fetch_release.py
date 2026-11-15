"""Resumable, concurrent official archive download with per-range and full hashes."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RANGE_BYTES = 64 * 1024 * 1024


def fetch(release="r5.2", workers=12):
    folder = ROOT / "data/metadata" / release
    metadata = json.loads((folder / "cert_metadata.json").read_text())
    item = next(x for x in metadata["files"] if x["name"] == release+".tar.bz2")
    target = ROOT / "data/raw" / item["name"]
    marks = ROOT / "data/raw" / (release+"_ranges")
    marks.mkdir(exist_ok=True)
    if not target.exists():
        with target.open("wb") as handle:
            handle.truncate(item["size"])
    if target.stat().st_size != item["size"]:
        raise ValueError("Existing archive has an unexpected length")
    bounds = [(s, min(s+RANGE_BYTES, item["size"])-1) for s in range(0, item["size"], RANGE_BYTES)]
    total_started = time.monotonic()

    def one(pair):
        start, end = pair
        mark = marks / f"{start}-{end}.json"
        if mark.exists():
            saved = json.loads(mark.read_text())
            with target.open("rb") as handle:
                handle.seek(start)
                digest = hashlib.sha256(handle.read(end-start+1)).hexdigest()
            if digest == saved["sha256"]:
                return
        for attempt in range(5):
            try:
                url = item["download_url"]+f"?range={start}&ts={time.time_ns()}"
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  timeout=(45, 120), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{item['size']}":
                        raise ValueError("Server did not return the requested byte range")
                    digest = hashlib.sha256()
                    length = 0
                    with target.open("r+b") as handle:
                        handle.seek(start)
                        for block in response.iter_content(2*1024*1024):
                            handle.write(block)
                            length += len(block)
                            digest.update(block)
                        handle.flush()
                    if length != end-start+1:
                        raise ValueError("Incomplete byte range")
                    temporary = mark.with_suffix(".tmp")
                    temporary.write_text(json.dumps(dict(start=start, end=end, bytes=length,
                                                          sha256=digest.hexdigest())), encoding="utf-8")
                    temporary.replace(mark)
                return
            except Exception as exc:
                print(f"Range {start}, attempt {attempt+1}: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one, pair) for pair in bounds}
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=25,
                                                  return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
            completed = sum(min(RANGE_BYTES, item["size"]-start) for start, end in bounds
                            if (marks / f"{start}-{end}.json").exists())
            elapsed = time.monotonic()-total_started
            print(f"{release}: {completed/1e9:.3f}/{item['size']/1e9:.3f} GB, "
                  f"{completed/1e6/max(elapsed,1):.2f} MB/s, ranges {len(bounds)-len(pending)}/{len(bounds)}", flush=True)
    hashes = {name: hashlib.new(name) for name in ["md5", "sha256"]}
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            for digest in hashes.values():
                digest.update(block)
    result = dict(file=target.relative_to(ROOT).as_posix(), bytes=target.stat().st_size,
                  url=item["download_url"], **{k:v.hexdigest() for k,v in hashes.items()})
    expected_sha = dict(line.split()[::-1] for line in (ROOT / "data/raw/CHECKSUMS-sha256.txt").read_text().splitlines())
    if result["md5"] != item["computed_md5"] or result["sha256"] != expected_sha[item["name"]]:
        raise ValueError("Publisher checksum mismatch")
    target.with_suffix(target.suffix+".provenance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="r5.2")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    fetch(args.release, args.workers)
