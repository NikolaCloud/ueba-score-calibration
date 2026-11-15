"""Fetch the large official archive using verified HTTP byte ranges."""
import concurrent.futures
import json
import shutil
import time
from pathlib import Path

import requests

from download_data import ROOT, download


def main():
    meta = json.loads((ROOT / "data/metadata/cert_metadata.json").read_text())
    item = next(f for f in meta["files"] if f["name"] == "r4.2.tar.bz2")
    target = ROOT / "data/raw" / item["name"]
    if target.exists():
        download(item["download_url"], target, item["computed_md5"], item["size"])
        return
    prefix = target.with_suffix(target.suffix + ".partial")
    prefix_size = prefix.stat().st_size if prefix.exists() else 0
    folder = ROOT / "data/raw/cert_chunks"
    folder.mkdir(exist_ok=True)
    chunks = [(start, min(start + 128 * 1024 * 1024, item["size"]) - 1)
              for start in range(prefix_size, item["size"], 128 * 1024 * 1024)]

    def fetch(bounds):
        start, end = bounds
        path = folder / f"{start}-{end}.part"
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(5):
            try:
                url = item["download_url"] + f"?range_start={start}&ts={int(time.time())}"
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  timeout=(30, 120), stream=True) as response:
                    response.raise_for_status()
                    expected = f"bytes {start}-{end}/{item['size']}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Invalid range response: {response.status_code}, {response.headers.get('Content-Range')}")
                    with path.open("wb") as out:
                        for block in response.iter_content(4 * 1024 * 1024):
                            out.write(block)
                if path.stat().st_size != end - start + 1:
                    raise ValueError("Incomplete range")
                print(f"Completed range {start}-{end}", flush=True)
                return path
            except Exception as exc:
                print(f"Range {start}, attempt {attempt+1}: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    print(f"Preserved prefix: {prefix_size/1e6:.1f} MB; {len(chunks)} ranges, 6 workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(fetch, chunks))
    assembled = target.with_suffix(target.suffix + ".assembled")
    with assembled.open("wb") as out:
        if prefix_size:
            with prefix.open("rb") as handle:
                shutil.copyfileobj(handle, out, 8 * 1024 * 1024)
        for path in paths:
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, out, 8 * 1024 * 1024)
    if assembled.stat().st_size != item["size"]:
        raise ValueError("Assembled size mismatch")
    assembled.replace(target)
    download(item["download_url"], target, item["computed_md5"], item["size"])
    # Keep range files until the complete archive has been checksum-validated.


if __name__ == "__main__":
    main()
