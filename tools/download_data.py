"""Download official data with checksums, bounded retries, and resumable files."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def download(url, destination, expected_md5=None, expected_size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".partial")
        for attempt in range(5):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                # Avoid cached, expired signed redirects from the CDN.
                source = url + ("&" if "?" in url else "?") + f"research_download={int(time.time())}"
                with requests.get(source, stream=True, timeout=(30, 120),
                                  headers={"Range": f"bytes={offset}-"} if offset else {}) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        offset = 0
                    mode = "ab" if offset else "wb"
                    started = last = time.monotonic()
                    current = offset
                    with partial.open(mode) as handle:
                        for block in response.iter_content(4 * 1024 * 1024):
                            if block:
                                handle.write(block)
                                current += len(block)
                            if time.monotonic() - last > 25:
                                print(f"{destination.name}: {current / 1e6:.1f} MB; "
                                      f"{(current-offset)/1e6/(time.monotonic()-started):.1f} MB/s", flush=True)
                                last = time.monotonic()
                if expected_size and partial.stat().st_size != expected_size:
                    raise ValueError(f"Wrong length: {partial.stat().st_size} != {expected_size}")
                partial.replace(destination)
                break
            except Exception as exc:
                print(f"Download attempt {attempt + 1}: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    hashes = {name: hashlib.new(name) for name in ("md5", "sha256")}
    with destination.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            for digest in hashes.values():
                digest.update(block)
    digests = {name: value.hexdigest() for name, value in hashes.items()}
    if expected_md5 and digests["md5"] != expected_md5:
        raise ValueError(f"MD5 mismatch: {destination}")
    result = dict(file=str(destination.relative_to(ROOT)), url=url,
                  bytes=destination.stat().st_size, **digests)
    destination.with_suffix(destination.suffix + ".provenance.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    cert = json.loads((ROOT / "data/metadata/cert_metadata.json").read_text())
    wanted = {"SEI_Insider_README.txt", "answers.tar.bz2", "CHECKSUMS-sha256.txt"}
    if args.full:
        wanted.add("r4.2.tar.bz2")
    files = sorted((f for f in cert["files"] if f["name"] in wanted), key=lambda f: f["size"])
    for item in files:
        download(item["download_url"], ROOT / "data/raw" / item["name"],
                 item["computed_md5"], item["size"])
    meta = json.loads((ROOT / "data/metadata/zenodo_metadata.json").read_text())
    for item in meta["files"]:
        download(item["links"]["self"], ROOT / "data/external" / item["key"],
                 item["checksum"].split(":")[1], item["size"])


if __name__ == "__main__":
    main()
