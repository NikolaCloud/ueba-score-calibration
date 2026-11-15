"""Extract selected regular files, avoiding traversal and links; verify publisher SHA256."""
import hashlib
import json
import argparse
import io
import bz2
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


class GrowingRangeReader(io.RawIOBase):
    """Read sequential verified-range downloads while the downloader fills them."""
    def __init__(self, source, size):
        self.prefix = source.with_suffix(source.suffix + ".partial")
        prefix_size = self.prefix.stat().st_size
        self.parts = [(self.prefix, prefix_size)]
        for start in range(prefix_size, size, 128*1024*1024):
            end = min(start+128*1024*1024, size)-1
            self.parts.append((source.parent / "cert_chunks" / f"{start}-{end}.part", end-start+1))
        self.part, self.offset = 0, 0

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Bounded reads required")
        payload = bytearray()
        waiting_since = time.monotonic()
        while len(payload) < size and self.part < len(self.parts):
            path, expected_size = self.parts[self.part]
            if self.offset == expected_size:
                self.part += 1
                self.offset = 0
                continue
            available = path.stat().st_size if path.exists() else 0
            if available <= self.offset:
                if time.monotonic()-waiting_since > 1800:
                    raise TimeoutError(f"Waiting for {path}")
                time.sleep(.5)
                continue
            length = min(size-len(payload), available-self.offset, expected_size-self.offset)
            with path.open("rb") as handle:
                handle.seek(self.offset)
                block = handle.read(length)
            payload.extend(block)
            self.offset += len(block)
            waiting_since = time.monotonic()
        return bytes(payload)


def verify_archive(source):
    expected = dict(line.split()[::-1] for line in
                    (ROOT / "data/raw/CHECKSUMS-sha256.txt").read_text().splitlines())
    sidecar = source.with_suffix(source.suffix + ".provenance.json")
    started = time.monotonic()
    while not sidecar.exists():
        if time.monotonic()-started > 1800:
            raise TimeoutError("Archive checksum validation not completed")
        time.sleep(1)
    provenance = json.loads(sidecar.read_text())
    if provenance["sha256"] != expected[source.name]:
        raise ValueError("Publisher SHA256 mismatch")


def extract(from_ranges=False):
    source = ROOT / "data/raw/r4.2.tar.bz2"
    if not from_ranges:
        verify_archive(source)
    wanted = {"logon.csv", "device.csv", "file.csv", "email.csv"}
    manifest = []
    started = last = time.monotonic()
    # Streaming mode avoids building a huge tar index or materializing excluded HTTP.
    size = 4824287500
    compressed = GrowingRangeReader(source, size) if from_ranges else source
    # CERT uses concatenated bzip2 streams: BZ2File supports them, while tarfile's
    # internal r|bz2 decompressor stops at the first stream.
    with bz2.BZ2File(compressed, "rb") as decompressed, tarfile.open(
            fileobj=decompressed, mode="r|", bufsize=4 * 1024 * 1024) as archive:
        for item in archive:
            if time.monotonic() - last > 25:
                print(f"Archive scan {time.monotonic()-started:.0f}s: {item.name}", flush=True)
                last = time.monotonic()
            relative = PurePosixPath(item.name)
            if not item.isfile():
                continue
            name = relative.name
            if name not in wanted and name != "license.txt" and "LDAP" not in relative.parts and "readme" not in name.lower():
                print(f"Skip {item.name} ({item.size / 1e9:.3f} GB)", flush=True)
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(item.name)
            target = ROOT / "data/raw" / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Extract {item.name} ({item.size / 1e6:.1f} MB)", flush=True)
            digest = hashlib.sha256()
            if target.exists() and target.stat().st_size == item.size:
                with target.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
            else:
                with archive.extractfile(item) as handle, target.open("wb") as out:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                        out.write(block)
            manifest.append(dict(archive_member=item.name, bytes=item.size, sha256=digest.hexdigest()))
            # Once selected logs and LDAP are present we still scan to EOF to establish
            # completeness; excluded HTTP is decompressed but never written to disk.
    output = ROOT / "data/metadata/extracted_manifest.json"
    verify_archive(source)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Extraction complete in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-ranges", action="store_true")
    extract(parser.parse_args().from_ranges)
