"""Extract selected CERT logs from a completed or range-verified growing archive."""
import argparse
import bz2
import io
import json
import hashlib
import tarfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RANGE_BYTES = 64 * 1024 * 1024


class VerifiedRangeReader(io.RawIOBase):
    def __init__(self, release, size):
        self.path = ROOT / "data/raw" / (release+".tar.bz2")
        self.folder = ROOT / "data/raw" / (release+"_ranges")
        self.size, self.position, self.confirmed = size, 0, set()
        self.handle = self.path.open("rb")
        self.last = time.monotonic()
        self.release = release

    def readable(self):
        return True

    def read(self, length=-1):
        if length < 0:
            raise ValueError("Bounded reads required")
        result = bytearray()
        while len(result) < length and self.position < self.size:
            start = self.position // RANGE_BYTES * RANGE_BYTES
            end = min(start+RANGE_BYTES, self.size)-1
            if start not in self.confirmed:
                mark = self.folder / f"{start}-{end}.json"
                started = time.monotonic()
                while not mark.exists():
                    if time.monotonic()-started > 1800:
                        raise TimeoutError(f"Range not available: {mark.name}")
                    time.sleep(.5)
                self.confirmed.add(start)
            count = min(length-len(result), end-self.position+1)
            self.handle.seek(self.position)
            block = self.handle.read(count)
            if len(block) != count:
                raise IOError("Verified range unexpectedly short")
            result.extend(block)
            self.position += len(block)
            if time.monotonic()-self.last > 30:
                print(f"{self.release} decoding: {self.position/1e9:.3f}/{self.size/1e9:.3f} GB compressed", flush=True)
                self.last = time.monotonic()
        return bytes(result)

    def close(self):
        self.handle.close()
        super().close()


def extract(release="r5.2", stream=False):
    directory = ROOT / "data/metadata" / release
    metadata = json.loads((directory / "cert_metadata.json").read_text())
    item = next(x for x in metadata["files"] if x["name"] == release+".tar.bz2")
    source = ROOT / "data/raw" / item["name"]
    reader = VerifiedRangeReader(release, item["size"]) if stream else None
    if stream:
        decoder = bz2.BZ2File(reader)
    else:
        import indexed_bzip2
        decoder = indexed_bzip2.open(str(source), parallelization=6)
    manifest, archive_members = [], []
    started = time.monotonic()
    try:
        with decoder as decompressed, tarfile.open(fileobj=decompressed, mode="r|", bufsize=4*1024*1024) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe archive path")
                archive_members.append(dict(path=member.name, bytes=member.size))
                wanted = (relative.name in {"logon.csv", "device.csv", "file.csv", "email.csv", "license.txt"}
                          or "readme" in relative.name.lower() or any(p.lower() == "ldap" for p in relative.parts))
                if not wanted:
                    print(f"Skip {member.name}: {member.size/1e9:.3f} GB", flush=True)
                    continue
                destination = ROOT / "data/raw" / Path(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                print(f"Extract {member.name}: {member.size/1e6:.1f} MB", flush=True)
                digest = hashlib.sha256()
                with archive.extractfile(member) as incoming, destination.open("wb") as outgoing:
                    for block in iter(lambda: incoming.read(8*1024*1024), b""):
                        digest.update(block)
                        outgoing.write(block)
                manifest.append(dict(archive_member=member.name, bytes=member.size, sha256=digest.hexdigest()))
    finally:
        if reader is not None:
            reader.close()
    sidecar = source.with_suffix(source.suffix+".provenance.json")
    deadline = time.monotonic()+1800
    while not sidecar.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Full publisher checksum verification not completed")
        time.sleep(1)
    proof = json.loads(sidecar.read_text())
    if proof["md5"] != item["computed_md5"]:
        raise ValueError("Invalid archive proof")
    (directory / "archive_members.json").write_text(json.dumps(archive_members, indent=2), encoding="utf-8")
    (directory / "extracted_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified extraction completed in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="r5.2")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    extract(args.release, args.stream)
