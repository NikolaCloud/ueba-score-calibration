"""Check the unpacked review bundle using only Python's standard library."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(f"Missing or invalid path: {relative}")
            continue
        payload = path.read_bytes()
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            failures.append(f"Checksum mismatch: {relative}")
    print(json.dumps(dict(passed=not failures, files_checked=len(manifest["files"]),
                          failures=failures), ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
