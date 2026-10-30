"""Copy and verify reference files from an existing, lawfully held source directory."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="Source root containing data/, models/, outputs/")
    parser.add_argument("--profile", choices=["inputs", "archives", "all"], default="all")
    parser.add_argument("--check", action="store_true", help="Check staged files without copying")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    files = [f for f in manifest["files"] if args.profile == "all" or f["profile"] == args.profile]
    if not args.check and args.source is None:
        parser.error("Supply a source root or use --check")
    for entry in files:
        target = (ROOT / entry["path"]).resolve()
        if not target.is_relative_to(ROOT):
            raise ValueError("Manifest path escapes package")
        if not args.check:
            source = (args.source.resolve() / entry["path"]).resolve()
            if not source.is_relative_to(args.source.resolve()):
                raise ValueError("Manifest path escapes source")
            if digest(source) != entry["sha256"]:
                raise ValueError(f"Source checksum mismatch: {entry['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source != target:
                shutil.copy2(source, target)
        if target.stat().st_size != entry["bytes"] or digest(target) != entry["sha256"]:
            raise ValueError(f"Staged checksum mismatch: {entry['path']}")
    print(f"Verified {len(files)} {args.profile} files")


if __name__ == "__main__":
    main()
