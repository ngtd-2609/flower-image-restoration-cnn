"""Verify archive structure, exclusions and SHA256SUMS from the packaged bytes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import PurePosixPath
from zipfile import ZipFile

FORBIDDEN_DIRS = {
    ".git",
    ".venv",
    "legacy",
    "qa",
    "tmp",
    ".tmp",
    ".keras_cache",
    "node_modules",
    "office_qa",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".rar", ".zip"}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()

    with ZipFile(args.archive) as archive:
        names = [PurePosixPath(name) for name in archive.namelist() if not name.endswith("/")]
        roots = {name.parts[0] for name in names if name.parts}
        errors: list[str] = []
        if roots != {"TUNGDUONG_flower-image-restoration-cnn_95plus_merged"}:
            errors.append(f"Unexpected archive roots: {sorted(roots)}")
        for name in names:
            if FORBIDDEN_DIRS.intersection(name.parts) or name.suffix.lower() in FORBIDDEN_SUFFIXES:
                errors.append(f"Forbidden packaged path: {name}")

        checksum_name = PurePosixPath(
            "TUNGDUONG_flower-image-restoration-cnn_95plus_merged/SHA256SUMS.txt"
        )
        if checksum_name not in names:
            errors.append("SHA256SUMS.txt is missing")
        else:
            checksum_text = archive.read(checksum_name.as_posix()).decode("utf-8")
            for line in checksum_text.splitlines():
                expected, relative = line.split("  ", 1)
                member = checksum_name.parent / relative
                if member not in names:
                    errors.append(f"Checksum member is missing: {relative}")
                elif digest(archive.read(member.as_posix())) != expected:
                    errors.append(f"Checksum mismatch: {relative}")

        if errors:
            raise SystemExit("\n".join(errors))
        print(
            f"PACKAGE_VERIFY_PASS files={len(names)} root={next(iter(roots))} "
            f"comment={archive.comment.decode('utf-8', errors='replace')}"
        )


if __name__ == "__main__":
    main()
