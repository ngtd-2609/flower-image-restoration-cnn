"""Create deterministic SHA-256 entries for every file in the clean submission tree."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS.txt"
EXCLUDED_DIRS = {
    ".git", ".venv", "legacy", "qa", "tmp", ".tmp", ".keras_cache", "node_modules",
    "office_qa", "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".rar", ".zip"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != OUTPUT
        and relative.parts[:2] != ("data", "flower_photos")
        and not EXCLUDED_DIRS.intersection(relative.parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in sorted(ROOT.rglob("*"))
        if included(path)
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {OUTPUT}")


if __name__ == "__main__":
    main()
