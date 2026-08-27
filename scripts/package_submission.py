"""Build the clean student submission ZIP without audit caches or legacy sources."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FINAL_OUTPUT = ROOT.parent / "TUNGDUONG_flower-image-restoration-cnn_95plus_merged.zip"
CANDIDATE_OUTPUT = FINAL_OUTPUT
ARCHIVE_ROOT = Path("TUNGDUONG_flower-image-restoration-cnn_95plus_merged")
EXCLUDED_DIRS = {
    ".git", ".venv", "legacy", "qa", "tmp", ".tmp", ".keras_cache", "node_modules",
    "office_qa", "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".rar", ".zip"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and relative.parts[:2] != ("data", "flower_photos")
        and not EXCLUDED_DIRS.intersection(relative.parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final",
        action="store_true",
        help="Require every final evidence gate before writing the submission-named archive.",
    )
    args = parser.parse_args()
    if args.final:
        subprocess.run(
            [sys.executable, "scripts/validate_project.py", "--require-final"],
            cwd=ROOT,
            check=True,
        )
    output = FINAL_OUTPUT if args.final else CANDIDATE_OUTPUT
    if not (ROOT / "SHA256SUMS.txt").exists():
        raise SystemExit("Run scripts/build_submission_manifest.py before packaging.")
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(ROOT.rglob("*")):
            if included(path):
                archive.write(path, (ARCHIVE_ROOT / path.relative_to(ROOT)).as_posix())
        archive.comment = b"TRAINED_CHECKPOINT_READY; FULL_49_AND_DEPLOY_PENDING_USER_GPU"
    print(output)


if __name__ == "__main__":
    main()
