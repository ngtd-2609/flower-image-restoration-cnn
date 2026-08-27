from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_matrix import matrix_as_records


def main() -> None:
    output = ROOT / "configs" / "degradation_matrix.json"
    output.write_text(json.dumps(matrix_as_records(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(matrix_as_records())} conditions to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
