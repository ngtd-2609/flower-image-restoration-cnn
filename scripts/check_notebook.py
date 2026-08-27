from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "BTL_XuLyAnh_NhanDienHoa.ipynb"


def output_text(cell: dict) -> str:
    chunks = []
    for output in cell.get("outputs", []):
        text = output.get("text", "")
        chunks.extend(text if isinstance(text, list) else [text])
        data = output.get("data", {})
        for value in data.values():
            chunks.extend(value if isinstance(value, list) else [value])
    return "\n".join(str(item) for item in chunks)


def inspect_notebook(*, require_executed: bool = False, require_full_run: bool = False) -> tuple[dict, list[str]]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    required_sections = [f"Phần {index}" for index in range(17)]
    missing = [title for title in required_sections if title not in markdown]
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    long_cells = [index for index, cell in enumerate(code_cells, 1) if len(cell.get("source", [])) > 80]
    executed_cells = [cell for cell in code_cells if cell.get("execution_count") is not None]
    error_cells = [
        index
        for index, cell in enumerate(code_cells, 1)
        if any(output.get("output_type") == "error" for output in cell.get("outputs", []))
    ]
    combined_outputs = "\n".join(output_text(cell) for cell in code_cells)
    errors = []
    if missing:
        errors.append(f"Missing notebook sections: {missing}")
    if long_cells:
        errors.append(f"Code cells exceed 80 lines: {long_cells}")
    if require_executed and len(executed_cells) != len(code_cells):
        errors.append(f"Notebook is not Restart/Run All evidence: {len(executed_cells)}/{len(code_cells)} code cells executed")
    if error_cells:
        errors.append(f"Notebook contains error outputs in code cells: {error_cells}")
    if require_full_run:
        marker = "FULL_RUN_COMPLETE"
        if marker not in combined_outputs:
            errors.append(f"Executed outputs do not contain marker: {marker}")
    report = {
        "total_cells": len(notebook["cells"]),
        "code_cells": len(code_cells),
        "executed_code_cells": len(executed_cells),
        "missing_sections": missing,
        "code_cells_over_80_lines": long_cells,
        "error_output_cells": error_cells,
        "full_run_markers_present": "FULL_RUN_COMPLETE" in combined_outputs,
        "require_executed": require_executed,
        "require_full_run": require_full_run,
    }
    return report, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-executed", action="store_true")
    parser.add_argument("--require-full-run", action="store_true")
    args = parser.parse_args()
    report, errors = inspect_notebook(
        require_executed=args.require_executed or args.require_full_run,
        require_full_run=args.require_full_run,
    )
    report["errors"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__": main()
