"""Execute the evidence notebook sequentially in a fresh Python process."""
from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "BTL_XuLyAnh_NhanDienHoa.ipynb"
notebook = json.loads(PATH.read_text(encoding="utf-8"))
namespace = {"__name__": "__main__", "display": lambda value: print(value)}
counter = 0
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") != "code":
        continue
    counter += 1
    source = "".join(cell.get("source", []))
    stream = io.StringIO()
    cell["execution_count"] = counter
    cell["outputs"] = []
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            exec(  # noqa: S102
                compile(source, f"{PATH.name}:cell-{counter}", "exec"),
                namespace,
            )
        text = stream.getvalue()
        if text:
            cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": text.splitlines(True)}]
    except Exception as exc:
        text = stream.getvalue()
        if text:
            cell["outputs"].append({"name": "stdout", "output_type": "stream", "text": text.splitlines(True)})
        cell["outputs"].append({
            "ename": type(exc).__name__, "evalue": str(exc), "output_type": "error",
            "traceback": traceback.format_exc().splitlines(),
        })
        PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        raise
PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Executed {counter} code cells without errors")
