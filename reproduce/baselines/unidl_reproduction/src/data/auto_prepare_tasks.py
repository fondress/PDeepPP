from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils import ensure_dir
from .adapters.prepare_bio_datasets import prepare_bp_dataset


def auto_prepare_bp_tasks(bp_root: str | Path, output_root: str | Path) -> pd.DataFrame:
    """Scan a raw BP data root, standardize each task directory, and write the task table.

    Each subdirectory of ``bp_root`` is treated as one BP task. Standardized CSVs are
    written under ``output_root/prepared_bp/<task>/<task>.csv`` and the returned table
    uses the same columns as ``config/unidl_tasks.csv``.
    """
    bp_root = Path(bp_root)
    if not bp_root.is_dir():
        raise FileNotFoundError(f"BP root does not exist: {bp_root}")
    output_root = ensure_dir(output_root)
    prepared_root = ensure_dir(output_root / "prepared_bp")

    rows = []
    for task_dir in sorted(p for p in bp_root.iterdir() if p.is_dir()):
        task_name = task_dir.name
        out_dir = ensure_dir(prepared_root / task_name)
        try:
            prepare_bp_dataset(task_dir, task_name, out_dir)
        except FileNotFoundError:
            print(f"[PREP] skip {task_name}: no _train/_test files found", flush=True)
            continue
        rows.append({
            "task": task_name,
            "task_type": "BP",
            "input_path": f"prepared_bp/{task_name}/{task_name}.csv",
            "sequence_col": "sequence",
            "label_col": "label",
            "split_group_col": "split_group",
            "negative_source": "auto-detected",
            "primary_benchmark": "yes",
            "source_task_dir": str(task_dir),
        })
        print(f"[PREP] prepared {task_name}", flush=True)

    if not rows:
        raise FileNotFoundError(f"No BP task directories found under {bp_root}")

    table = pd.DataFrame(rows)
    out_path = output_root / "unidl_tasks.csv"
    table.to_csv(out_path, index=False)
    print(f"[PREP] wrote task table with {len(table)} tasks to {out_path}", flush=True)
    return table
