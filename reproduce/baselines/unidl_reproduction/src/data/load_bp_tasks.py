from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils import read_table


def load_task_table(task_row: pd.Series, base_dir: str | Path | None = None) -> pd.DataFrame:
    input_path = Path(str(task_row["input_path"]))
    if not input_path.is_absolute() and base_dir is not None:
        input_path = Path(base_dir) / input_path
    return read_table(input_path)


def load_tasks(tasks_csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(tasks_csv)
