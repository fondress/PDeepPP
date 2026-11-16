from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split

from ..utils import ensure_dir, write_json


def _split_with_groups(df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if df["split_group"].nunique() >= 3:
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["split_group"].astype(str)))
            return df.iloc[train_idx].copy(), df.iloc[test_idx].copy(), "group"
        except ValueError:
            pass
    if df["label"].nunique() == 2 and df["label"].value_counts().min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, df["label"]))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy(), "stratified"
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    return train_df.copy(), test_df.copy(), "random"


def make_bp_splits(df: pd.DataFrame, task: str, output_dir: str | Path, split_seed: int, test_size: float, val_size: float) -> dict:
    output_dir = ensure_dir(output_dir)
    train_val, test, first_strategy = _split_with_groups(df, test_size=test_size, seed=split_seed)
    relative_val = val_size / max(1e-12, 1 - test_size)
    train, val, second_strategy = _split_with_groups(train_val, test_size=relative_val, seed=split_seed + 1000)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    metadata = {
        "task": task,
        "split_seed": split_seed,
        "test_size": test_size,
        "val_size": val_size,
        "first_strategy": first_strategy,
        "second_strategy": second_strategy,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
    }
    write_json(metadata, output_dir / "metadata.json")
    return metadata
