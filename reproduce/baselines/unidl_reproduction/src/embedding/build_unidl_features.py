from __future__ import annotations

from pathlib import Path

import numpy as np


def save_feature_arrays(train_x, val_x, test_x, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", np.asarray(train_x, dtype=np.float32))
    np.save(out_dir / "val.npy", np.asarray(val_x, dtype=np.float32))
    np.save(out_dir / "test.npy", np.asarray(test_x, dtype=np.float32))


def load_feature_arrays(out_dir: str | Path):
    out_dir = Path(out_dir)
    return np.load(out_dir / "train.npy"), np.load(out_dir / "val.npy"), np.load(out_dir / "test.npy")
