from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, features: np.ndarray):
        self.meta = meta.reset_index(drop=True)
        self.features = features.astype(np.float32)
        if len(self.meta) != len(self.features):
            raise ValueError("metadata and feature count mismatch")
        self.labels = self.meta["label"].astype(float).to_numpy()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)


def load_split(split_dir: str | Path, embedding_dir_name: str = "default"):
    split_dir = Path(split_dir)
    emb_dir = split_dir / "embeddings" / embedding_dir_name
    train = EmbeddingDataset(pd.read_csv(split_dir / "train.csv"), np.load(emb_dir / "train.npy"))
    val = EmbeddingDataset(pd.read_csv(split_dir / "val.csv"), np.load(emb_dir / "val.npy"))
    test = EmbeddingDataset(pd.read_csv(split_dir / "test.csv"), np.load(emb_dir / "test.npy"))
    return train, val, test
