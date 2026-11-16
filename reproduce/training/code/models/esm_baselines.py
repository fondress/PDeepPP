from __future__ import annotations

import torch
import torch.nn as nn


def pool_embeddings(embeddings: torch.Tensor, pooling: str = "mean", center_index: int = 16) -> torch.Tensor:
    if embeddings.ndim == 2:
        return embeddings
    if pooling == "mean":
        return embeddings.mean(dim=1)
    if pooling == "center":
        idx = min(center_index, embeddings.size(1) - 1)
        return embeddings[:, idx, :]
    if pooling == "mean_center_concat":
        idx = min(center_index, embeddings.size(1) - 1)
        return torch.cat([embeddings.mean(dim=1), embeddings[:, idx, :]], dim=-1)
    raise ValueError(f"Unsupported pooling: {pooling}")


class ESMLinearClassifier(nn.Module):
    def __init__(self, input_size: int = 1280, pooling: str = "mean", center_index: int = 16):
        super().__init__()
        self.pooling = pooling
        self.center_index = center_index
        pooled_size = input_size * 2 if pooling == "mean_center_concat" else input_size
        self.classifier = nn.Linear(pooled_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = pool_embeddings(x, self.pooling, self.center_index)
        return self.classifier(pooled).squeeze(1)


class ESMMLPClassifier(nn.Module):
    def __init__(self, input_size: int = 1280, hidden_size: int = 256, dropout: float = 0.3, pooling: str = "mean", center_index: int = 16):
        super().__init__()
        self.pooling = pooling
        self.center_index = center_index
        pooled_size = input_size * 2 if pooling == "mean_center_concat" else input_size
        self.classifier = nn.Sequential(
            nn.Linear(pooled_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = pool_embeddings(x, self.pooling, self.center_index)
        return self.classifier(pooled).squeeze(1)
