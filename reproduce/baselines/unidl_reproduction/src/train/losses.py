from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        labels = labels.float()
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(labels == 1, p, 1 - p)
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


class TimInspiredLoss(nn.Module):
    def __init__(self, lambda_value: float = 1.0, pos_weight: float = 1.0):
        super().__init__()
        self.lambda_value = lambda_value
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, labels):
        labels = labels.float()
        bce = self.bce(logits, labels)
        probs = torch.sigmoid(logits)
        ent = -(probs * torch.log(probs + 1e-12) + (1 - probs) * torch.log(1 - probs + 1e-12)).mean()
        reg = ent
        return self.lambda_value * bce + (1 - self.lambda_value) * reg


def build_loss(loss_name: str, lambda_value: float = 1.0, pos_weight: float = 1.0):
    loss_name = loss_name.lower()
    if loss_name in {"bce", "ce"}:
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    if loss_name == "focal":
        return BinaryFocalLoss()
    if loss_name in {"tim", "tim_inspired"}:
        return TimInspiredLoss(lambda_value=lambda_value, pos_weight=pos_weight)
    raise ValueError(f"Unsupported loss name: {loss_name}")
