from __future__ import annotations

import torch
import torch.nn as nn


class TIMInspiredLoss(nn.Module):
    def __init__(self, lambda_value: float = 0.95, alpha: float = 1.0):
        super().__init__()
        self.lambda_value = lambda_value
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float().view(-1)
        logits = logits.float().view(-1)
        bce_loss = self.bce(logits, labels)
        probs = torch.sigmoid(logits)
        eps = torch.finfo(probs.dtype).eps
        probs = probs.clamp(min=eps, max=1.0 - eps)
        entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log1p(-probs)).mean()
        positive_cond_entropy = -(probs * torch.log(probs)).mean()
        regularization_loss = self.alpha * entropy - self.alpha * positive_cond_entropy
        return self.lambda_value * bce_loss + (1 - self.lambda_value) * regularization_loss


def build_loss(loss_name: str = "tim", lambda_value: float = 0.95, pos_weight: float | None = None) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name in {"tim", "tim_inspired"}:
        return TIMInspiredLoss(lambda_value=lambda_value)
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_name == "weighted_bce":
        if pos_weight is None:
            raise ValueError("pos_weight is required for weighted_bce")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    raise ValueError(f"Unsupported loss: {loss_name}")


# --- R7 (user 2026-11-03): the loss as written in Section 2.2 of the manuscript, appended to a copy of code/training/losses.py ---
class TIMPaperLoss(nn.Module):
    """L = lambda * CE - H(Y) + beta * H(Y|X), K = 2, computed on each mini-batch.
    H(Y): entropy of the batch-average predicted class distribution; H(Y|X): mean per-sample predictive entropy; beta = 1."""

    def __init__(self, lambda_value: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.lambda_value = lambda_value
        self.beta = beta

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float().view(-1)
        logits = logits.float().view(-1)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        eps = torch.finfo(probs.dtype).eps
        probs = probs.clamp(min=eps, max=1.0 - eps)
        p_bar = probs.mean()
        h_marginal = -(p_bar * torch.log(p_bar) + (1.0 - p_bar) * torch.log1p(-p_bar))
        h_conditional = -(probs * torch.log(probs) + (1.0 - probs) * torch.log1p(-probs)).mean()
        return self.lambda_value * ce - h_marginal + self.beta * h_conditional


_build_loss_round3 = build_loss


def build_loss(loss_name: str = "tim", lambda_value: float = 0.95, pos_weight: float | None = None) -> nn.Module:
    if loss_name.lower() == "tim_paper":
        return TIMPaperLoss(lambda_value=lambda_value)
    return _build_loss_round3(loss_name, lambda_value=lambda_value, pos_weight=pos_weight)


if __name__ == "__main__":   # self-check against a direct evaluation of the formula
    torch.manual_seed(0)
    z, y = torch.randn(64), (torch.rand(64) < 0.3).float()
    p = torch.sigmoid(z).double()
    ce = -(y * p.log() + (1 - y) * (1 - p).log()).mean()
    hm = -(p.mean() * p.mean().log() + (1 - p.mean()) * (1 - p.mean()).log())
    hc = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
    for lam in (0.5, 1.0, 2.0):
        got = float(build_loss("tim_paper", lambda_value=lam)(z, y))
        assert abs(got - float(lam * ce - hm + hc)) < 1e-5, (lam, got)
    assert isinstance(build_loss("tim", lambda_value=0.5), TIMInspiredLoss)
    print("tim_paper self-check OK")
