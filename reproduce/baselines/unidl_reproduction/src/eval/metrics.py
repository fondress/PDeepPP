from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sn = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else np.nan,
        "sn": sn,
        "sp": sp,
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) == 2 else 0.0,
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
