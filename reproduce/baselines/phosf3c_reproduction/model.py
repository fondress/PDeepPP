from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn


def load_conformer_class():
    path = Path(__file__).resolve().parents[0] / "conformer.py"
    spec = importlib.util.spec_from_file_location("phosf3c_original_conformer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    import sys as _sys
    _sys.modules["phosf3c_original_conformer"] = module
    spec.loader.exec_module(module)
    return module.Conformer


class PhosF3C(nn.Module):
    def __init__(self, alpha: float = 0.5, **kwargs):
        super().__init__()
        self.alpha = alpha
        num_classes = int(kwargs.pop("num_classes", 2))
        self.conformer = load_conformer_class()(num_classes=num_classes, **kwargs)

    def forward(self, x):
        result = self.conformer(x)
        return self.alpha * result["conv_cls"] + (1 - self.alpha) * result["tran_cls"]