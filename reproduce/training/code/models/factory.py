from __future__ import annotations

from code.models.esm_baselines import ESMLinearClassifier, ESMMLPClassifier
from code.models.pdeeppp import PDeepPPClassifier


def build_model(
    model_name: str,
    input_size: int = 1280,
    pooling: str = "mean",
    center_index: int = 16,
    **model_kwargs,
):
    model_name = model_name.lower()
    if model_name == "pdeeppp":
        return PDeepPPClassifier(input_size=input_size, **model_kwargs)
    if model_name == "esm_linear":
        return ESMLinearClassifier(input_size=input_size, pooling=pooling, center_index=center_index)
    if model_name == "esm_mlp":
        return ESMMLPClassifier(input_size=input_size, pooling=pooling, center_index=center_index)
    raise ValueError(f"Unsupported model name: {model_name}")
