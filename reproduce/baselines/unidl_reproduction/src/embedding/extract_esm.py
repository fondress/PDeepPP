from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np


class ESM2UnavailableError(RuntimeError):
    pass


@contextmanager
def _allow_legacy_torch_load():
    import torch

    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


def _load_checkpoint_fallback(model_path: Path):
    import torch
    import esm

    with _allow_legacy_torch_load():
        checkpoint = torch.load(str(model_path), map_location="cpu")

    if hasattr(checkpoint, "state_dict"):
        checkpoint = checkpoint.state_dict()

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict", "module"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
        if state_dict is None and all(isinstance(k, str) for k in checkpoint.keys()):
            state_dict = checkpoint
    if state_dict is None:
        raise ESM2UnavailableError(f"Could not infer state_dict structure from checkpoint: {model_path}.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[ESM] Loaded with missing keys={len(missing)}, unexpected keys={len(unexpected)}")
    return model, alphabet


def _load_esm_model(model_path: str | Path | None = None, device_name: str | None = None):
    try:
        import torch
        import esm
    except Exception as exc:
        raise ESM2UnavailableError(
            "esm package is not available. Install fair-esm and make sure torch is available."
        ) from exc

    if device_name is None or device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if model_path is None:
        raise ESM2UnavailableError("Local ESM checkpoint path is required; online download is disabled.")
    model_path = Path(model_path)
    if not model_path.exists():
        raise ESM2UnavailableError(f"Local ESM checkpoint not found: {model_path}")
    try:
        with _allow_legacy_torch_load():
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_path))
    except Exception:
        model, alphabet = _load_checkpoint_fallback(model_path)
    model = model.to(device)
    model.eval()
    return model, alphabet, device


def _infer_batch_size(sequences: Sequence[str], max_batch_tokens: int) -> int:
    if not sequences:
        return 1
    lengths = [len(s) + 2 for s in sequences]
    avg_len = max(1, int(np.median(lengths)))
    batch_size = max(1, max_batch_tokens // avg_len)
    return min(batch_size, len(sequences))


def extract_with_esm2(
    sequences: Sequence[str],
    model_path: str | Path | None = None,
    batch_size: int | None = None,
    max_batch_tokens: int = 2048,
    device_name: str | None = None,
) -> np.ndarray:
    import torch

    model, alphabet, device = _load_esm_model(model_path, device_name=device_name)
    batch_converter = alphabet.get_batch_converter()

    if batch_size is None:
        batch_size = _infer_batch_size(sequences, max_batch_tokens=max_batch_tokens)

    seq_reps = []
    embed_dim = None
    max_len = 0

    for start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[start:start + batch_size]
        batch = [(str(i), seq) for i, seq in enumerate(batch_seqs, start=start)]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[33])
        reps = out["representations"][33]
        for i, seq in enumerate(batch_seqs):
            s = len(seq)
            rep = reps[i, 1 : s + 1].cpu().numpy().astype(np.float32)
            seq_reps.append(rep)
            max_len = max(max_len, rep.shape[0])
            if embed_dim is None:
                embed_dim = rep.shape[1]

    if not seq_reps:
        return np.empty((0, 0, 0), dtype=np.float32)

    padded = np.zeros((len(seq_reps), max_len, embed_dim), dtype=np.float32)
    for i, rep in enumerate(seq_reps):
        padded[i, : rep.shape[0], :] = rep
    return padded
