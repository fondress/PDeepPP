from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from code.utils import ensure_dir


VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
PIPELINE_NAME = "unidl_sequence_mean_minmax"


class EmbeddingPretrainedModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embedding(indices))


def _normalize_sequence(value, split_name: str, row_index: int) -> str:
    if pd.isna(value):
        raise ValueError(f"{split_name}: missing sequence at CSV row {row_index + 2}")
    sequence = "".join(str(value).strip().upper().split())
    sequence = "".join(residue if residue in VALID_AA else "X" for residue in sequence)
    if not sequence:
        raise ValueError(f"{split_name}: empty sequence at CSV row {row_index + 2}")
    return sequence


def _model_layer_and_dimension(model: nn.Module) -> tuple[int, int]:
    layers = getattr(model, "num_layers", None)
    args = getattr(model, "args", None)
    if layers is None and args is not None:
        layers = getattr(args, "layers", None)
    dimension = getattr(model, "embed_dim", None)
    if dimension is None and args is not None:
        dimension = getattr(args, "embed_dim", None)
    if layers is None or dimension is None:
        raise ValueError("Unable to infer ESM layer count or embedding dimension")
    return int(layers), int(dimension)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _load_local_esm(esm_model_path: str | Path, device: torch.device):
    import esm

    model_path = Path(esm_model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Local ESM model not found: {model_path}")

    # fair-esm checkpoints contain trusted argparse metadata and predate the
    # PyTorch 2.6 change that made weights_only=True the torch.load default.
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_path))
    finally:
        torch.load = original_torch_load
    return model.to(device).eval(), alphabet


def _encode_batch_unidl_style(
    sequences: list[str],
    labels: list[str],
    model: nn.Module,
    alphabet,
    layer: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter(list(zip(labels, sequences)))
    lengths = (tokens != alphabet.padding_idx).sum(1)
    tokens = tokens.to(device)
    with torch.inference_mode():
        output = model(tokens, repr_layers=[layer], return_contacts=False)
    token_representations = output["representations"][layer]
    pooled = []
    residues = []
    for index, sequence in enumerate(sequences):
        representation = token_representations[index, 1 : len(sequence) + 1]
        if representation.shape[0] != len(sequence):
            raise RuntimeError(
                f"ESM residue length mismatch for {labels[index]}: "
                f"expected={len(sequence)}, actual={representation.shape[0]}"
            )
        residues.append(representation)
        pooled.append(representation.mean(dim=0))
    return torch.stack(pooled), residues


def _fit_minmax(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = train.min(axis=0).astype(np.float32, copy=False)
    maximum = train.max(axis=0).astype(np.float32, copy=False)
    return minimum, maximum


def _transform_minmax(values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    scale = maximum - minimum
    safe_scale = np.where(scale == 0, 1.0, scale)
    return ((values - minimum) / safe_scale).astype(np.float32, copy=False)


def bp_unidl_embeddings_exist(
    split_dir: str | Path,
    output_root: str | Path,
    esm_ratios: Iterable[float],
) -> bool:
    split_dir = Path(split_dir)
    root = Path(output_root) / split_dir.parent.name / split_dir.name
    for ratio in esm_ratios:
        ratio_root = root / str(float(ratio))
        for split_name in ("train", "val", "test"):
            target = ratio_root / split_name
            manifest_path = target / "manifest.json"
            if not manifest_path.is_file():
                return False
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if manifest.get("embedding_pipeline") != PIPELINE_NAME:
                return False
            shards = manifest.get("shards", [])
            if not shards or not all((target / item["file"]).is_file() for item in shards):
                return False
    return True


def _save_array_shards(
    values: np.ndarray,
    target: Path,
    split_name: str,
    shard_size: int,
    manifest_extra: dict,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.npy"):
        old.unlink()
    shards = []
    for shard_index, start in enumerate(range(0, len(values), shard_size)):
        subset = values[start : start + shard_size]
        path = target / f"{split_name}_{shard_index:05d}.npy"
        np.save(path, subset, allow_pickle=False)
        shards.append({"file": path.name, "start": start, "size": len(subset)})
    manifest = {
        "split": split_name,
        "size": len(values),
        "shard_size": shard_size,
        "shape": list(values.shape),
        "dtype": "float32",
        "shards": shards,
        **manifest_extra,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _extract_split(
    frame: pd.DataFrame,
    split_name: str,
    model: nn.Module,
    alphabet,
    layer: int,
    embedding_dim: int,
    projection: EmbeddingPretrainedModel,
    vocab: dict[str, int],
    ratios: list[float],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    if "sequence" not in frame.columns:
        raise ValueError(f"{split_name}: required column 'sequence' is missing")
    sequences = [
        _normalize_sequence(value, split_name, row_index)
        for row_index, value in frame["sequence"].items()
    ]
    pooled_batches = []
    combined_by_ratio: dict[float, list[np.ndarray]] = {ratio: [] for ratio in ratios}
    for batch_start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[batch_start : batch_start + batch_size]
        labels = [f"{split_name}:{batch_start + index}" for index in range(len(batch_sequences))]
        pooled, residue_representations = _encode_batch_unidl_style(
            batch_sequences, labels, model, alphabet, layer, device
        )
        pooled_batches.append(pooled.cpu().numpy().astype(np.float32))
        for index, sequence in enumerate(batch_sequences):
            indices = torch.tensor(
                [vocab.get(residue, 0) for residue in sequence],
                dtype=torch.long,
                device=device,
            )
            with torch.inference_mode():
                learned = projection(indices)
            esm_residues = residue_representations[index]
            for ratio in ratios:
                combined = ratio * esm_residues + (1.0 - ratio) * learned
                combined_by_ratio[ratio].append(combined.mean(dim=0).cpu().numpy().astype(np.float32))
    pooled_array = (
        np.concatenate(pooled_batches, axis=0)
        if pooled_batches
        else np.empty((0, embedding_dim), dtype=np.float32)
    )
    combined_arrays = {
        ratio: np.stack(values).astype(np.float32, copy=False)
        if values
        else np.empty((0, embedding_dim), dtype=np.float32)
        for ratio, values in combined_by_ratio.items()
    }
    return pooled_array, combined_arrays


def build_bp_embeddings_unidl_aligned(
    split_dir: str | Path,
    output_root: str | Path,
    esm_ratios: Iterable[float],
    batch_size: int = 16,
    device_name: str = "auto",
    esm_model_path: str | Path = "esm2_t33_650M_UR50D.pt",
    shard_size: int = 1000,
    embedding_seed: int = 42,
) -> None:
    random.seed(embedding_seed)
    np.random.seed(embedding_seed)
    torch.manual_seed(embedding_seed)
    device = _resolve_device(device_name)
    model, alphabet = _load_local_esm(esm_model_path, device)
    layer, embedding_dim = _model_layer_and_dimension(model)
    split_dir = Path(split_dir)
    frames = {name: pd.read_csv(split_dir / f"{name}.csv") for name in ("train", "val", "test")}
    ratios = sorted({float(ratio) for ratio in esm_ratios})
    if not ratios:
        raise ValueError("esm_ratios must not be empty")
    vocab = {residue: index for index, residue in enumerate("XACDEFGHIKLMNPQRSTVWY")}
    projection = EmbeddingPretrainedModel(len(vocab), 128, embedding_dim).to(device).eval()

    combined_by_split = {}
    for split_name, frame in frames.items():
        print(
            f"[BP-UNIDL] extracting {split_name}: n={len(frame)} model_layer={layer} dim={embedding_dim}",
            flush=True,
        )
        _, combined = _extract_split(
            frame,
            split_name,
            model,
            alphabet,
            layer,
            embedding_dim,
            projection,
            vocab,
            ratios,
            device,
            batch_size,
        )
        combined_by_split[split_name] = combined

    root = Path(output_root) / split_dir.parent.name / split_dir.name
    ensure_dir(root)
    for ratio in ratios:
        ratio_root = ensure_dir(root / str(float(ratio)))
        train_minimum, train_maximum = _fit_minmax(combined_by_split["train"][ratio])
        np.savez(
            ratio_root / "unidl_minmax_scaler.npz",
            data_min=train_minimum,
            data_max=train_maximum,
            data_range=train_maximum - train_minimum,
        )
        for split_name in ("train", "val", "test"):
            values = _transform_minmax(
                combined_by_split[split_name][ratio], train_minimum, train_maximum
            )
            # PDeepPP expects a 3-D tensor; this singleton axis preserves the
            # UniDL sequence-level vector without inventing padded residues.
            values = values[:, None, :]
            _save_array_shards(
                values,
                ratio_root / split_name,
                split_name,
                shard_size,
                {
                    "esm_ratio": ratio,
                    "embedding_pipeline": PIPELINE_NAME,
                    "esm_layer": layer,
                    "embedding_dim": embedding_dim,
                    "sequence_axis": 1,
                },
            )
    print(
        f"[BP-UNIDL] saved split={split_dir} ratios={ratios} shape=(*, 1, {embedding_dim})",
        flush=True,
    )
