from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from code.evaluation.metrics import binary_metrics
from code.models.factory import build_model
from code.training.dataset import load_embeddings
from code.training.losses import build_loss
from code.utils import ensure_dir, set_seed


def _device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _features_and_labels(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("A data-loader batch must contain at least features and labels")
    return batch[0], batch[1]


def _predict(model, loader, device, amp: bool = False):
    model.eval()
    probs = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            x, y = _features_and_labels(batch)
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(x)
            probs.append(torch.sigmoid(logits.float()).detach().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def train_one_run(
    task: str,
    split_dir: str | Path,
    model_name: str,
    esm_ratio: float,
    lambda_value: float,
    seed: int,
    output_dir: str | Path,
    task_type: str = "BP",
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 20,
    selection_metric: str = "pr_auc",
    device_name: str = "auto",
    num_workers: int = 0,
    loss_name: str = "tim",
    model_kwargs: Dict | None = None,
    embeddings_root: str | Path | None = None,
    distributed_training: bool = False,
    artifact_root: str | Path | None = None,
    amp: bool = True,
    prefetch_factor: int = 4,
) -> Dict:
    distributed = _is_distributed() and distributed_training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        set_seed(seed)
    else:
        set_seed(seed)
        device = _device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_set, val_set, test_set = load_embeddings(split_dir, esm_ratio, embeddings_root=embeddings_root)
    input_size = int(train_set.embeddings.shape[-1])
    pooling = "center" if task_type.upper() == "PTM" else "mean"
    model_kwargs = model_kwargs or {}
    model = build_model(model_name, input_size=input_size, pooling=pooling, **model_kwargs).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
    }
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=distributed,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    labels = train_set.labels.astype(int)
    pos = max(1, int((labels == 1).sum()))
    neg = max(1, int((labels == 0).sum()))
    criterion = build_loss(loss_name, lambda_value=lambda_value, pos_weight=neg / pos).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    use_amp = bool(amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_score = -np.inf
    best_state = None
    best_epoch = 0
    no_improve = 0
    start_train = time.perf_counter()
    epoch_history = []
    log_root = ensure_dir(artifact_root) if artifact_root is not None else ensure_dir(output_dir).parent
    ensure_dir(log_root / "history")
    ensure_dir(log_root / "logs")
    run_tag = f"{task}_{model_name}_seed{seed}_esm{esm_ratio}_lambda{lambda_value}"
    text_log_path = log_root / "logs" / f"{run_tag}.jsonl"
    if text_log_path.exists():
        text_log_path.unlink()

    val_score_tensor = torch.tensor([-np.inf], device=device)
    stop_tensor = torch.tensor([0], device=device)

    for epoch in range(1, max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_started = time.perf_counter()
        model.train()
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        for batch in train_loader:
            x, y = _features_and_labels(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss_sum += float(loss.detach().cpu()) * len(y)
            epoch_sample_count += len(y)

        if distributed:
            dist.barrier()
        source_model = model.module if distributed else model
        y_val, p_val = _predict(source_model, val_loader, device, use_amp)
        val_metrics = binary_metrics(y_val, p_val)
        score = val_metrics.get(selection_metric, np.nan)
        if np.isnan(score):
            score = -np.inf
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in source_model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        should_stop = no_improve >= patience
        epoch_record = {
            "task": task,
            "task_type": task_type,
            "model": model_name,
            "seed": seed,
            "epoch": epoch,
            "train_loss": epoch_loss_sum / max(1, epoch_sample_count),
            "val_acc": val_metrics.get("acc", np.nan),
            "val_bacc": val_metrics.get("bacc", np.nan),
            "val_mcc": val_metrics.get("mcc", np.nan),
            "val_roc_auc": val_metrics.get("roc_auc", np.nan),
            "val_pr_auc": val_metrics.get("pr_auc", np.nan),
            "selection_metric": selection_metric,
            "selection_score": float(score),
            "best_validation_score": float(best_score),
            "no_improve": no_improve,
            "epoch_time_sec": time.perf_counter() - epoch_started,
            "elapsed_train_time_sec": time.perf_counter() - start_train,
        }
        if not distributed or dist.get_rank() == 0:
            epoch_history.append(epoch_record)
            with text_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record, default=float) + "\n")
        if distributed:
            val_score_tensor.fill_(float(score))
            stop_tensor.fill_(1 if should_stop else 0)
            dist.broadcast(val_score_tensor, src=0)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            break

    train_time_sec = time.perf_counter() - start_train
    if distributed:
        dist.barrier()
    if not distributed or dist.get_rank() == 0:
        source_model = model.module if distributed else model
        if best_state is not None:
            source_model.load_state_dict(best_state)
        start_infer = time.perf_counter()
        y_test, p_test = _predict(source_model, test_loader, device, use_amp)
        inference_time_sec = time.perf_counter() - start_infer
        metrics = binary_metrics(y_test, p_test)
        output_dir = ensure_dir(output_dir)
        artifact_root = ensure_dir(artifact_root) if artifact_root is not None else output_dir.parent
        for folder in ("history", "logs", "predictions", "run_results"):
            ensure_dir(artifact_root / folder)
        tag = run_tag
        checkpoint_path = output_dir / f"{tag}.pt"
        torch.save({
            "model_state_dict": source_model.state_dict(),
            "task": task,
            "model_name": model_name,
            "seed": seed,
            "esm_ratio": esm_ratio,
            "lambda_value": lambda_value,
        }, checkpoint_path)
        pd.DataFrame({"y_true": y_test.astype(int), "y_prob": p_test}).to_csv(artifact_root / "predictions" / f"{tag}.csv", index=False)
        pd.DataFrame(epoch_history).to_csv(artifact_root / "history" / f"{tag}.csv", index=False)
        with (artifact_root / "logs" / f"{tag}.txt").open("w", encoding="utf-8") as handle:
            handle.write(f"task={task}\nmodel={model_name}\nseed={seed}\n")
            handle.write(f"best_epoch={best_epoch}\nbest_validation_score={float(best_score)}\n")
            handle.write(f"epochs_trained={len(epoch_history)}\ntrain_time_sec={train_time_sec}\n")
            handle.write(f"inference_time_sec={inference_time_sec}\n")
    else:
        inference_time_sec = 0.0
        metrics = {}
        checkpoint_path = Path("")

    result = {
        "task": task,
        "task_type": task_type,
        "model": model_name,
        "seed": seed,
        "esm_ratio": esm_ratio,
        "lambda_value": lambda_value,
        "loss_name": loss_name,
        "best_epoch": best_epoch,
        "epochs_trained": len(epoch_history),
        "best_validation_score": float(best_score),
        "train_time_sec": train_time_sec,
        "inference_time_sec": inference_time_sec,
        "throughput_seq_per_sec": len(test_set) / inference_time_sec if inference_time_sec > 0 else np.nan,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "checkpoint_path": str(checkpoint_path),
        **metrics,
    }
    if not distributed or dist.get_rank() == 0:
        if artifact_root is not None:
            (Path(artifact_root) / "run_results" / f"{tag}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
    if distributed:
        dist.barrier()
        if dist.get_rank() == 0:
            return result
        return {}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model on one prepared split and embedding ratio.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-type", default="BP", choices=["BP", "PTM"])
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--model", required=True, choices=["pdeeppp", "esm_linear", "esm_mlp"])
    parser.add_argument("--esm-ratio", type=float, default=1.0)
    parser.add_argument("--lambda-value", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/checkpoints")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--selection-metric", default="pr_auc")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = train_one_run(
        task=args.task,
        task_type=args.task_type,
        split_dir=args.split_dir,
        model_name=args.model,
        esm_ratio=args.esm_ratio,
        lambda_value=args.lambda_value,
        seed=args.seed,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        selection_metric=args.selection_metric,
        device_name=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
