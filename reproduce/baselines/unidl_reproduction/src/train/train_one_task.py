from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..eval.metrics import binary_metrics
from ..model.unidl_model import UniDL4BioPepClassifier
from .dataset import load_split
from .losses import build_loss
from ..utils import ensure_dir, set_seed


def _device(name: str):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else (name if name != "auto" else "cpu"))


def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            ys.append(y.numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one_task(task: str, split_dir: str | Path, seed: int, output_dir: str | Path, batch_size: int = 64, lr: float = 1e-3, max_epochs: int = 100, patience: int = 20, device_name: str = "auto", loss_name: str = "tim", lambda_value: float = 1.0, pr_auc_target: float | None = None):
    set_seed(seed)
    device = _device(device_name)
    print(f"[{task} seed={seed}] loading embeddings", flush=True)
    train_set, val_set, test_set = load_split(split_dir)
    print(
        f"[{task} seed={seed}] embeddings loaded train={len(train_set)} "
        f"shape={train_set.features.shape}",
        flush=True,
    )
    model = UniDL4BioPepClassifier(input_size=train_set.features.shape[-1]).to(device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    pos = max(1, int((train_set.labels == 1).sum()))
    neg = max(1, int((train_set.labels == 0).sum()))
    criterion = build_loss(loss_name, lambda_value=lambda_value, pos_weight=neg / pos).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = -np.inf
    best_state = None
    best_epoch = 0
    no_improve = 0
    stop_state = None
    stop_epoch = 0
    stopped_by_target = False
    start_train = time.perf_counter()
    history = []
    output_dir = ensure_dir(output_dir)
    artifact_root = output_dir.parent
    history_dir = ensure_dir(artifact_root / "history")
    log_dir = ensure_dir(artifact_root / "logs")
    log_path = log_dir / f"{task}_unidl_seed{seed}.jsonl"
    if log_path.exists():
        log_path.unlink()

    for epoch in range(max_epochs):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(y)
            sample_count += len(y)

        y_val, p_val = predict(model, val_loader, device)
        val_metrics = binary_metrics(y_val, p_val)
        score = val_metrics["pr_auc"]
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"[{task} seed={seed}] epoch={epoch + 1}/{max_epochs} "
                f"val_pr_auc={score:.6f}",
                flush=True,
            )
        if pr_auc_target is not None and score >= pr_auc_target:
            stopped_by_target = True
            stop_epoch = epoch + 1
            stop_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[{task} seed={seed}] target reached epoch={stop_epoch} "
                f"val_pr_auc={score:.6f} >= target={pr_auc_target:.6f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_epoch = stop_epoch
                best_state = {k: v.detach().cpu().clone() for k, v in stop_state.items()}
                no_improve = 0
            epoch_record = {
                "task": task,
                "model": "unidl",
                "seed": seed,
                "epoch": stop_epoch,
                "train_loss": loss_sum / max(1, sample_count),
                "val_acc": val_metrics.get("acc", np.nan),
                "val_bacc": val_metrics.get("bacc", np.nan),
                "val_mcc": val_metrics.get("mcc", np.nan),
                "val_roc_auc": val_metrics.get("roc_auc", np.nan),
                "val_pr_auc": val_metrics.get("pr_auc", np.nan),
                "best_validation_pr_auc": float(best_score),
                "no_improve": no_improve,
                "epoch_time_sec": time.perf_counter() - epoch_started,
                "elapsed_train_time_sec": time.perf_counter() - start_train,
                "pr_auc_target": pr_auc_target,
                "stopped_by_target": stopped_by_target,
                "stop_epoch": stop_epoch,
            }
            history.append(epoch_record)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record, default=float) + "\n")
            break
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        epoch_record = {
            "task": task,
            "model": "unidl",
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(1, sample_count),
            "val_acc": val_metrics.get("acc", np.nan),
            "val_bacc": val_metrics.get("bacc", np.nan),
            "val_mcc": val_metrics.get("mcc", np.nan),
            "val_roc_auc": val_metrics.get("roc_auc", np.nan),
            "val_pr_auc": val_metrics.get("pr_auc", np.nan),
            "best_validation_pr_auc": float(best_score),
            "no_improve": no_improve,
            "epoch_time_sec": time.perf_counter() - epoch_started,
            "elapsed_train_time_sec": time.perf_counter() - start_train,
        }
        history.append(epoch_record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, default=float) + "\n")
        if no_improve >= patience:
            break

    train_time = time.perf_counter() - start_train
    if stopped_by_target and stop_state is not None:
        model.load_state_dict(stop_state)
    elif best_state is not None:
        model.load_state_dict(best_state)
    start_inf = time.perf_counter()
    y_test, p_test = predict(model, test_loader, device)
    infer_time = time.perf_counter() - start_inf
    metrics = binary_metrics(y_test, p_test)

    pd.DataFrame(history).to_csv(history_dir / f"{task}_unidl_seed{seed}.csv", index=False)
    summary_path = log_dir / f"{task}_unidl_seed{seed}.txt"
    summary_path.write_text(
        f"task={task}\nmodel=unidl\nseed={seed}\nbest_epoch={best_epoch}\n"
        f"epochs_trained={len(history)}\nbest_validation_pr_auc={best_score}\n"
        f"train_time_sec={train_time}\ninference_time_sec={infer_time}\n"
        f"pr_auc_target={pr_auc_target}\nstopped_by_target={stopped_by_target}\n"
        f"stop_epoch={stop_epoch}\n",
        encoding="utf-8",
    )
    torch.save(
        {"model_state_dict": model.state_dict(), "task": task, "model": "unidl", "seed": seed},
        output_dir / f"{task}_seed{seed}.pt",
    )
    result = {
        "task": task,
        "seed": seed,
        "model": "unidl",
        "loss_name": loss_name,
        "lambda_value": lambda_value,
        "best_validation_pr_auc": float(best_score),
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "throughput_seq_per_sec": len(test_set) / infer_time if infer_time > 0 else np.nan,
        "pr_auc_target": pr_auc_target,
        "stopped_by_target": stopped_by_target,
        "stop_epoch": stop_epoch,
        **metrics,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/checkpoints")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pr-auc-target", type=float, default=None, help="Stop as soon as validation PR AUC reaches this value.")
    args = parser.parse_args()
    print(train_one_task(args.task, args.split_dir, args.seed, args.output_dir, args.batch_size, args.lr, args.max_epochs, args.patience, args.device, "tim", 1.0, args.pr_auc_target))


if __name__ == "__main__":
    main()
