from __future__ import annotations

import argparse, bisect, json, os, platform, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class EncodedDataset(Dataset):
    def __init__(self, encoded_dir: Path, split_name: str):
        manifest = json.loads((encoded_dir / "manifest.json").read_text(encoding="utf-8")); entries = manifest["splits"][split_name]["shards"]
        if not entries: raise FileNotFoundError(f"No encoded shards found for {split_name} in {encoded_dir}")
        self.shards, self.ends, total = [], [], 0
        for entry in entries:
            prefix = entry["prefix"]; arrays = tuple(np.load(encoded_dir / f"{prefix}_{kind}.npy", mmap_mode="r", allow_pickle=False) for kind in ("tokens", "labels", "ids"))
            self.shards.append(arrays); total += len(arrays[1]); self.ends.append(total)
    def __len__(self): return self.ends[-1]
    def __getitem__(self, index):
        shard = bisect.bisect_right(self.ends, index); local = index - (0 if shard == 0 else self.ends[shard - 1]); tokens, labels, ids = self.shards[shard]
        return torch.tensor(np.asarray(tokens[local]), dtype=torch.long), torch.tensor(float(labels[local])), str(ids[local])


class AttentionPool(nn.Module):
    def __init__(self, channels: int, hidden: int): super().__init__(); self.score = nn.Sequential(nn.Linear(channels, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, x, mask):
        weights = torch.softmax(self.score(x).squeeze(-1).masked_fill(~mask, -1e9), dim=1); return torch.sum(x * weights.unsqueeze(-1), dim=1)


class MusiteDeepClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 21):
        super().__init__(); self.embedding = nn.Embedding(22, embedding_dim, padding_idx=0)
        self.convs = nn.Sequential(nn.Conv1d(embedding_dim, 200, 1, padding="same"), nn.Dropout(.75), nn.ReLU(), nn.Conv1d(200, 150, 9, padding="same"), nn.Dropout(.75), nn.ReLU(), nn.Conv1d(150, 200, 10, padding="same"), nn.ReLU())
        self.attention_a = AttentionPool(200, 10); self.attention_b = AttentionPool(200, 8)
        self.head = nn.Sequential(nn.Linear(400, 149), nn.ReLU(), nn.Dropout(.298224), nn.Linear(149, 8), nn.ReLU(), nn.Linear(8, 1))
    def forward(self, tokens):
        mask = torch.ones_like(tokens, dtype=torch.bool); features = self.convs(self.embedding(tokens).transpose(1, 2)).transpose(1, 2)
        return self.head(torch.cat([self.attention_a(features, mask), self.attention_b(features, mask)], 1)).squeeze(1)


def predict(model, loader, device):
    model.eval(); ys, ps, ids = [], [], []
    with torch.no_grad():
        for x, y, sample_ids in loader: ys.extend(y.numpy()); ps.extend(torch.sigmoid(model(x.to(device))).cpu().numpy()); ids.extend(sample_ids)
    return np.asarray(ys), np.asarray(ps), ids


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any(): value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def best_threshold(y, p):
    candidates = np.unique(np.r_[0.5, p]); scores = [matthews_corrcoef(y, p >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def calculate_metrics(y, probability, threshold=.5):
    pred = (probability >= threshold).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel(); precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"threshold": threshold, "acc": accuracy_score(y, pred), "bacc": balanced_accuracy_score(y, pred), "precision": precision, "recall": recall, "f1": f1,
            "sn": tp/(tp+fn) if tp+fn else 0., "sp": tn/(tn+fp) if tn+fp else 0., "mcc": matthews_corrcoef(y, pred), "roc_auc": roc_auc_score(y, probability),
            "pr_auc": average_precision_score(y, probability), "brier": brier_score_loss(y, probability), "ece": expected_calibration_error(y, probability), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def train_one(task: str, encoded_dir: Path, seed: int, config: dict, output: Path, gpu_id: int | None = None) -> dict:
    set_seed(seed); requested = config.get("device", "auto"); device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested))
    sets = {name: EncodedDataset(encoded_dir, name) for name in ("train", "val", "test")}; loaders = {name: DataLoader(data, batch_size=int(config["batch_size"]), shuffle=name == "train", num_workers=int(config.get("num_workers", 0))) for name, data in sets.items()}
    model = MusiteDeepClassifier(int(config.get("embedding_dim", 21))).to(device); labels = np.asarray([float(sets["train"][i][1]) for i in range(len(sets["train"]))]); positive = max(1, int(labels.sum())); negative = max(1, len(labels)-positive); pos_weight = negative/positive
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device)); optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"])); total_params = sum(p.numel() for p in model.parameters()); trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    history_dir = output/"history"; log_dir = output/"logs"; history_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True); history = []
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    best_score, best_state, stale, best_epoch = -np.inf, None, 0, 0; started = time.perf_counter(); max_epochs = int(config["max_epochs"])
    log_path = log_dir/f"{task}_seed{seed}.jsonl"
    if log_path.exists(): log_path.unlink()
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter(); model.train(); loss_total = 0.0
        for x, y, _ in loaders["train"]: optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(x.to(device)), y.to(device)); loss.backward(); optimizer.step(); loss_total += float(loss.detach()) * len(y)
        y_val, p_val, _ = predict(model, loaders["val"], device); score = average_precision_score(y_val, p_val)
        if score > best_score: best_score, best_state, stale, best_epoch = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0, epoch
        else: stale += 1
        val_metrics = calculate_metrics(y_val, p_val)
        row = {"task": task, "model": "musitedeep", "seed": seed, "epoch": epoch, "train_loss": loss_total/len(sets["train"]), "val_acc": val_metrics["acc"], "val_bacc": val_metrics["bacc"], "val_pr_auc": score, "val_roc_auc": val_metrics["roc_auc"], "val_mcc": val_metrics["mcc"], "best_val_pr_auc": best_score, "stale": stale, "epoch_time_sec": time.perf_counter()-epoch_started, "elapsed_train_time_sec": time.perf_counter()-started}; history.append(row)
        with log_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(row) + "\n")
        if stale >= int(config["early_stopping_patience"]): break
    train_time = time.perf_counter()-started; model.load_state_dict(best_state); y_val, p_val, _ = predict(model, loaders["val"], device); selected_threshold = best_threshold(y_val, p_val)
    started = time.perf_counter(); y, p, ids = predict(model, loaders["test"], device); inference = time.perf_counter()-started; peak = torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else 0.0
    (output/"checkpoints").mkdir(parents=True, exist_ok=True); (output/"predictions").mkdir(parents=True, exist_ok=True); (output/"run_results").mkdir(parents=True, exist_ok=True)
    checkpoint = output/"checkpoints"/f"{task}_seed{seed}.pt"; torch.save({"model_state_dict": model.state_dict(), "task": task, "model": "musitedeep", "seed": seed}, checkpoint)
    pd.DataFrame({"sample_id": ids, "y_true": y.astype(int), "y_prob": p, "y_pred": (p>=.5).astype(int), "y_pred_val_threshold": (p>=selected_threshold).astype(int)}).to_csv(output/"predictions"/f"{task}_seed{seed}.csv", index=False); pd.DataFrame(history).to_csv(history_dir/f"{task}_seed{seed}.csv", index=False)
    (log_dir/f"{task}_seed{seed}.txt").write_text(f"task={task}\nmodel=musitedeep\nseed={seed}\nbest_epoch={best_epoch}\nepochs_trained={len(history)}\nbest_validation_pr_auc={best_score}\ntrain_time_sec={train_time}\ninference_time_sec={inference}\n", encoding="utf-8")
    fixed = calculate_metrics(y, p); tuned = {f"val_threshold_{k}": v for k, v in calculate_metrics(y, p, selected_threshold).items()}
    result = {"task": task, "seed": seed, "model": "musitedeep_pytorch_reimplementation", "protocol": config.get("protocol", "controlled"), "loss_name": "weighted_bce", "positive_weight": pos_weight,
              "best_validation_pr_auc": best_score, "best_epoch": best_epoch, "epochs_trained": len(history), "early_stopped": len(history)<max_epochs, "n_train": len(sets["train"]), "n_val": len(sets["val"]), "n_test": len(sets["test"]),
              "train_positive": positive, "train_negative": negative, "train_time_sec": train_time, "inference_time_sec": inference, "throughput_seq_per_sec": len(y)/max(inference, 1e-12), "total_params": total_params, "trainable_params": trainable_params,
              "device": str(device), "gpu_id": gpu_id, "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "peak_gpu_memory_mb": peak, "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "python_version": platform.python_version(), "pid": os.getpid(), **fixed, **tuned}
    (output/"run_results"/f"{task}_seed{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--task", required=True); parser.add_argument("--encoded-dir", required=True); parser.add_argument("--config", required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args(); result = train_one(args.task, Path(args.encoded_dir), args.seed, json.loads(Path(args.config).read_text(encoding="utf-8")), Path(args.output_root)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
