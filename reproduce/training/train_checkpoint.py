"""R7 copy of the R6 runner: identical except that the best-validation weights are saved to <out>/checkpoints/<run_id>.pt (state_dict only, no epoch).
R6 controlled re-run: component, objective and lean-baseline comparisons.

One process trains one (task, variant, loss) configuration for a list of seeds on one GPU.
v2 additions (off by default, results identical in protocol): --shm-cache keeps the float16 tensors of a
(task, split, ratio) in /dev/shm as .npy files that every process memory-maps (needed for the 1.75 M-window
pS/T training set, which does not fit in GPU memory); with data on the CPU, the next mini-batch is gathered
in a background thread while the GPU computes. --amp-dtype bfloat16 replaces float16 autocast (used for pS/T, where
float16 activations overflowed after ~30 epochs and produced NaN losses). A run whose loss stays non-finite for 3
epochs stops early and keeps its best-validation weights (diverged epochs are never selected). --weight-decay / --dropout (defaults
unchanged) are used only for the additional regularisation search.

Protocol (identical for every variant):
  * fixed partitions read from the split directory (train/val/test CSV);
  * cached ESM-2 embeddings read with code.training.dataset.load_embeddings;
  * Adam, fixed learning rate, AMP, family-specific batch size;
  * checkpoint selected only by validation average precision with patience;
  * the test partition is predicted once, after training, with the restored
    best-validation weights. No test quantity is read during training.

Variants share the exact PDeepPP classes in code.models.pdeeppp; `full` is
PDeepPPClassifier itself, so its parameter count equals the main comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
import fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))   # the project code is next to this script
sys.modules.pop("code", None)  # the stdlib `code` module must not shadow the project package
from code.models.esm_baselines import ESMLinearClassifier, ESMMLPClassifier
from code.models.pdeeppp import PDeepPPClassifier, PosCNN, TransConv1d
from code.revision.loss_comparison.losses import build_loss
from code.training.dataset import load_embeddings
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, matthews_corrcoef, roc_auc_score)

VARIANTS = ["full", "wo_poscnn", "wo_translinear", "wo_attention", "wo_pe", "esm_mlp", "esm_linear"]


class NoAttentionProjection(nn.Module):
    """SelfAttentionGlobalFeatures with the attention/residual step removed."""

    def __init__(self, input_size, output_size, hidden_size=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.fc1(x)))


class PDeepPPAblation(nn.Module):
    """PDeepPPClassifier with one component removed; all other layers unchanged."""

    def __init__(self, variant, input_size=1280, output_size=128, use_position_encoding=True,
                 transformer_heads=8, transformer_hidden=256, transformer_layers=4,
                 transformer_ff_dim=512, cnn_channels=64, fusion_channels=32, dropout=0.15):
        super().__init__()
        self.use_transformer = variant != "wo_translinear"
        self.use_cnn = variant != "wo_poscnn"
        if self.use_transformer:
            self.transformer = TransConv1d(input_size, output_size, num_heads=transformer_heads,
                                           hidden_size=transformer_hidden, num_layers=transformer_layers,
                                           ff_dim=transformer_ff_dim)
            if variant == "wo_attention":
                self.transformer.self_attention_global_features = NoAttentionProjection(
                    input_size, output_size, transformer_hidden)
        if self.use_cnn:
            self.cnn = PosCNN(input_size, output_size,
                              use_position_encoding=use_position_encoding and variant != "wo_pe",
                              conv_channels=cnn_channels)
        in_channels = output_size * (int(self.use_transformer) + int(self.use_cnn))
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels, fusion_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Dropout(dropout),
            nn.Conv1d(fusion_channels, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Dropout(dropout), nn.Flatten(), nn.Linear(64, 1))

    def forward(self, x):
        if self.use_transformer and self.use_cnn:
            t = self.transformer(x)
            c = self.cnn(x).unsqueeze(1).expand(-1, t.size(1), -1)
            z = torch.cat([t, c], dim=2)
        elif self.use_transformer:
            z = self.transformer(x)
        else:
            z = self.cnn(x).unsqueeze(1)
        return self.cnn_layers(z.permute(0, 2, 1)).squeeze(1)


MODEL_KWARGS = dict(output_size=128, use_position_encoding=True, transformer_heads=8, transformer_hidden=256,
                    transformer_layers=4, transformer_ff_dim=512, cnn_channels=64, fusion_channels=32, dropout=0.15)


def build(variant, task_type, input_size, dropout=None):
    pooling = "center" if task_type == "PTM" else "mean"
    kw = dict(MODEL_KWARGS, **({"dropout": dropout} if dropout is not None else {}))
    if variant == "full":
        return PDeepPPClassifier(input_size=input_size, **kw)
    if variant == "esm_mlp":
        return ESMMLPClassifier(input_size=input_size, pooling=pooling, center_index=16,
                                **({"dropout": dropout} if dropout is not None else {}))
    if variant == "esm_linear":
        return ESMLinearClassifier(input_size=input_size, pooling=pooling, center_index=16)
    return PDeepPPAblation(variant, input_size=input_size, **kw)


def to_tensor(dataset, dtype):
    """Materialise an EmbeddingDataset (sharded mmap) into one contiguous tensor."""
    parts = [torch.from_numpy(np.asarray(shard, dtype=np.float32)).to(dtype) for shard in dataset.shards]
    x = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    y = torch.from_numpy(dataset.labels.astype(np.float32))
    assert len(x) == len(y)
    return x, y


def shm_tensors(a, dtype):
    """float16 (x, y) tensors for train/val/test, memory-mapped from a /dev/shm cache shared by all processes."""
    d = Path(a.shm_cache) / f"{a.task}__{Path(a.split_dir).parent.parent.name}__{Path(a.split_dir).name}__{float(a.esm_ratio)}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)  # first process materialises, the others wait and then map
        if not (d / "ready").is_file():
            sets = load_embeddings(a.split_dir, a.esm_ratio, embeddings_root=a.embeddings_root)
            for name, ds in zip(("train", "val", "test"), sets):
                n = sum(len(sh) for sh in ds.shards)
                mm = np.lib.format.open_memmap(d / f"{name}_x.npy", mode="w+", dtype=np.float16, shape=(n, *ds.shards[0].shape[1:]))
                c = 0
                for sh in ds.shards:
                    mm[c:c + len(sh)] = np.asarray(sh, dtype=np.float32).astype(np.float16); c += len(sh)
                mm.flush(); del mm
                np.save(d / f"{name}_y.npy", ds.labels.astype(np.float32))
            (d / "ready").write_text("ok")
        fcntl.flock(lk, fcntl.LOCK_UN)
    out = []
    for name in ("train", "val", "test"):
        x = torch.from_numpy(np.load(d / f"{name}_x.npy", mmap_mode="r"))
        y = torch.from_numpy(np.load(d / f"{name}_y.npy"))
        out.append((x if dtype == torch.float16 else x.to(dtype), y))
    return out


AMP_DTYPE = torch.float16


def predict(model, x, device, batch, amp):
    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(x), batch):
            xb = x[s:s + batch].to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=amp):
                out.append(torch.sigmoid(model(xb).float()).cpu())
    return torch.cat(out).numpy()


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    v = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            v += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(v)


def metrics(y, p, t=0.5):
    z = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, z, labels=[0, 1]).ravel()
    return {"threshold": float(t), "acc": float((z == y).mean()), "bacc": float(balanced_accuracy_score(y, z)),
            "sn": tp / max(1, tp + fn), "sp": tn / max(1, tn + fp), "mcc": float(matthews_corrcoef(y, z)),
            "precision": tp / max(1, tp + fp), "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "ece": ece(y, p), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "prevalence": float(y.mean())}


def val_threshold(y, p):
    """Validation-only threshold maximising MCC (evaluated on a quantile grid)."""
    cand = np.unique(np.r_[0.5, np.quantile(p, np.linspace(0.001, 0.999, 999))])
    return float(cand[int(np.argmax([matthews_corrcoef(y, p >= c) for c in cand]))])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_seed(a, seed, data, out, load_sec):
    (xtr, ytr), (xva, yva), (xte, yte) = data
    run_id = f"{a.task}__{a.tag}__seed{seed}"
    result_path = out / "run_results" / f"{run_id}.json"
    if result_path.is_file():
        print(f"[SKIP] {run_id} exists", flush=True)
        return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    batch = a.batch_size or (4096 if a.task_type == "PTM" else 512)
    torch.cuda.reset_peak_memory_stats(device)
    model = build(a.variant, a.task_type, int(xtr.shape[-1]), a.dropout).to(device)
    pos = float(ytr.sum().item()); neg = float(len(ytr) - pos)
    criterion = build_loss(a.loss, lambda_value=a.lambda_value, pos_weight=neg / max(pos, 1.0)).to(device)
    opt = (torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay) if a.weight_decay > 0
           else torch.optim.Adam(model.parameters(), lr=a.lr))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)
    gen = torch.Generator(device=xtr.device); gen.manual_seed(seed)

    best, best_epoch, stale, best_state, hist = -np.inf, 0, 0, None, []
    pool = ThreadPoolExecutor(1)
    yv = yva.numpy().astype(int)
    t1 = time.perf_counter()
    for epoch in range(1, a.max_epochs + 1):
        model.train(); perm = torch.randperm(len(ytr), device=xtr.device, generator=gen); loss_sum = 0.0
        on_cpu = xtr.device.type == "cpu"
        gather = (lambda st: (xtr[torch.sort(perm[st:st + batch]).values].pin_memory(), ytr[torch.sort(perm[st:st + batch]).values])) if on_cpu else None
        fut = pool.submit(gather, 0) if on_cpu else None
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            if on_cpu:
                xc, yc = fut.result()
                if s + batch < len(perm):
                    fut = pool.submit(gather, s + batch)
                xb = xc.to(device, non_blocking=True).float(); yb = yc.to(device, non_blocking=True)
            else:
                xb = xtr[idx].to(device, non_blocking=True).float(); yb = ytr[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum += float(loss.detach()) * len(idx)
        pv = predict(model, xva, device, batch, True)
        finite = bool(np.isfinite(pv).all())
        score = float(average_precision_score(yv, pv)) if finite else -np.inf  # diverged epoch: never selected
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        hist.append({"epoch": epoch, "train_loss": loss_sum / len(perm), "val_pr_auc": score,
                     "best_val_pr_auc": best, "stale": stale, "elapsed_sec": time.perf_counter() - t1})
        if a.verbose_epochs:
            print(f"[EPOCH] {a.task} {a.tag} seed{seed} ep{epoch} loss={hist[-1]['train_loss']:.4f} valAP={score:.4f} "
                  f"best={best:.4f}@{best_epoch} t={hist[-1]['elapsed_sec']:.0f}s", flush=True)
        if stale >= a.patience:
            break
        if len(hist) >= 3 and not any(np.isfinite(h["train_loss"]) for h in hist[-3:]):
            print(f"[DIVERGED] {run_id} non-finite loss for 3 epochs; stopping, best-validation weights restored", flush=True)
            break
    train_sec = time.perf_counter() - t1

    model.load_state_dict(best_state)
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.save({n: v.detach().cpu() for n, v in best_state.items()}, out / "checkpoints" / f"{run_id}.pt")  # best-validation weights only; no epoch stored
    pv = predict(model, xva, device, batch, True)
    thr = val_threshold(yv, pv)
    t2 = time.perf_counter()
    pt = predict(model, xte, device, batch, True)
    infer_sec = time.perf_counter() - t2
    yt = yte.numpy().astype(int)

    res = {"task": a.task, "task_type": a.task_type, "tag": a.tag, "variant": a.variant, "loss": a.loss,
           "lambda_value": a.lambda_value, "esm_ratio": a.esm_ratio, "seed": seed, "batch_size": batch,
           "learning_rate": a.lr, "weight_decay": a.weight_decay, "dropout": a.dropout if a.dropout is not None else ("default(0.3)" if a.variant == "esm_mlp" else MODEL_KWARGS["dropout"]), "max_epochs": a.max_epochs, "patience": a.patience,
           "best_epoch": best_epoch, "epochs_trained": len(hist), "best_validation_pr_auc": best,
           "selection": "validation average precision; test predicted once after restoring best weights",
           "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte), "train_positive": pos,
           "params_total": sum(p.numel() for p in model.parameters()),
           "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "load_sec": load_sec, "train_time_sec": train_sec, "inference_time_sec": infer_sec,
           "gpu_name": torch.cuda.get_device_name(device), "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
           "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "python_version": platform.python_version(),
           "storage_dtype": a.storage_dtype, "amp_dtype": str(AMP_DTYPE).replace("torch.", ""),
           "nan_loss_epochs": int(sum(1 for h in hist if not np.isfinite(h["train_loss"]))), "data_location": "gpu" if xtr.device.type == "cuda" else ("cpu-shm" if a.shm_cache else "cpu"), "split_dir": str(a.split_dir), "embeddings_root": str(a.embeddings_root),
           "script_sha256": sha256(__file__), "argv": sys.argv,
           **metrics(yt, pt), **{f"val_threshold_{k}": v for k, v in metrics(yt, pt, thr).items()}}
    pd.DataFrame({"y_true": yt, "y_prob": pt}).to_csv(out / "predictions" / f"{run_id}.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "history" / f"{run_id}.csv", index=False)
    tmp = result_path.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=2, default=float)); tmp.rename(result_path)
    print(f"[DONE] {run_id} ep={len(hist)} best_ep={best_epoch} valAP={best:.4f} testAP={res['pr_auc']:.4f} "
          f"train={train_sec:.0f}s", flush=True)
    del model, opt, best_state; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--task-type", required=True, choices=["BP", "PTM"])
    ap.add_argument("--split-dir", required=True)
    ap.add_argument("--embeddings-root", required=True)
    ap.add_argument("--esm-ratio", type=float, default=0.95)
    ap.add_argument("--variant", default="full", choices=VARIANTS)
    ap.add_argument("--loss", default="tim", choices=["tim", "bce", "weighted_bce", "focal", "tim_paper"])
    ap.add_argument("--lambda-value", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--tag", required=True, help="configuration label, e.g. full, wo_pe, loss_bce")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--storage-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--data-on-gpu", type=int, default=1)
    ap.add_argument("--shm-cache", default=None, help="directory in /dev/shm for shared float16 tensors")
    ap.add_argument("--verbose-epochs", type=int, default=0)
    ap.add_argument("--amp-dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW decoupled weight decay (0 = Adam, as before)")
    ap.add_argument("--dropout", type=float, default=None, help="override the PDeepPP/ablation dropout (default 0.15)")
    a = ap.parse_args()
    global AMP_DTYPE
    AMP_DTYPE = getattr(torch, a.amp_dtype)
    out = Path(a.out_root)
    for d in ("run_results", "predictions", "history"):
        (out / d).mkdir(parents=True, exist_ok=True)
    todo = [s for s in a.seeds if not (out / "run_results" / f"{a.task}__{a.tag}__seed{s}.json").is_file()]
    if not todo:
        print(f"[SKIP] {a.task} {a.tag} all seeds exist", flush=True); return
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dtype = torch.float16 if a.storage_dtype == "float16" else torch.float32
    t0 = time.perf_counter()
    if a.shm_cache:
        (xtr, ytr), (xva, yva), (xte, yte) = shm_tensors(a, dtype)
    else:
        tr, va, te = load_embeddings(a.split_dir, a.esm_ratio, embeddings_root=a.embeddings_root)
        (xtr, ytr), (xva, yva), (xte, yte) = (to_tensor(s, dtype) for s in (tr, va, te))
        del tr, va, te
    load_sec = time.perf_counter() - t0
    if a.data_on_gpu:
        xtr, ytr = xtr.to("cuda"), ytr.to("cuda")
    for seed in todo:
        train_seed(a, seed, ((xtr, ytr), (xva, yva), (xte, yte)), out, load_sec)


if __name__ == "__main__":
    main()
