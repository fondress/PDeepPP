from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from ..data.auto_prepare_tasks import auto_prepare_bp_tasks
from ..data.clean_bp_data import BpColumns, clean_bp_dataframe
from ..data.check_leakage import build_bp_leakage_report
from ..data.split_bp_data import make_bp_splits
from ..data.load_bp_tasks import load_tasks, load_task_table
from ..embedding.extract_esm import extract_with_esm2, ESM2UnavailableError
from ..embedding.build_unidl_features import save_feature_arrays
from ..embedding.check_esm import esm_status
from ..utils import ensure_dir, write_json, read_json
from .train_one_task import train_one_task


def _load_or_prepare_tasks(tasks_csv: str | Path | None, bp_root: str | Path | None, output_root: str | Path, base_dir: str | Path | None):
    output_root = Path(output_root)
    if tasks_csv is not None and Path(tasks_csv).exists():
        return Path(tasks_csv)
    if bp_root is None:
        raise FileNotFoundError("tasks_csv not found and bp_root was not provided to generate it")
    auto_prepare_bp_tasks(bp_root=bp_root, output_root=output_root)
    generated = output_root / "unidl_tasks.csv"
    if not generated.exists():
        raise FileNotFoundError(f"Failed to generate task table at {generated}")
    return generated


def _make_bp_task_splits(tasks_csv: str | Path, base_dir: str | Path | None, split_seed: int, test_size: float, val_size: float, output_root: str | Path):
    tasks = load_tasks(tasks_csv)
    meta_rows = []
    for idx, (_, row) in enumerate(tasks.iterrows(), start=1):
        task = str(row["task"])
        print(f"[SPLIT {idx}/{len(tasks)}] preparing {task}", flush=True)
        raw = load_task_table(row, base_dir)
        columns = BpColumns(
            sequence_col=str(row.get("sequence_col", "sequence")),
            label_col=str(row.get("label_col", "label")),
            split_group_col=None if pd.isna(row.get("split_group_col", "")) else str(row.get("split_group_col")),
        )
        clean_df, clean_report = clean_bp_dataframe(raw, columns, task)
        split_dir = Path(output_root) / task / f"split_seed_{split_seed}"
        split_dir.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(split_dir / "cleaned.csv", index=False)
        metadata = make_bp_splits(clean_df, task, split_dir, split_seed, test_size, val_size)
        train = pd.read_csv(split_dir / "train.csv")
        val = pd.read_csv(split_dir / "val.csv")
        test = pd.read_csv(split_dir / "test.csv")
        leakage = build_bp_leakage_report(train, val, test)
        write_json({"cleaning": clean_report, "split": metadata, "leakage": leakage}, split_dir / "metadata.json")
        meta_rows.append({"task": task, **clean_report, **metadata, **leakage})
        print(f"[SPLIT {idx}/{len(tasks)}] done {task}", flush=True)
    pd.DataFrame(meta_rows).to_csv(Path(output_root) / "all_split_metadata.csv", index=False)


def _build_embeddings_from_split(split_dir: Path, esm_model_path: str | Path | None, esm_batch_size: int, esm_max_tokens: int, esm_device: str):
    train = pd.read_csv(split_dir / "train.csv")
    val = pd.read_csv(split_dir / "val.csv")
    test = pd.read_csv(split_dir / "test.csv")
    try:
        train_x = extract_with_esm2(train["sequence"].tolist(), model_path=esm_model_path, batch_size=esm_batch_size, max_batch_tokens=esm_max_tokens, device_name=esm_device)
        val_x = extract_with_esm2(val["sequence"].tolist(), model_path=esm_model_path, batch_size=esm_batch_size, max_batch_tokens=esm_max_tokens, device_name=esm_device)
        test_x = extract_with_esm2(test["sequence"].tolist(), model_path=esm_model_path, batch_size=esm_batch_size, max_batch_tokens=esm_max_tokens, device_name=esm_device)
    except ESM2UnavailableError as exc:
        raise RuntimeError(f"Cannot build embeddings for {split_dir}: {exc}")
    save_feature_arrays(train_x, val_x, test_x, split_dir / "embeddings" / "default")


def _load_pr_auc_targets(path: str | Path | None) -> Dict[str, float]:
    """Load a per-task validation PR AUC target table (columns: task, pr_auc_target)."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        print(f"[WARN] pr_auc_targets file not found: {path}; continuing without target stopping", flush=True)
        return {}
    frame = pd.read_csv(path)
    if "task" not in frame.columns or "pr_auc_target" not in frame.columns:
        raise ValueError(f"pr_auc_targets file must contain columns 'task' and 'pr_auc_target': {path}")
    targets = {}
    for _, row in frame.iterrows():
        task = str(row["task"])
        value = row["pr_auc_target"]
        if pd.notna(value):
            targets[task] = float(value)
    print(f"[INFO] loaded {len(targets)} per-task PR AUC targets from {path}", flush=True)
    return targets


def _gpu_worker(gpu_id: int, job_queue, result_queue) -> None:
    """One long-lived process bound to a single GPU; pulls jobs from a shared queue."""
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    print(f"[GPU {gpu_id}] initializing CUDA pid={os.getpid()}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        torch.empty(1, device=f"cuda:{gpu_id}")
        print(
            f"[GPU {gpu_id}] CUDA ready device={torch.cuda.get_device_name(gpu_id)}",
            flush=True,
        )
    else:
        print(f"[GPU {gpu_id}] CUDA unavailable; using CPU", flush=True)

    while True:
        job = job_queue.get()
        if job is None:
            return

        job_index = job.pop("job_index")
        device_name = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        try:
            print(
                f"[GPU {gpu_id}] [TRAIN {job_index}] start "
                f"task={job['task']} seed={job['seed']}",
                flush=True,
            )
            result = train_one_task(
                task=job["task"],
                split_dir=job["split_dir"],
                seed=int(job["seed"]),
                output_dir=job["output_dir"],
                batch_size=int(job["batch_size"]),
                lr=float(job["lr"]),
                max_epochs=int(job["max_epochs"]),
                patience=int(job["patience"]),
                device_name=device_name,
                loss_name=str(job["loss_name"]),
                lambda_value=float(job["lambda_value"]),
                pr_auc_target=job.get("pr_auc_target"),
            )
            result["gpu_id"] = gpu_id
            result_queue.put({"job_index": job_index, "gpu_id": gpu_id, "result": result})
            print(f"[GPU {gpu_id}] [TRAIN {job_index}] done task={job['task']} seed={job['seed']}", flush=True)
        except Exception as exc:
            error_traceback = traceback.format_exc()
            result_queue.put({
                "job_index": job_index,
                "gpu_id": gpu_id,
                "error": repr(exc),
                "traceback": error_traceback,
                "task": job.get("task"),
                "seed": job.get("seed"),
            })
            print(
                f"[GPU {gpu_id}] [TRAIN {job_index}] failed: {exc!r}\n{error_traceback}",
                flush=True,
            )


def _run_fixed_gpu_workers(jobs: List[Dict], gpu_ids: List[int], status_interval: int = 30) -> List[Dict]:
    """Start one worker per GPU; each worker stays on that card and consumes the job queue."""
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    workers = [
        context.Process(target=_gpu_worker, args=(gpu_id, job_queue, result_queue))
        for gpu_id in gpu_ids
    ]

    for worker, gpu_id in zip(workers, gpu_ids):
        worker.start()
        print(f"[WORKER] started gpu={gpu_id} pid={worker.pid}", flush=True)

    for job_index, job in enumerate(jobs, start=1):
        job_queue.put({"job_index": job_index, **job})
    for _ in workers:
        job_queue.put(None)

    results: List[Dict] = []
    failures: List[Dict] = []
    received = 0
    wait_started = time.monotonic()
    while received < len(jobs):
        try:
            message = result_queue.get(timeout=status_interval)
        except queue.Empty:
            worker_states = [
                f"pid={worker.pid},alive={worker.is_alive()},exitcode={worker.exitcode}"
                for worker in workers
            ]
            failed_workers = [
                worker.pid
                for worker in workers
                if not worker.is_alive() and worker.exitcode not in (None, 0)
            ]
            if failed_workers:
                raise RuntimeError(f"GPU worker crashed: pids={failed_workers}; states={worker_states}")
            if not any(worker.is_alive() for worker in workers):
                raise RuntimeError(
                    f"All GPU workers exited before returning every result "
                    f"({received}/{len(jobs)} received); states={worker_states}"
                )
            elapsed = int(time.monotonic() - wait_started)
            print(
                f"[WAIT] no completed job yet; received={received}/{len(jobs)} "
                f"elapsed={elapsed}s | {'; '.join(worker_states)}",
                flush=True,
            )
            continue
        received += 1
        if "error" in message:
            failures.append(message)
            print(
                f"[FAILED] job={message.get('job_index')} task={message.get('task')} "
                f"seed={message.get('seed')} gpu={message.get('gpu_id')} err={message.get('error')}",
                flush=True,
            )
        else:
            results.append(message["result"])
            print(f"[RESULT] collected {len(results)}/{len(jobs)}", flush=True)

    for worker in workers:
        worker.join()

    if failures:
        details = "\n".join(
            f"job={item.get('job_index')} task={item.get('task')} seed={item.get('seed')} "
            f"gpu={item.get('gpu_id')}: {item.get('error')}\n{item.get('traceback', '')}"
            for item in failures
        )
        raise RuntimeError(f"{len(failures)} training job(s) failed:\n{details}")
    return results


def _resolve_gpu_ids(config: dict) -> List[int]:
    configured_gpus = [int(gpu) for gpu in config.get("gpus", list(range(8)))]
    visible_gpus = torch.cuda.device_count()
    if visible_gpus:
        gpu_ids = [gpu for gpu in configured_gpus if gpu < visible_gpus]
        if not gpu_ids:
            raise RuntimeError(
                f"No configured GPU is visible. configured={configured_gpus}, visible={visible_gpus}"
            )
    else:
        gpu_ids = [0]
        print("[WARN] CUDA not available; falling back to a single CPU worker (device=cpu)", flush=True)
    return gpu_ids


def run_repeated_seeds(
    tasks_csv: str | Path | None,
    config_path: str | Path,
    splits_root: str | Path,
    output_root: str | Path,
    base_dir: str | Path | None = None,
    bp_root: str | Path | None = None,
    selected_tasks: list[str] | None = None,
    use_existing_splits: bool = False,
    pr_auc_targets: str | Path | None = None,
):
    config = read_json(config_path)
    output_root = ensure_dir(output_root)
    ensure_dir(output_root / "checkpoints")
    ensure_dir(output_root / "predictions")
    ensure_dir(output_root / "metrics")

    esm_model_path = config.get("esm_model_path")
    if esm_model_path is not None and not str(esm_model_path).strip():
        raise ValueError(f"Set esm_model_path in {config_path}")
    if esm_model_path is not None:
        esm_model_path = Path(config_path).parent / esm_model_path if not Path(esm_model_path).is_absolute() else Path(esm_model_path)

    print("ESM status:", esm_status(esm_model_path), flush=True)

    tasks_csv = _load_or_prepare_tasks(tasks_csv, bp_root, output_root, base_dir)
    tasks = load_tasks(tasks_csv)
    if selected_tasks:
        requested = set(selected_tasks)
        available = set(tasks["task"].astype(str))
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Unknown task(s): {missing}")
        tasks = tasks[tasks["task"].astype(str).isin(requested)].copy()

    split_seed = int(config.get("split_seed", 42))
    test_size = float(config.get("test_size", 0.2))
    val_size = float(config.get("val_size", 0.1))
    if use_existing_splits:
        for task in tasks["task"].astype(str):
            split_dir = Path(splits_root) / task / f"split_seed_{split_seed}"
            missing_files = [name for name in ("train.csv", "val.csv", "test.csv") if not (split_dir / name).is_file()]
            if missing_files:
                raise FileNotFoundError(f"Existing split for {task} is incomplete at {split_dir}: {missing_files}")
    else:
        selected_manifest = Path(output_root) / "selected_tasks.csv"
        tasks.to_csv(selected_manifest, index=False)
        _make_bp_task_splits(selected_manifest, base_dir, split_seed, test_size, val_size, splits_root)

    esm_batch_size = int(config.get("esm_batch_size", 2))
    esm_max_tokens = int(config.get("esm_max_tokens", 512))
    esm_device = str(config.get("esm_device", "cuda:0"))
    for idx, (_, row) in enumerate(tasks.iterrows(), start=1):
        task = str(row["task"])
        split_dir = Path(splits_root) / task / f"split_seed_{split_seed}"
        print(f"[EMBEDDING {idx}/{len(tasks)}] checking {task}", flush=True)
        if not (split_dir / "embeddings" / "default" / "train.npy").exists():
            print(f"[EMBEDDING {idx}/{len(tasks)}] build {task}", flush=True)
            _build_embeddings_from_split(split_dir, esm_model_path, esm_batch_size, esm_max_tokens, esm_device)
        print(f"[EMBEDDING {idx}/{len(tasks)}] done {task}", flush=True)

    seeds = list(config.get("seeds", [42]))
    target_table = _load_pr_auc_targets(pr_auc_targets)
    jobs: List[Dict] = []
    for _, row in tasks.iterrows():
        task = str(row["task"])
        split_dir = Path(splits_root) / task / f"split_seed_{split_seed}"
        task_target = target_table.get(task)
        for seed in seeds:
            jobs.append({
                "task": task,
                "seed": seed,
                "split_dir": split_dir,
                "output_dir": output_root / "checkpoints",
                "batch_size": int(config.get("batch_size", 64)),
                "lr": float(config.get("learning_rate", 1e-3)),
                "max_epochs": int(config.get("max_epochs", 100)),
                "patience": int(config.get("early_stopping_patience", 20)),
                "loss_name": str(config.get("loss_name", "tim")),
                "lambda_value": float(config.get("lambda_value", 1.0)),
                "pr_auc_target": task_target,
            })

    if not jobs:
        raise RuntimeError("No training jobs were created")

    gpu_ids = _resolve_gpu_ids(config)
    status_interval = max(5, int(config.get("worker_status_interval_sec", 30)))
    print(
        f"[INFO] total jobs: {len(jobs)} | fixed GPU workers: {gpu_ids} "
        f"| status_interval={status_interval}s",
        flush=True,
    )
    final_results = _run_fixed_gpu_workers(jobs, gpu_ids, status_interval=status_interval)

    final_df = pd.DataFrame(final_results)
    best_configs = []
    for task, task_df in final_df.groupby("task"):
        best = task_df.sort_values("best_validation_pr_auc", ascending=False).iloc[0]
        best_configs.append({
            "task": task,
            "selected_seed": best["seed"],
            "selected_metric": "pr_auc",
            "best_validation_pr_auc": best["best_validation_pr_auc"],
            "gpu_id": best.get("gpu_id", None),
        })

    final_df.to_csv(output_root / "metrics" / "raw_results.csv", index=False)
    pd.DataFrame(best_configs).to_csv(output_root / "metrics" / "best_config_by_task.csv", index=False)
    final_df.to_csv(output_root / "metrics" / "all_trials.csv", index=False)
    print(f"[INFO] completed {len(final_results)} jobs", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run UniDL4BioPep BP reproduction with repeated seeds (fixed multi-GPU workers).")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits-root", default="outputs/splits")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--bp-root", default=None, help="BP root directory used to auto-generate tasks.csv if needed.")
    parser.add_argument("--task", action="append", dest="selected_tasks", help="Run only this task; repeat to select multiple tasks.")
    parser.add_argument("--use-existing-splits", action="store_true", help="Use pre-generated cluster splits without replacing them.")
    parser.add_argument("--pr-auc-targets", default=None, help="CSV with columns 'task' and 'pr_auc_target'; training stops once validation PR AUC reaches the per-task target.")
    args = parser.parse_args()
    run_repeated_seeds(
        args.tasks, args.config, args.splits_root, args.output_root,
        args.base_dir, args.bp_root, args.selected_tasks,
        args.use_existing_splits, args.pr_auc_targets,
    )


if __name__ == "__main__":
    main()
