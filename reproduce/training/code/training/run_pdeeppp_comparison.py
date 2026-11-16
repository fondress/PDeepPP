from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
from pathlib import Path

import pandas as pd
import torch

from code.embedding.build_pdeeppp_embeddings import build_embeddings_for_split, embeddings_exist_for_split
from code.embedding.build_bp_unidl_aligned_embeddings import (
    build_bp_embeddings_unidl_aligned,
    bp_unidl_embeddings_exist,
)
from code.training.train_one_run import train_one_run
from code.utils import ensure_dir, read_json


def _gpu_worker(gpu_id: int, job_queue, result_queue) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    while True:
        job = job_queue.get()
        if job is None:
            return

        job_index = job.pop("job_index")
        job["device_name"] = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        try:
            print(
                f"[GPU {gpu_id}] [TRAIN {job_index}] start "
                f"task={job['task']} split={Path(job['split_dir']).name} "
                f"model={job['model_name']} seed={job['seed']} "
                f"esm={job['esm_ratio']} lambda={job['lambda_value']}",
                flush=True,
            )
            result = train_one_run(**job)
            result_queue.put({"job_index": job_index, "gpu_id": gpu_id, "result": result})
            print(f"[GPU {gpu_id}] [TRAIN {job_index}] done", flush=True)
        except Exception as exc:
            result_queue.put({"job_index": job_index, "gpu_id": gpu_id, "error": repr(exc)})
            print(f"[GPU {gpu_id}] [TRAIN {job_index}] failed: {exc!r}", flush=True)


def _build_embeddings(jobs: list[dict], config: dict) -> None:
    pending: dict[tuple[str, str], set[float]] = {}
    for job in jobs:
        if job["model_name"] != "pdeeppp":
            continue
        key = (str(job["split_dir"]), str(job["embeddings_root"]))
        pending.setdefault(key, set()).add(float(job["esm_ratio"]))

    total = len(pending)
    for idx, ((split_dir, embeddings_root), ratios) in enumerate(pending.items(), start=1):
        sorted_ratios = sorted(ratios)
        pipeline = str(config.get("bp_embedding_pipeline", "unidl_aligned")).lower()
        exists = (
            bp_unidl_embeddings_exist(split_dir, embeddings_root, sorted_ratios)
            if pipeline == "unidl_aligned"
            else embeddings_exist_for_split(split_dir, embeddings_root, sorted_ratios)
        )
        if exists:
            print(
                f"[EMBEDDING {idx}/{total}] found existing embeddings; skip generation "
                f"split={split_dir} ratios={sorted_ratios} pipeline={pipeline}",
                flush=True,
            )
            continue
        print(
            f"[EMBEDDING {idx}/{total}] start split={split_dir} "
            f"ratios={sorted_ratios} pipeline={pipeline}",
            flush=True,
        )
        builder_kwargs = {
            "split_dir": split_dir,
            "output_root": embeddings_root,
            "esm_ratios": sorted_ratios,
            "batch_size": int(config.get("esm_batch_size", 16)),
            "device_name": str(config.get("embedding_device", "cuda:0")),
            "esm_model_path": config.get("esm_model_path", "esm2_t33_650M_UR50D.pt"),
        }
        if pipeline == "unidl_aligned":
            build_bp_embeddings_unidl_aligned(
                **builder_kwargs,
                shard_size=int(config.get("embedding_shard_size", 1000)),
                embedding_seed=int(config.get("embedding_seed", 42)),
            )
        elif pipeline == "residue_fixed_33":
            build_embeddings_for_split(**builder_kwargs)
        else:
            raise ValueError(f"Unsupported bp_embedding_pipeline: {pipeline}")
        print(f"[EMBEDDING {idx}/{total}] done", flush=True)


def _parse_seed_list(seed_arg: str | None, config_seeds) -> list[int]:
    if seed_arg is not None and str(seed_arg).strip() != "":
        return [int(x.strip()) for x in str(seed_arg).split(",") if str(x).strip() != ""]
    return [int(s) for s in (config_seeds if config_seeds is not None else [42])]


def _build_jobs(
    tasks: pd.DataFrame,
    config: dict,
    splits_root: str | Path,
    checkpoint_dir: Path,
    seeds: list[int] | None = None,
) -> list[dict]:
    jobs = []
    models = list(config.get("models", ["pdeeppp"]))
    seeds = seeds if seeds is not None else [int(s) for s in config.get("seeds", [42])]
    split_seeds = list(config.get("split_seeds", [config.get("split_seed", 42)]))
    pdeeppp_kwargs = config.get("pdeeppp_model_kwargs", {})
    if not isinstance(pdeeppp_kwargs, dict):
        raise TypeError("pdeeppp_model_kwargs must be a JSON object")

    for task_index, (_, row) in enumerate(tasks.iterrows(), start=1):
        task = str(row["task"]).strip()
        task_type = str(row["task_type"]).strip().upper()
        print(f"[TASK {task_index}/{len(tasks)}] task={task} type={task_type}", flush=True)
        for split_seed in split_seeds:
            split_dir = Path(splits_root) / task / f"split_seed_{split_seed}"
            if not split_dir.exists():
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
            for model_name in models:
                esm_ratios = list(config.get("esm_ratios", [1.0])) if model_name == "pdeeppp" else [1.0]
                lambda_values = list(config.get("lambda_values", [0.95])) if model_name == "pdeeppp" else [float(config.get("control_lambda", 1.0))]   # R7 11-04: lambda of the ESM-2 controls under the Section 2.2 loss
                for seed in seeds:
                    for esm_ratio in esm_ratios:
                        for lambda_value in lambda_values:
                            jobs.append({
                                "task": task,
                                "task_type": task_type,
                                "split_dir": split_dir,
                                "model_name": model_name,
                                "esm_ratio": float(esm_ratio),
                                "lambda_value": float(lambda_value),
                                "seed": int(seed),
                                "output_dir": checkpoint_dir,
                                "batch_size": int(config.get("batch_size", 128)),
                                "learning_rate": float(config.get("learning_rate", 1e-3)),
                                "max_epochs": int(config.get("max_epochs", 100)),
                                "patience": int(config.get("early_stopping_patience", 20)),
                                "selection_metric": str(config.get("selection_metric", "pr_auc")),
                                "num_workers": int(config.get("num_workers", 0)),
                                "loss_name": str(config.get("loss_name", "tim")),
                                "model_kwargs": pdeeppp_kwargs if model_name == "pdeeppp" else None,
                                "embeddings_root": config.get("embeddings_root"),
                            })
    return jobs


def _run_fixed_gpu_workers(jobs: list[dict], gpu_ids: list[int]) -> list[dict]:
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    workers = [context.Process(target=_gpu_worker, args=(gpu_id, job_queue, result_queue)) for gpu_id in gpu_ids]

    for worker in workers:
        worker.start()
    for job_index, job in enumerate(jobs, start=1):
        job_queue.put({"job_index": job_index, **job})
    for _ in workers:
        job_queue.put(None)

    results = []
    failures = []
    received = 0
    while received < len(jobs):
        try:
            message = result_queue.get(timeout=30)
        except queue.Empty:
            failed_workers = [worker.pid for worker in workers if not worker.is_alive() and worker.exitcode not in (None, 0)]
            if failed_workers:
                raise RuntimeError(f"GPU worker crashed: pids={failed_workers}")
            continue
        received += 1
        if "error" in message:
            failures.append(message)
        else:
            results.append(message["result"])

    for worker in workers:
        worker.join()
    if failures:
        details = "\n".join(f"job={item['job_index']} gpu={item['gpu_id']}: {item['error']}" for item in failures)
        raise RuntimeError(f"{len(failures)} training job(s) failed:\n{details}")
    return results


def run_pdeeppp_comparison(
    tasks_csv: str | Path,
    config_path: str | Path,
    splits_root: str | Path,
    output_root: str | Path,
    seeds: list[int] | None = None,
) -> None:
    config = read_json(config_path)
    for key in ("embeddings_root", "esm_model_path"):
        if key in config and not str(config[key] or "").strip():
            raise ValueError(f"Set {key} in {config_path}")
    tasks = pd.read_csv(tasks_csv)
    seeds = seeds if seeds is not None else _parse_seed_list(None, config.get("seeds", [42]))
    seed_tag = "_".join(str(s) for s in seeds)
    output_root = ensure_dir(output_root)
    checkpoint_dir = ensure_dir(output_root / "checkpoints" / f"seed_{seed_tag}")
    metrics_dir = ensure_dir(output_root / "metrics")
    hyper_dir = ensure_dir(output_root / "hyperparameter_search")

    jobs = _build_jobs(tasks, config, splits_root, checkpoint_dir, seeds=seeds)
    if not jobs:
        raise RuntimeError("No training jobs were created")
    print(f"[INFO] seeds={seeds} | total training jobs: {len(jobs)}", flush=True)
    _build_embeddings(jobs, config)

    configured_gpus = [int(gpu) for gpu in config.get("gpus", list(range(8)))]
    visible_gpus = torch.cuda.device_count()
    if visible_gpus:
        gpu_ids = [gpu for gpu in configured_gpus if gpu < visible_gpus]
        if not gpu_ids:
            raise RuntimeError(f"No configured GPU is visible. configured={configured_gpus}, visible={visible_gpus}")
    else:
        gpu_ids = [0]
    print(f"[INFO] fixed workers: {gpu_ids}", flush=True)

    all_trials = _run_fixed_gpu_workers(jobs, gpu_ids)
    all_trials_df = pd.DataFrame(all_trials)
    best_configs = []
    raw_results = []
    for (task, task_type, model_name), group in all_trials_df.groupby(["task", "task_type", "model"]):
        best_trial = group.sort_values("best_validation_score", ascending=False).iloc[0].to_dict()
        raw_results.append(best_trial)
        best_configs.append({
            "task": task,
            "task_type": task_type,
            "model": model_name,
            "selected_seed": best_trial["seed"],
            "selected_esm_ratio": best_trial["esm_ratio"],
            "selected_lambda": best_trial["lambda_value"],
            "selection_metric": str(config.get("selection_metric", "pr_auc")),
            "best_validation_score": best_trial["best_validation_score"],
        })

    all_path = hyper_dir / f"all_trials_seed{seed_tag}.csv"
    best_path = hyper_dir / f"best_config_by_task_seed{seed_tag}.csv"
    raw_path = metrics_dir / f"raw_results_seed{seed_tag}.csv"
    all_trials_df.to_csv(all_path, index=False)
    pd.DataFrame(best_configs).to_csv(best_path, index=False)
    pd.DataFrame(raw_results).to_csv(raw_path, index=False)
    print(f"[SAVE] {all_path}", flush=True)
    print(f"[SAVE] {best_path}", flush=True)
    print(f"[SAVE] {raw_path}", flush=True)
    print(f"[INFO] completed {len(all_trials)} jobs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PDeepPP comparison experiments with fixed GPU workers.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits-root", default="results/splits")
    parser.add_argument("--output-root", default="results/comparison")
    parser.add_argument(
        "--seed",
        default=None,
        help="One seed or comma-separated seeds, e.g. 42 or 42,43. Overrides config seeds.",
    )
    args = parser.parse_args()
    config = read_json(args.config)
    seeds = _parse_seed_list(args.seed, config.get("seeds", [42]))
    run_pdeeppp_comparison(args.tasks, args.config, args.splits_root, args.output_root, seeds=seeds)

if __name__ == "__main__":
    main()
