from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from phosf3c_reproduction.embedding import complete, embed, embed_all_gpus
from phosf3c_reproduction.train import train_one


def maximize_gpu_performance(gpus):
    """Best-effort: enable persistence mode (safe, non-blocking)."""
    try:
        subprocess.run(["nvidia-smi", "-pm", "1"], capture_output=True, timeout=5)
    except Exception:
        pass


def worker(gpu, jobs, results, config, output):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    while True:
        job = jobs.get()
        if job is None:
            return
        try:
            results.put((train_one(**job, config=config, output=output, gpu=gpu), None))
        except torch.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.put((None, f"task={job.get('task')} seed={job.get('seed')} gpu={gpu}: {exc!r}"))
        except Exception as exc:
            results.put((None, f"task={job.get('task')} seed={job.get('seed')} gpu={gpu}: {exc!r}"))


def aggregate(output):
    files = sorted((output / "run_results").glob("*.json"))
    raw = pd.DataFrame([json.loads(path.read_text()) for path in files])
    metrics = output / "metrics"
    metrics.mkdir(exist_ok=True)
    raw.to_csv(metrics / "raw_results.csv", index=False)
    raw.to_csv(metrics / "all_trials.csv", index=False)
    if raw.empty:
        return
    numeric = [column for column in raw.select_dtypes("number") if column not in ("seed", "gpu_id")]
    group = raw.groupby(["task", "model"])[numeric].agg(["mean", "std"])
    group.columns = [f"{first}_{second}" for first, second in group.columns]
    group.reset_index().to_csv(metrics / "mean_sd_by_task.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task", action="append", dest="selected")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--force-rebuild-embedding", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    for key in ("embeddings_root", "esm_model_path"):
        if not str(config.get(key) or "").strip():
            raise ValueError(f"Set {key} in {config_path}")
    config_dir = config_path.parent
    tasks = pd.read_csv(args.tasks)
    if args.selected:
        configured_tasks = set(args.selected)
    else:
        configured_tasks = set(config.get("tasks", tasks["task"].astype(str)))
    tasks = tasks[tasks.task.astype(str).isin(configured_tasks)]
    if tasks.empty:
        raise ValueError("No PhosF3C tasks were selected")

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    embedding_root = Path(config["embeddings_root"])
    embedding_root = embedding_root if embedding_root.is_absolute() else (config_dir / embedding_root).resolve()
    config["embeddings_root"] = str(embedding_root)
    esm_model_path = Path(config["esm_model_path"])
    esm_model_path = esm_model_path if esm_model_path.is_absolute() else (config_dir / esm_model_path).resolve()
    config["esm_model_path"] = str(esm_model_path)

    # Step 1: Embeddings (sequential by task, parallel across GPUs)
    visible = torch.cuda.device_count()
    gpus = [gpu for gpu in config["gpus"] if gpu < visible] if visible else [0]
    print(f"[RUN] Step 1: checking/creating embeddings for {len(tasks)} task(s) on GPUs {gpus}...", flush=True)
    for task in tasks.task.astype(str):
        split = Path(args.splits_root) / task / f"split_seed_{config['split_seed']}"
        destination = embedding_root / task / f"split_seed_{config['split_seed']}"
        print(f"  [{task}] checking embeddings at {destination} ...", flush=True)
        if not args.skip_embedding:
            embedding_is_complete = complete(split, destination)
            if args.force_rebuild_embedding or not embedding_is_complete:
                if args.force_rebuild_embedding and embedding_is_complete:
                    print(f"  [{task}] forcing embedding rebuild...", flush=True)
                else:
                    print(f"  [{task}] embeddings not found, generating now...", flush=True)
                embed_all_gpus(
                    split,
                    destination,
                    config["esm_model_path"],
                    gpus,
                    int(config.get("embedding_batch_size", 16)),
                    int(config.get("embedding_shard_size", 10000)),
                )
                print(f"  [{task}] embeddings done.", flush=True)
            else:
                print(f"  [{task}] embeddings already exist, skipping.", flush=True)
        elif not complete(split, destination):
            raise FileNotFoundError(f"Embeddings not found for {task}: {destination}")
        else:
            print(f"  [{task}] embeddings already exist, skipping.", flush=True)

    # Step 2: GPU power optimization
    visible = torch.cuda.device_count()
    gpus = [gpu for gpu in config["gpus"] if gpu < visible] if visible else [0]
    print(f"[RUN] Step 2: {len(tasks)} task(s), {len(gpus)} GPU(s) available ({gpus})", flush=True)
    print(f"[RUN] Enabling GPU persistence mode (nvidia-smi -pm 1)...", flush=True)
    t0 = time.time()
    maximize_gpu_performance(gpus)
    print(f"[RUN] GPU setup done in {time.time()-t0:.1f}s", flush=True)

    # Step 3: Training (sequential by task, parallel across GPUs)
    context = mp.get_context("spawn")
    for task in tasks.task.astype(str):
        split = Path(args.splits_root) / task / f"split_seed_{config['split_seed']}"
        embeddings = embedding_root / task / f"split_seed_{config['split_seed']}"
        task_dir = output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        for folder in ("best", "checkpoints", "history", "logs", "predictions", "run_results", "summary"):
            (task_dir / folder).mkdir(exist_ok=True)

        seeds_to_run = [
            seed for seed in config["seeds"]
            if not (output / "run_results" / f"{task}_seed{seed}.json").exists()
        ]
        if not seeds_to_run:
            print(f"[RUN] [{task}] all seeds already completed, skipping", flush=True)
            continue

        print(f"[RUN] [{task}] training {len(seeds_to_run)} seed(s) on {len(gpus)} GPU(s): {seeds_to_run}", flush=True)
        jobs = context.Queue()
        results = context.Queue()
        workers = [context.Process(target=worker, args=(gpu, jobs, results, config, output)) for gpu in gpus]
        for process in workers:
            process.start()

        submitted = 0
        for seed in seeds_to_run:
            jobs.put({"task": task, "split": split, "embeddings": embeddings, "seed": int(seed)})
            submitted += 1
        for _ in workers:
            jobs.put(None)

        failures = []
        for _ in range(submitted):
            _, error = results.get()
            if error:
                failures.append(error)
        for process in workers:
            process.join()
        if failures:
            raise RuntimeError(f"Training failures for task {task}:\n" + "\n".join(failures))
        print(f"[RUN] [{task}] completed ({submitted} runs on {len(gpus)} GPUs)", flush=True)

    aggregate(output)
    print(f"[RUN] All {len(tasks)} tasks completed", flush=True)


if __name__ == "__main__":
    main()