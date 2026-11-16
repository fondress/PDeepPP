from __future__ import annotations
import argparse, json, multiprocessing as mp, queue
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from .analysis import aggregate
from .train_encoded import train_one


def _worker(gpu_id, jobs, results, statuses, config, output):
    cfg={**config,"device":f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"}
    if torch.cuda.is_available(): torch.cuda.set_device(gpu_id)
    while True:
        job=jobs.get()
        if job is None:return
        index,task,encoded,seed=job; statuses.put(f"[GPU {gpu_id}] START {index} task={task} seed={seed}")
        try: result=train_one(task,encoded,seed,cfg,output,gpu_id); results.put((index,result,None)); statuses.put(f"[GPU {gpu_id}] DONE {index} task={task} seed={seed}")
        except Exception as exc: results.put((index,None,repr(exc))); statuses.put(f"[GPU {gpu_id}] FAILED {index} task={task} seed={seed}: {exc!r}")


def _drain(statuses,path):
    while True:
        try: message=statuses.get_nowait()
        except queue.Empty:return
        with path.open("a",encoding="utf-8") as f:f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")


def start_workers(config,output):
    ctx=mp.get_context("spawn"); jobs,results,statuses=ctx.Queue(),ctx.Queue(),ctx.Queue(); visible=torch.cuda.device_count(); configured=[int(x) for x in config.get("gpus",range(visible))]; gpu_ids=[x for x in configured if x<visible] if visible else [0]
    workers=[ctx.Process(target=_worker,args=(g,jobs,results,statuses,config,output)) for g in gpu_ids]
    for w in workers:w.start()
    return jobs,results,statuses,workers


def collect(total,results,statuses,workers,status_path):
    received=0; failures=[]
    while received<total:
        _drain(statuses,status_path)
        try:index,result,error=results.get(timeout=30)
        except queue.Empty:
            crashed=[w.pid for w in workers if not w.is_alive() and w.exitcode not in (None,0)]
            if crashed:raise RuntimeError(f"GPU workers crashed: {crashed}")
            continue
        received+=1
        if error:failures.append((index,error))
    for w in workers:w.join()
    _drain(statuses,status_path)
    if failures:raise RuntimeError("Training failures: "+"; ".join(f"job {i}: {e}" for i,e in failures))


def run(tasks_path,config_path,splits_root,output):
    config=json.loads(config_path.read_text(encoding="utf-8")); tasks=pd.read_csv(tasks_path); output.mkdir(parents=True,exist_ok=True); status=output/"run_status.txt"; status.write_text(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] START\n",encoding="utf-8"); jobs,results,statuses,workers=start_workers(config,output); submitted=0
    for task in tasks.task.astype(str):
        encoded=splits_root/task/f"split_seed_{config['split_seed']}"/"encoded"
        if not (encoded/"manifest.json").exists():raise FileNotFoundError(encoded/"manifest.json")
        for seed in config["seeds"]:
            completed=output/"run_results"/f"{task}_seed{seed}.json"
            if completed.exists():statuses.put(f"[SKIP] task={task} seed={seed} completed result exists");continue
            submitted+=1;jobs.put((submitted,task,encoded,int(seed)))
    for _ in workers:jobs.put(None)
    collect(submitted,results,statuses,workers,status); aggregate(output)


def main():
    p=argparse.ArgumentParser();p.add_argument("--tasks",required=True);p.add_argument("--config",required=True);p.add_argument("--splits-root",default="outputs/splits");p.add_argument("--output-root",default="outputs");a=p.parse_args();run(Path(a.tasks).resolve(),Path(a.config).resolve(),Path(a.splits_root).resolve(),Path(a.output_root).resolve())
if __name__=="__main__":main()
