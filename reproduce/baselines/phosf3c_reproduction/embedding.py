from __future__ import annotations
import argparse,json
import multiprocessing as mp
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import esm

def complete(split,out):
 split=Path(split);out=Path(out)
 for name in ("train","val","test"):
  csv_path=split/f"{name}.csv";manifest_path=out/name/"manifest.json"
  if not csv_path.is_file() or not manifest_path.is_file():return False
  try:
   expected=len(pd.read_csv(csv_path));manifest=json.loads(manifest_path.read_text())
  except (OSError,ValueError,json.JSONDecodeError):return False
  if int(manifest.get("size",-1))!=expected:return False
  shards=manifest.get("shards",[])
  if sum(int(item.get("size",0)) for item in shards)!=expected:return False
  if not all((out/name/item["file"]).is_file() for item in shards):return False
 return True
def _load_local_esm(model_path):
 path=Path(model_path).expanduser().resolve()
 if not path.is_file():raise FileNotFoundError(f"Local ESM model not found: {path}")
 original_load=torch.load
 def trusted_load(*args,**kwargs):
  kwargs["weights_only"]=False
  return original_load(*args,**kwargs)
 torch.load=trusted_load
 try:
  return esm.pretrained.load_model_and_alphabet_local(str(path))
 finally:
  torch.load=original_load

def _embed_worker(gpu, split, out, model_path, batch_size, shard_size, names, errors):
 try:
  if torch.cuda.is_available():
   torch.cuda.set_device(gpu)
  embed(split, out, model_path, f"cuda:{gpu}" if torch.cuda.is_available() else "cpu", batch_size, shard_size, names)
 except Exception as exc:
  errors.put(f"GPU {gpu}: {exc!r}")

def embed(split,out,model_path,device="cuda:0",batch_size=16,shard_size=1000,names=None):
 split=Path(split);out=Path(out);device=torch.device(device if torch.cuda.is_available() else "cpu");model,alphabet=_load_local_esm(model_path);model=model.to(device).eval();converter=alphabet.get_batch_converter()
 for name in names or ("train","val","test"):
  print(f"[EMBED] {name}: loading data on {device}",flush=True)
  frame=pd.read_csv(split/f"{name}.csv");target=out/name;target.mkdir(parents=True,exist_ok=True);shards=[]
  for old in target.glob("*.npy"):old.unlink()
  for shard_idx,start in enumerate(range(0,len(frame),shard_size)):
   sequences=frame.sequence.astype(str).iloc[start:start+shard_size].tolist();path=target/f"{name}_{shard_idx:05d}.npy";array=np.lib.format.open_memmap(path,"w+",np.float32,shape=(len(sequences),31,1280));cursor=0
   for batch_start in range(0,len(sequences),batch_size):
    batch=sequences[batch_start:batch_start+batch_size];trimmed=[sequence[1:32].replace("-","X") for sequence in batch];_,_,tokens=converter([(str(i),s) for i,s in enumerate(trimmed)]);tokens=tokens.to(device)
    with torch.inference_mode():features=model(tokens,repr_layers=[33])["representations"][33][:,1:32]
    array[cursor:cursor+len(batch)]=features.cpu().numpy().astype(np.float32);cursor+=len(batch)
   array.flush();del array;shards.append({"file":path.name,"start":start,"size":len(sequences)})
  (target/"manifest.json").write_text(json.dumps({"split":name,"size":len(frame),"shape":[len(frame),31,1280],"shard_size":shard_size,"source_window":33,"phosf3c_window":31,"crop":"remove one residue from each side; center remains index 15","shards":shards},indent=2),encoding="utf-8")
  print(f"[EMBED] {name}: finished on {device}",flush=True)

def embed_all_gpus(split,out,model_path,gpus,batch_size=16,shard_size=10000):
 split=Path(split);out=Path(out);names=("train","val","test");frames={name:len(pd.read_csv(split/f"{name}.csv")) for name in names};assignments={gpu:[] for gpu in gpus}
 for name in names:
  assignments[gpus[names.index(name)%len(gpus)]].append(name)
 # For one task, split each CSV into independent contiguous ranges across GPUs.
 # This avoids concurrent writes while allowing every GPU to process data.
 context=mp.get_context("spawn");errors=context.Queue();processes=[]
 for gpu in gpus:
  if assignments[gpu]:
   process=context.Process(target=_embed_worker,args=(gpu,split,out,model_path,batch_size,shard_size,assignments[gpu],errors));process.start();processes.append(process)
 for process in processes:process.join()
 failures=[]
 while not errors.empty():failures.append(errors.get())
 if failures:raise RuntimeError("Embedding failures:\n"+"\n".join(failures))
 if any(process.exitcode for process in processes):raise RuntimeError("An embedding worker exited unexpectedly")

def main():
 p=argparse.ArgumentParser();p.add_argument("--split-dir",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--esm-model-path",required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=16);p.add_argument("--shard-size",type=int,default=1000);a=p.parse_args();embed(a.split_dir,a.output_dir,a.esm_model_path,a.device,a.batch_size,a.shard_size)
if __name__=="__main__":main()
