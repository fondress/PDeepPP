from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
AA="ACDEFGHIKLMNPQRSTVWY"; AA_TO_ID={a:i+1 for i,a in enumerate(AA)}; AA_TO_ID.update({"-":21,"X":0})
def encode(sequences,max_length=33):
    out=np.zeros((len(sequences),max_length),dtype=np.int16)
    for i,s in enumerate(sequences): out[i,:len(s)]=[AA_TO_ID.get(a,0) for a in str(s)[:max_length]]
    return out
def encode_task(task,config,splits_root):
    seed=int(config["split_seed"]); size=int(config.get("max_sequence_length",33)); chunk=int(config.get("encoding_chunk_size",1000)); root=splits_root/task/f"split_seed_{seed}"; out=root/"encoded";out.mkdir(parents=True,exist_ok=True)
    for old in out.glob("*.npy"):old.unlink()
    manifest={"task":task,"split_seed":seed,"max_sequence_length":size,"chunk_size":chunk,"format":"npy_shards","vocabulary":AA_TO_ID,"splits":{}}
    for name in ("train","val","test"):
        frame=pd.read_csv(root/f"{name}.csv"); entries=[]
        if "sample_id" not in frame.columns:
            if {"protein_id", "site_position"}.issubset(frame.columns):
                frame["sample_id"] = [f"{task}_{protein_id}_{int(position)}" for protein_id, position in zip(frame["protein_id"], frame["site_position"])]
            else:
                frame["sample_id"] = [f"{task}_{name}_{index}" for index in range(len(frame))]
        for j,start in enumerate(range(0,len(frame),chunk)):
            part=frame.iloc[start:start+chunk];prefix=f"{name}_{j:05d}";np.save(out/f"{prefix}_tokens.npy",encode(part.sequence.tolist(),size),allow_pickle=False);np.save(out/f"{prefix}_labels.npy",part.label.to_numpy(np.float32),allow_pickle=False);np.save(out/f"{prefix}_ids.npy",np.asarray(part.sample_id.astype(str),dtype=np.str_),allow_pickle=False);entries.append({"prefix":prefix,"start":start,"size":len(part)})
        manifest["splits"][name]={"size":len(frame),"shards":entries}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");return out
def encode_all(tasks_path,config_path,splits_root):
    config=json.loads(config_path.read_text(encoding="utf-8"));
    for task in pd.read_csv(tasks_path).task.astype(str):encode_task(task,config,splits_root)
def main():
    p=argparse.ArgumentParser();p.add_argument("--tasks",required=True);p.add_argument("--config",required=True);p.add_argument("--splits-root",default="outputs/splits");a=p.parse_args();encode_all(Path(a.tasks).resolve(),Path(a.config).resolve(),Path(a.splits_root).resolve())
if __name__=="__main__":main()
