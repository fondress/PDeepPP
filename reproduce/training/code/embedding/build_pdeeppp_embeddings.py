from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import esm
from code.utils import ensure_dir,read_json

MAX_SEQUENCE_LENGTH = 33
VALID_EMBEDDING_AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO-")

class EmbeddingPretrainedModel(nn.Module):
 def __init__(self,vocab_size,embedding_dim):super().__init__();self.embedding=nn.Embedding(vocab_size,embedding_dim);self.fc=nn.Linear(embedding_dim,1280)
 def forward(self,x):return self.fc(self.embedding(x))

def _prepare_fixed_length_sequence(value,max_length=MAX_SEQUENCE_LENGTH):
 if pd.isna(value):raise ValueError("sequence is missing")
 sequence="".join(str(value).strip().upper().split())
 sequence="".join(residue if residue in VALID_EMBEDDING_AA else "X" for residue in sequence)
 if not sequence or not sequence.replace("-",""):raise ValueError("sequence is empty")
 effective_length=len(sequence)
 return sequence[:max_length].ljust(max_length,"-"),effective_length

def embeddings_exist_for_split(split_dir,output_root,esm_ratios):
 split=Path(split_dir);root=Path(output_root)/split.parent.name/split.name
 for ratio in esm_ratios:
  directory=root/str(float(ratio))
  for name in ("train","val","test"):
   manifest=directory/name/"manifest.json"
   if not manifest.is_file():return False
   data=json.loads(manifest.read_text(encoding="utf-8"))
   if not data.get("shards") or not all((directory/name/x["file"]).is_file() for x in data["shards"]):return False
 return True
def _save_split(frame,name,ratio,out,model,alphabet,projection,vocab,device,batch_size,shard_size):
 target=out/name;target.mkdir(parents=True,exist_ok=True)
 for old in target.glob("*.npy"):old.unlink()
 if "sequence" not in frame.columns:raise ValueError(f"{name}: required column 'sequence' is missing")
 sequences=[];lengths=[]
 for row_index,value in frame["sequence"].items():
  try:sequence,length=_prepare_fixed_length_sequence(value)
  except ValueError as exc:raise ValueError(f"{name}: invalid sequence at CSV row {row_index + 2}: {exc}") from exc
  sequences.append(sequence);lengths.append(length)
 truncated=sum(length>MAX_SEQUENCE_LENGTH for length in lengths)
 if truncated:print(f"[EMBEDDING] {name}: truncated {truncated}/{len(sequences)} sequences to {MAX_SEQUENCE_LENGTH} residues",flush=True)
 converter=alphabet.get_batch_converter();shards=[]
 for shard_index,start in enumerate(range(0,len(sequences),shard_size)):
  subset=sequences[start:start+shard_size];path=target/f"{name}_{shard_index:05d}.npy";mmap=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(len(subset),MAX_SEQUENCE_LENGTH,1280));cursor=0
  for batch_start in range(0,len(subset),batch_size):
   seqs=subset[batch_start:batch_start+batch_size];data=[(str(start+batch_start+i),s.replace("-","X")) for i,s in enumerate(seqs)];_,_,tokens=converter(data);tokens=tokens.to(device)
   indices=torch.tensor([[vocab.get(a,0) for a in s] for s in seqs],dtype=torch.long,device=device)
   with torch.inference_mode():esm_features=model(tokens,repr_layers=[33])["representations"][33][:,1:MAX_SEQUENCE_LENGTH+1];learned=projection(indices);combined=ratio*esm_features+(1-ratio)*learned
   mmap[cursor:cursor+len(seqs)]=combined.cpu().numpy().astype(np.float32);cursor+=len(seqs)
  mmap.flush();del mmap;shards.append({"file":path.name,"start":start,"size":len(subset)})
 (target/"manifest.json").write_text(json.dumps({"split":name,"size":len(frame),"shard_size":shard_size,"shape":[len(frame),MAX_SEQUENCE_LENGTH,1280],"dtype":"float32","esm_ratio":ratio,"shards":shards},indent=2),encoding="utf-8")
def build_embeddings_for_split(split_dir,output_root,esm_ratios,batch_size=16,device_name="auto",esm_model_path="esm2_t33_650M_UR50D.pt",shard_size=1000,embedding_seed=42):
 random.seed(embedding_seed);np.random.seed(embedding_seed);torch.manual_seed(embedding_seed);split=Path(split_dir);device=torch.device("cuda" if device_name=="auto" and torch.cuda.is_available() else device_name)
 # The local ESM checkpoint is trusted, and newer PyTorch versions default to
 # weights_only=True, which cannot deserialize ESM's pickled model metadata.
 original_torch_load=torch.load
 def load_trusted_checkpoint(*args,**kwargs):
  kwargs["weights_only"]=False
  return original_torch_load(*args,**kwargs)
 torch.load=load_trusted_checkpoint
 try:
  model,alphabet=esm.pretrained.load_model_and_alphabet_local(str(Path(esm_model_path).expanduser().resolve()))
 finally:
  torch.load=original_torch_load
 model=model.to(device).eval();frames={n:pd.read_csv(split/f"{n}.csv") for n in ("train","val","test")};vocab={a:i for i,a in enumerate("XACDEFGHIKLMNPQRSTVWY-")};projection=EmbeddingPretrainedModel(len(vocab),128).to(device).eval()
 for ratio in esm_ratios:
  out=ensure_dir(Path(output_root)/split.parent.name/split.name/str(float(ratio)))
  for name,frame in frames.items():_save_split(frame,name,float(ratio),out,model,alphabet,projection,vocab,device,batch_size,shard_size)
def main():
 p=argparse.ArgumentParser();p.add_argument("--split-dir",required=True);p.add_argument("--output-root",required=True);p.add_argument("--config",required=True);p.add_argument("--device",default="auto");a=p.parse_args();c=read_json(a.config);build_embeddings_for_split(a.split_dir,a.output_root,c.get("esm_ratios",[1.]),int(c.get("esm_batch_size",16)),a.device,c["esm_model_path"],int(c.get("embedding_shard_size",1000)),int(c.get("embedding_seed",42)))
if __name__=="__main__":main()
