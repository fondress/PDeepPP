from __future__ import annotations
import bisect,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
class ShardedDataset(Dataset):
 def __init__(self,split_dir,embedding_dir,name):
  self.meta=pd.read_csv(Path(split_dir)/f"{name}.csv").reset_index(drop=True)
  if "sample_id" not in self.meta.columns:
   if {"protein_id","site_position"}.issubset(self.meta.columns):self.meta["sample_id"]=[f"{protein}_{int(position)}" for protein,position in zip(self.meta.protein_id,self.meta.site_position)]
   else:self.meta["sample_id"]=[f"{name}_{index}" for index in range(len(self.meta))]
  manifest=json.loads((Path(embedding_dir)/name/"manifest.json").read_text());self.arrays=[];self.ends=[];total=0
  for shard in manifest["shards"]:array=np.load(Path(embedding_dir)/name/shard["file"],mmap_mode="r");self.arrays.append(array);total+=len(array);self.ends.append(total)
  if total!=len(self.meta):raise ValueError(f"{name}: metadata={len(self.meta)}, embeddings={total}; remove stale embeddings or rerun without --skip-embedding")
 def __len__(self):return len(self.meta)
 def __getitem__(self,index):
  shard=bisect.bisect_right(self.ends,index);local=index-(self.ends[shard-1] if shard else 0);return torch.from_numpy(np.array(self.arrays[shard][local],copy=True)),torch.tensor(int(self.meta.label.iloc[index]),dtype=torch.long),str(self.meta.sample_id.iloc[index])
