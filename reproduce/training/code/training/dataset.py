from __future__ import annotations
import bisect,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self,metadata:pd.DataFrame,embedding_source,split_name:str):
        self.metadata=metadata.reset_index(drop=True);self.labels=self.metadata["label"].astype(float).to_numpy();self.shards=[];self.ends=[];self.split_name=split_name
        if isinstance(embedding_source,(str,Path)):
            root=Path(embedding_source);manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"));total=0
            print(f"[EMBED-LOAD:{split_name}] root={root} shards={len(manifest['shards'])}",flush=True)
            for shard_index, item in enumerate(manifest["shards"], start=1):
                shard_path = root / item["file"]
                loaded = total + len(np.load(shard_path, mmap_mode="r", allow_pickle=False))
                percent = loaded / len(self.metadata) * 100 if len(self.metadata) else 100.0
                print(
                    f"[EMBED-LOAD:{split_name}] loading shard {shard_index}/{len(manifest['shards'])} "
                    f"({percent:.1f}%): {shard_path}",
                    flush=True,
                )
                array=np.load(shard_path,mmap_mode="r",allow_pickle=False);self.shards.append(array);total+=len(array);self.ends.append(total)
            self.shape=(total,*self.shards[0].shape[1:])
        else:
            array=np.asarray(embedding_source,dtype=np.float32);self.shards=[array];self.ends=[len(array)];self.shape=array.shape
        if len(self.metadata)!=self.ends[-1]:raise ValueError(f"{split_name}: Metadata rows ({len(self.metadata)}) and embeddings ({self.ends[-1]}) do not match")
        print(f"[EMBED-LOAD:{split_name}] loaded n={len(self.metadata)} shape={self.shape}",flush=True)
    @property
    def embeddings(self):
        class ShapeProxy:
            def __init__(self,shape):self.shape=shape
        return ShapeProxy(self.shape)
    def __len__(self):return len(self.metadata)
    def __getitem__(self, idx):
        shard = bisect.bisect_right(self.ends, idx)
        local = idx - (0 if shard == 0 else self.ends[shard - 1])
        array = np.array(self.shards[shard][local], dtype=np.float32, copy=True)
        return torch.from_numpy(array), torch.tensor(self.labels[idx], dtype=torch.float32), idx


def _source(directory:Path,name:str):
    sharded=directory/name
    if (sharded/"manifest.json").is_file():return sharded
    legacy=directory/f"{name}.npy"
    if legacy.is_file():return np.load(legacy,mmap_mode="r",allow_pickle=False)
    raise FileNotFoundError(f"Neither sharded nor legacy embedding exists for {name}: {directory}")
def load_embeddings(split_dir,esm_ratio,embeddings_root=None):
    split_dir=Path(split_dir);ratio=str(float(esm_ratio));root=split_dir/"embeddings" if embeddings_root is None else Path(embeddings_root)/split_dir.parent.name/split_dir.name;directory=root/ratio
    if not directory.exists():directory=root/f"{esm_ratio:.2f}".rstrip("0").rstrip(".")
    if not directory.exists():raise FileNotFoundError(f"Embedding directory not found: {directory}")
    return tuple(EmbeddingDataset(pd.read_csv(split_dir/f"{name}.csv"),_source(directory,name),name) for name in ("train","val","test"))
