from __future__ import annotations

import argparse, hashlib, json, re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

AA = set("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_AA = set("BJOUZ")


def resolve(path: str, base: Path) -> Path:
    value = Path(path.replace("\\", "/")); return value if value.is_absolute() else (base / value).resolve()


def read_annotated_fasta(path: Path) -> list[tuple[str, str, set[int]]]:
    records=[]; current_id=None; chunks=[]
    for raw in path.read_text(encoding="utf-8").splitlines():
        line=raw.strip()
        if not line: continue
        if line.startswith(">"):
            if current_id is not None: records.append(_finish_record(current_id, "".join(chunks)))
            current_id=line[1:].strip().split()[0]; chunks=[]
        else: chunks.append(line.replace(" ", ""))
    if current_id is not None: records.append(_finish_record(current_id, "".join(chunks)))
    if not records: raise ValueError(f"No FASTA records found: {path}")
    return records


def _finish_record(identifier: str, marked: str) -> tuple[str, str, set[int]]:
    sequence=[]; positives=set(); residue_position=-1
    for char in marked.upper():
        if char == "#":
            if residue_position >= 0: positives.add(residue_position)
            continue
        if char in AA or char == "X":
            residue_position += 1; sequence.append(char)
        elif char in AMBIGUOUS_AA:
            residue_position += 1; sequence.append("X")
        elif char in "-_." or char.isspace():
            continue
        else:
            raise ValueError(f"Invalid FASTA character {char!r} in {identifier}")
    return identifier, "".join(sequence), positives


def extract_windows(records, focus: set[str], flank: int = 16, padding: str = "-") -> pd.DataFrame:
    rows=[]
    for protein_id, sequence, positives in records:
        for position, center in enumerate(sequence):
            if center not in focus: continue
            left=sequence[max(0, position-flank):position]; right=sequence[position+1:position+flank+1]
            window=padding*(flank-len(left))+left+center+right+padding*(flank-len(right))
            if len(window) != 2*flank+1: raise AssertionError(f"Invalid window length for {protein_id}:{position}")
            rows.append({"protein_id":protein_id,"site_position":position,"center_residue":center,"sequence":window,"label":int(position in positives),"source":"annotated_fasta"})
    if not rows: raise ValueError(f"No candidate residues found for focus={sorted(focus)}")
    return pd.DataFrame(rows)


def clean(frame: pd.DataFrame, task: str, focus: set[str], flank: int) -> tuple[pd.DataFrame, dict]:
    frame=frame.copy(); frame["sequence"]=frame.sequence.astype(str).str.upper(); frame["label"]=frame.label.astype(int)
    frame=frame[frame.sequence.str.len().eq(2*flank+1)].copy(); invalid=~frame.sequence.map(lambda s: all(c in AA or c=="-" or c=="X" for c in s)); invalid_n=int(invalid.sum()); frame=frame[~invalid].copy()
    conflict=frame.groupby(["protein_id","site_position"]).label.nunique(); conflicts=conflict[conflict>1].index; frame=frame[~frame.set_index(["protein_id","site_position"]).index.isin(conflicts)].copy(); duplicate_n=len(frame)-len(frame.drop_duplicates(["protein_id","site_position","label"])); frame=frame.drop_duplicates(["protein_id","site_position","label"])
    frame.insert(0,"task",task); frame.insert(0,"sample_id",[f"{task}_{i}" for i in range(len(frame))]); frame["split_group"]=frame.protein_id
    report={"task":task,"raw_n":len(frame)+duplicate_n,"clean_n":len(frame),"positive_n":int(frame.label.sum()),"negative_n":int((frame.label==0).sum()),"invalid_length_n":0,"invalid_alphabet_n":invalid_n,"conflict_site_n":len(conflicts),"duplicate_site_label_n":duplicate_n,"required_sequence_length":2*flank+1,"focus_residues":"".join(sorted(focus))}
    return frame.reset_index(drop=True),report


def split_groups(frame: pd.DataFrame, test_size: float, val_size: float, seed: int):
    outer=GroupShuffleSplit(1,test_size=test_size,random_state=seed); trainval_i,test_i=next(outer.split(frame,frame.label,frame.split_group)); trainval=frame.iloc[trainval_i]; test=frame.iloc[test_i]
    val_fraction=val_size/(1-test_size); inner=GroupShuffleSplit(1,test_size=val_fraction,random_state=seed+1000); train_i,val_i=next(inner.split(trainval,trainval.label,trainval.split_group)); return trainval.iloc[train_i].reset_index(drop=True),trainval.iloc[val_i].reset_index(drop=True),test.reset_index(drop=True)


def sha(frame):
    text="\n".join(f"{r.sequence}\t{r.label}\t{r.protein_id}\t{r.site_position}" for r in frame.itertuples()); return hashlib.sha256(text.encode()).hexdigest()


def prepare_all(tasks_path: Path, config_path: Path, splits_root: Path) -> None:
    config=json.loads(config_path.read_text(encoding="utf-8")); tasks=pd.read_csv(tasks_path).fillna(""); metadata=[]; flank=int(config.get("window_flank",16))
    for _,row in tasks.iterrows():
        if pd.isna(row.fasta_path) or not str(row.fasta_path).strip(): raise ValueError(f"Set fasta_path for task {row.task} in {tasks_path}")
        task=str(row.task); fasta=resolve(str(row.fasta_path),tasks_path.parent); focus=set(str(row.center_residues).replace(",", "").replace(" ", "")); records=read_annotated_fasta(fasta); extracted=extract_windows(records,focus,flank,"-"); frame,report=clean(extracted,task,focus,flank); train,val,test=split_groups(frame,float(config["test_size"]),float(config["val_size"]),int(config["split_seed"]))
        directory=splits_root/task/f"split_seed_{config['split_seed']}"; directory.mkdir(parents=True,exist_ok=True); frame.to_csv(directory/"cleaned.csv",index=False)
        for name,part in (("train",train),("val",val),("test",test)): part.to_csv(directory/f"{name}.csv",index=False)
        meta={**report,"source_fasta":str(fasta),"source_sha256":hashlib.sha256(fasta.read_bytes()).hexdigest(),"split_seed":config["split_seed"],"split_strategy":"protein_group_disjoint","n_train":len(train),"n_val":len(val),"n_test":len(test),"train_proteins":train.protein_id.nunique(),"val_proteins":val.protein_id.nunique(),"test_proteins":test.protein_id.nunique(),"train_hash":sha(train),"val_hash":sha(val),"test_hash":sha(test)}
        (directory/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); metadata.append(meta)
    splits_root.mkdir(parents=True,exist_ok=True); pd.DataFrame(metadata).to_csv(splits_root/"all_split_metadata.csv",index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument("--tasks",required=True);p.add_argument("--config",required=True);p.add_argument("--splits-root",default="outputs/splits");a=p.parse_args();prepare_all(Path(a.tasks).resolve(),Path(a.config).resolve(),Path(a.splits_root).resolve())
if __name__=="__main__":main()
