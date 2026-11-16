from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score

BINS=[-1e-9,.3,.5,.7,.9,1.000001]; LABELS=["<30%","30-50%","50-70%","70-90%",">=90%"]
def sha256(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def canonical_hash(frame):
    text="\n".join(f"{s}\t{int(y)}" for s,y in sorted(zip(frame.sequence.astype(str),frame.label.astype(int))));return hashlib.sha256(text.encode()).hexdigest()
def identity(a,b):
    length=max(len(a),len(b));return sum(x==y for x,y in zip(a,b))/length if length else 0.0
def metrics(y,p):
    z=p>=.5
    result={"n":len(y),"positive":int(y.sum()),"negative":int((y==0).sum()),"acc":float(np.mean(z==y)) if len(y) else np.nan,"pr_auc":np.nan,"roc_auc":np.nan,"mcc":matthews_corrcoef(y,z) if len(y) else np.nan,"bacc":balanced_accuracy_score(y,z) if len(np.unique(y))>1 else np.nan}
    if len(np.unique(y))>1:result.update(pr_auc=average_precision_score(y,p),roc_auc=roc_auc_score(y,p))
    return result
def bootstrap(pred,repeats=1000,seed=42):
    rng=np.random.default_rng(seed);y=pred.y_true.to_numpy();p=pred.y_prob.to_numpy();rows=[]
    for _ in range(repeats):
        idx=rng.integers(0,len(y),len(y));yy,pp=y[idx],p[idx]
        if len(np.unique(yy))<2:continue
        m=metrics(yy,pp);rows.append([m[x] for x in ("pr_auc","roc_auc","mcc","bacc")])
    a=np.asarray(rows);names=["pr_auc","roc_auc","mcc","bacc"]
    return {f"{n}_{q}":float(np.percentile(a[:,i],v)) for i,n in enumerate(names) for q,v in (("ci_low",2.5),("ci_high",97.5))}
def aggregate(output:Path):
    d=output/"metrics";d.mkdir(parents=True,exist_ok=True);files=sorted((output/"run_results").glob("*.json"));raw=pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in files]);raw.to_csv(d/"raw_results.csv",index=False);raw.to_csv(d/"all_trials.csv",index=False)
    numeric=[c for c in raw.select_dtypes(include="number") if c not in {"seed","gpu_id","pid"}];g=raw.groupby(["task","model"])[numeric].agg(["mean","std"]);g.columns=[f"{a}_{b}" for a,b in g.columns];g.reset_index().to_csv(d/"mean_sd_by_task.csv",index=False)
    pd.DataFrame([{**{"task":r.task,"seed":r.seed},**bootstrap(pd.read_csv(output/"predictions"/f"{r.task}_seed{r.seed}.csv"))} for r in raw.itertuples()]).to_csv(d/"bootstrap_ci.csv",index=False);return raw
def qc(tasks:pd.DataFrame,splits_root:Path,config:dict,output:Path):
    root=output/"qc";root.mkdir(parents=True,exist_ok=True);counts=[];sequences={};labels={};identity_rows=[];bin_rows=[];fingerprints=[];overlap_details=[]
    expected={str(row.task):str(row.get("center_residues", "")) for _,row in tasks.fillna("").iterrows()};centers=[]
    for _,row in tasks.fillna("").iterrows():
        task=str(row.task);d=splits_root/task/f"split_seed_{config['split_seed']}";parts={n:pd.read_csv(d/f"{n}.csv") for n in ("train","val","test")};allf=pd.concat(parts.values());sequences[task]=set(allf.sequence);labels[task]=dict(zip(allf.sequence,allf.label))
        for name,f in parts.items():
            path=d/f"{name}.csv";counts.append({"task":task,"split":name,"n":len(f),"positive":int(f.label.sum()),"negative":int((f.label==0).sum()),"positive_rate":float(f.label.mean()),"min_length":int(f.sequence.str.len().min()),"max_length":int(f.sequence.str.len().max())});fingerprints.append({"task":task,"artifact":name,"path":str(path),"sha256":sha256(path),"canonical_sequence_label_sha256":canonical_hash(f)})
        overlaps={f"exact_{a}_{b}_overlap":len(set(parts[a].sequence)&set(parts[b].sequence)) for a,b in (("train","val"),("train","test"),("val","test"))};(d/"leakage_report.json").write_text(json.dumps(overlaps,indent=2),encoding="utf-8")
        train_sequences=parts["train"].sequence.astype(str).tolist()
        for row_index, r in enumerate(parts["test"].itertuples(index=False)):
            sample_id=getattr(r, "sample_id", f"{task}_test_{row_index}")
            best=max((identity(str(r.sequence),x) for x in train_sequences),default=0.0);identity_rows.append({"task":task,"sample_id":sample_id,"sequence":r.sequence,"label":r.label,"max_train_identity":best,"identity_bin":pd.cut([best],BINS,labels=LABELS,include_lowest=True)[0]})
        target=expected.get(task,"");bad=allf[~allf.sequence.map(lambda s:bool(s) and s[len(s)//2] in target)] if target else allf.iloc[0:0];centers.append({"task":task,"expected_center":target,"invalid_center_n":len(bad)})
    identity_df=pd.DataFrame(identity_rows);identity_df.to_csv(root/"max_train_identity_by_test_sample.csv",index=False)
    for (task,seed_file) in [(p.stem.rsplit("_seed",1)[0],p) for p in (output/"predictions").glob("*_seed*.csv")]:
        seed=int(seed_file.stem.rsplit("seed",1)[1]);pred=pd.read_csv(seed_file);joined=pred.merge(identity_df[["task","sample_id","identity_bin"]],on="sample_id");
        for bucket,f in joined.groupby("identity_bin",observed=False):
            if len(f):bin_rows.append({"task":task,"seed":seed,"identity_bin":str(bucket),**metrics(f.y_true.to_numpy(),f.y_prob.to_numpy())})
    pd.DataFrame(bin_rows).to_csv(root/"performance_by_identity_bin.csv",index=False);pd.DataFrame(counts).to_csv(root/"class_balance_and_length.csv",index=False);pd.DataFrame(centers).to_csv(root/"center_residue_check.csv",index=False);pd.DataFrame(fingerprints).to_csv(root/"data_fingerprints.csv",index=False)
    pairs=[];names=list(sequences)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            common=sequences[a]&sequences[b];agree=sum(labels[a][s]==labels[b][s] for s in common);pairs.append({"task_a":a,"task_b":b,"overlap":len(common),"jaccard":len(common)/max(1,len(sequences[a]|sequences[b])),"label_agreement_n":agree,"label_conflict_n":len(common)-agree,"label_agreement_rate":agree/len(common) if common else np.nan});overlap_details.extend({"task_a":a,"task_b":b,"sequence":s,"label_a":labels[a][s],"label_b":labels[b][s]} for s in common)
    pd.DataFrame(pairs).to_csv(root/"task_pairwise_sequence_overlap.csv",index=False);pd.DataFrame(overlap_details).to_csv(root/"task_pairwise_overlap_details.csv",index=False)
