"""R6 BP benchmark: MMseqs2 30%-identity / 80%-coverage clusters, cluster-level stratified 70/10/20 split.

Input rows = union of the exact-duplicate-free partitions used in round 3
(unidl_reproduction/outputs/splits/<task>/split_seed_42/{train,val,test}.csv), so the
sample set and labels are unchanged; only the partition assignment changes.

Clustering: mmseqs cluster, --min-seq-id 0.3, -c 0.8, --cov-mode 0 (both sequences),
--cluster-mode 1 (connected component: any detected pair above threshold ends up in the
same cluster, so no detected above-threshold pair can cross partitions), -s 7.5,
k-mer 5, no spaced k-mers, permissive e-value so short peptides can be aligned.

Assignment: whole clusters, largest first (random tie order, seed 42), each to the
partition whose class-wise deficit is largest -> approximately 70/10/20 per class.
"""
import json
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(os.environ["BP_SOURCE_SPLITS"])   # directory with <task>/split_seed_42/{train,val,test}.csv of the source BP datasets
MM = os.environ.get("MMSEQS", "mmseqs")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
TARGET = {"train": 0.7, "val": 0.1, "test": 0.2}
CL = ["--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0", "--cluster-mode", "1", "-s", "7.5", "-k", "5",
      "--spaced-kmer-mode", "0", "-e", "100", "--threads", "16", "-v", "1"]


def run(cmd):
    subprocess.run(cmd, check=True)


def clusters(task, seqs, work):
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "seqs.fasta", "w") as h:
        for i, s in enumerate(seqs):
            h.write(f">s{i}\n{s}\n")
    run([MM, "createdb", str(work / "seqs.fasta"), str(work / "db"), "-v", "1"])
    run([MM, "cluster", str(work / "db"), str(work / "clu"), str(work / "tmp")] + CL)
    run([MM, "createtsv", str(work / "db"), str(work / "db"), str(work / "clu"), str(work / "clu.tsv"), "-v", "1"])
    tab = pd.read_csv(work / "clu.tsv", sep="\t", header=None, names=["rep", "mem"])
    # union-find over (i) mmseqs cluster membership and (ii) every all-vs-all search hit that
    # passes the same thresholds, so no detected >=30%/80% pair can cross partitions
    parent = list(range(len(seqs)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for r, mbr in zip(tab.rep, tab.mem):
        union(int(r[1:]), int(mbr[1:]))
    run([MM, "easy-search", str(work / "seqs.fasta"), str(work / "seqs.fasta"), str(work / "all.m8"), str(work / "tmp2"),
         "--threads", "16", "-s", "7.5", "-e", "100", "--max-seqs", "2000", "-k", "5", "--spaced-kmer-mode", "0",
         "--format-output", "query,target,fident,qcov,tcov", "-v", "1"])
    try:
        hits = pd.read_csv(work / "all.m8", sep="\t", header=None, names=["q", "t", "fident", "qcov", "tcov"])
    except pd.errors.EmptyDataError:
        hits = pd.DataFrame(columns=["q", "t", "fident", "qcov", "tcov"])
    hits = hits[(hits.fident >= 0.3) & (hits[["qcov", "tcov"]].min(axis=1) >= 0.8) & (hits.q != hits.t)]
    for q, t in zip(hits.q, hits.t):
        union(int(q[1:]), int(t[1:]))
    roots = [find(i) for i in range(len(seqs))]
    lab = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([lab[r] for r in roots])


def assign(df, seed=42):
    n_c = df.groupby("label").size()
    comp = df.groupby(["cluster_id", "label"]).size().unstack(fill_value=0)
    order = comp.sample(frac=1.0, random_state=seed)
    order = order.loc[order.sum(1).sort_values(ascending=False, kind="stable").index]
    filled = {s: pd.Series(0.0, index=comp.columns) for s in TARGET}
    out = {}
    for cid, row in order.iterrows():
        score = {s: float(((TARGET[s] * n_c - filled[s]) / n_c * row).sum()) for s in TARGET}
        s = max(score, key=score.get)
        out[cid] = s; filled[s] += row
    return df.cluster_id.map(out)


summ = []
for tdir in sorted(p for p in SRC.iterdir() if (p / "split_seed_42/test.csv").is_file()):
    task = tdir.name
    df = pd.concat([pd.read_csv(tdir / "split_seed_42" / f"{s}.csv").assign(r3_split=s) for s in TARGET], ignore_index=True)
    assert df.sequence.is_unique, task
    df["cluster_id"] = clusters(task, df.sequence.astype(str).tolist(), OUT / "_mmseqs" / task)
    df["r6_split"] = assign(df)
    df["split_group"] = df.cluster_id.astype(str)
    d = OUT / task / "split_seed_42"; d.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["r6_split"]).to_csv(d / "clustered.csv", index=False)
    sizes = df.groupby("cluster_id").size()
    meta = {"task": task, "split_strategy": "sequence_identity_cluster_stratified", "edges": "mmseqs cluster (connected component) + all-vs-all easy-search hits with identity>=0.3 and min(qcov,tcov)>=0.8", "backend": "mmseqs2",
            "mmseqs_version": subprocess.run([MM, "version"], capture_output=True, text=True).stdout.strip(),
            "mmseqs_cluster_args": CL, "split_seed": 42, "targets": TARGET, "n_total": len(df),
            "n_clusters": int(sizes.size), "n_singletons": int((sizes == 1).sum()),
            "largest_cluster": int(sizes.max()), "largest_cluster_frac": float(sizes.max() / len(df)),
            "cluster_size_quantiles": {q: float(sizes.quantile(q)) for q in (0.5, 0.9, 0.99)},
            "n_seq_shorter_than_8": int((df.sequence.str.len() < 8).sum())}
    for s in TARGET:
        part = df[df.r6_split == s].drop(columns=["r6_split", "cluster_id"]).reset_index(drop=True)
        part[["sample_id", "task", "sequence", "label", "split_group"]].to_csv(d / f"{s}.csv", index=False)
        meta[f"n_{s}"] = len(part); meta[f"{s}_pos"] = int((part.label == 1).sum()); meta[f"{s}_neg"] = int((part.label == 0).sum())
        meta[f"{s}_clusters"] = int(df[df.r6_split == s].cluster_id.nunique())
    cs = {s: set(df[df.r6_split == s].cluster_id) for s in TARGET}
    meta["cluster_overlap"] = {"train_val": len(cs["train"] & cs["val"]), "train_test": len(cs["train"] & cs["test"]),
                               "val_test": len(cs["val"] & cs["test"])}
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))
    sizes.rename("size").to_csv(d / "cluster_sizes.csv")
    summ.append(meta); print(task, meta["n_clusters"], meta["largest_cluster_frac"], meta["n_train"], meta["n_val"], meta["n_test"], flush=True)
pd.DataFrame(summ).drop(columns=["mmseqs_cluster_args", "targets", "cluster_size_quantiles", "cluster_overlap"]).to_csv(OUT / "bp_cluster_split_summary.csv", index=False)
print("ALL-DONE", flush=True)
