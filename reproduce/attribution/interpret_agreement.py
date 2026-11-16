"""Agreement of the PTM attributions with sequence statistics (matched-negative enrichment, one-hot logistic regression).

Inputs: interpret.py outputs (arrays, summaries) and the existing enrichment / one-hot LR tables
(r6/p1_cpu/enrichment/enrichment_<task>.csv, onehot_lr/coef_<task>.csv). Writes OUT/agree.json and OUT/agree_cells_<task>.csv.
usage: python3 interpret_agreement.py [OUTDIR]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import os
G = Path(os.environ["PDEEPPP_ROOT"])  # root directory of the inputs and outputs (see README.md)
M = G
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else M / "r7/ig_out"
P1 = M / "r6/p1_cpu"
PTM = ["Hydroxylysine_K", "Hydroxyproline_P", "Methylarginine_R", "Methyllysine_K", "N-linked_glycosylation_N", "N6-acetyllysine_K", "O-linked_glycosylation_ST",
       "Phosphoserine_threonine_ST", "Phosphotyrosine_Y", "Pyrrolidone_carboxylic_acid_Q", "SUMOylation_K", "S_palmitoylation_C", "Ubiquitination_K"]
AA = "ACDEFGHIKLMNPQRSTVWY"
C = 16
nc = [i for i in range(33) if i != C]
res = {}
for t in PTM:
    p = OUT / t / f"{t}_summary.json"
    if not p.is_file(): continue
    s = json.load(open(p)); z = np.load(OUT / t / f"{t}_arrays.npz"); NP = s["n_pos"]; seeds = s["seeds"]
    sp = M / "musitedeep_reproduction/outputs/splits" / t / "split_seed_42" / "val.csv"
    val = pd.read_csv(sp, usecols=["sequence"]).sequence.values
    ok_aa = set("ACDEFGHIKLMNPQRSTVWYXBZUO-")
    seqs = ["".join(r if r in ok_aa else "X" for r in "".join(str(val[i]).strip().upper().split()))[:33].ljust(33, "-") for i in z["rows"]]
    enr = pd.read_csv(P1 / "enrichment" / f"enrichment_{t}.csv"); enr = enr[enr.negatives == "matched"]
    emax = enr.assign(a=enr.log2_or.abs()).groupby("position").a.max().reindex([i - C for i in nc]).fillna(0).values
    co = pd.read_csv(P1 / "onehot_lr" / f"coef_{t}.csv", index_col=0); cmax = co.abs().max(1).reindex([i - C for i in nc]).values
    r = {}
    for base, tag in (("expected_8neg", ""), ("zero", "_zero"), ("mean_negative", "_mneg")):
        mean_abs = np.mean([np.abs(z[f"ig_s{sd}_{base}"][:NP]).mean(0) for sd in seeds], axis=0)
        r["rho_enrichment" + tag] = float(spearmanr(mean_abs[nc], emax).correlation); r["rho_lr" + tag] = float(spearmanr(mean_abs[nc], cmax).correlation)
    avg = np.mean([z[f"ig_s{sd}_expected_8neg"] for sd in seeds], axis=0)    # signed IG relative to real negative windows
    res_sum, res_cnt = np.zeros((33, 20)), np.zeros((33, 20))
    for i in range(NP):
        for pos, q in enumerate(seqs[i]):
            if q in AA: res_sum[pos, AA.index(q)] += avg[i, pos]; res_cnt[pos, AA.index(q)] += 1
    mean_signed = res_sum / np.maximum(res_cnt, 1)
    rows = []
    for pos in range(33):
        if pos == C: continue
        for k, a in enumerate(AA):
            if res_cnt[pos, k] >= 10: rows.append((pos - C, a, mean_signed[pos, k], int(res_cnt[pos, k])))
    cells = pd.DataFrame(rows, columns=["position", "residue", "mean_signed_ig", "n_pos_windows"])
    cells = cells.merge(enr[["position", "residue", "log2_or", "q"]], on=["position", "residue"], how="left")
    cells.to_csv(OUT / f"agree_cells_{t}.csv", index=False)
    ok = cells.dropna(subset=["log2_or"])
    sig = ok[(ok.q < 0.05) & (ok.log2_or.abs() >= 0.2) & (ok.mean_signed_ig.abs() > 0)]
    r.update(cells_n=int(len(ok)), cells_rho=float(spearmanr(ok.mean_signed_ig, ok.log2_or).correlation) if len(ok) > 5 else None,
             sig_cells_n=int(len(sig)), sig_cells_same_sign=int((np.sign(sig.mean_signed_ig) == np.sign(sig.log2_or)).sum()))
    res[t] = r
(OUT / "agree.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
for t, r in res.items():
    print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
print("median rho enrichment %.3f, LR %.3f" % (np.nanmedian([r["rho_enrichment"] for r in res.values()]), np.nanmedian([r["rho_lr"] for r in res.values()])))
