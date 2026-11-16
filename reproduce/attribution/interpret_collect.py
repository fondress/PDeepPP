"""Collect the 33 per-task outputs of interpret.py into one table, run the step checks, and write the figure data.

usage: python3 interpret_collect.py [OUTDIR]   -> OUTDIR/collect.md, collect.json, fig_data.json
"""
import json
import sys
from pathlib import Path

import numpy as np

import os
G = Path(os.environ["PDEEPPP_ROOT"])  # root directory of the inputs and outputs (see README.md)
A7 = G / "r7"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else A7 / "ig_out"
PTM = ["Hydroxylysine_K", "Hydroxyproline_P", "Methylarginine_R", "Methyllysine_K", "N-linked_glycosylation_N", "N6-acetyllysine_K", "O-linked_glycosylation_ST",
       "Phosphoserine_threonine_ST", "Phosphotyrosine_Y", "Pyrrolidone_carboxylic_acid_Q", "SUMOylation_K", "S_palmitoylation_C", "Ubiquitination_K"]
BP = ["ACE", "Anti-MRSA", "Antibacterial", "Anticancer_alternative", "Anticancer_main", "Antifungal", "Antimalarial_alternative", "Antimalarial_main", "Antimicrobial",
      "Antioxidant", "Antiparasitic", "Antiviral", "BBP", "DPPIV", "Quorum", "TTCA", "Toxicity", "bitter", "neuro", "umami"]
AA = "ACDEFGHIKLMNPQRSTVWY"
S, Z, bad, notes, miss = {}, {}, [], [], []
for t in PTM + BP:
    p = OUT / t / f"{t}_summary.json"
    if not p.is_file(): miss.append(t); continue
    S[t] = json.load(open(p)); Z[t] = np.load(OUT / t / f"{t}_arrays.npz")
nn = lambda v: np.nan if v is None else v
f3 = lambda v: "NA" if v is None or not np.isfinite(v) else f"{v:.3f}"
lines = [f"# R7 item 29 attribution reliability, {len(S)}/33 tasks", "", f"missing: {miss or 'none'}", ""]
rows = {}
for t, s in S.items():
    st, bs, mt = s["stability"], s["baseline_sensitivity"], s["mutation_test"]
    if s["family"] == "PTM":
        r = dict(stab_zero=nn(st["zero|class1"]["spearman_median"]), stab_mneg=nn(st["mean_negative|class1"]["spearman_median"]), stab_e8=nn(st["expected_8neg|class1"]["spearman_median"]),
                 stab_zero_neg=nn(st["zero|class0"]["spearman_median"]),
                 base=float(np.nanmedian([nn(v["median"]) for k, v in bs.items() if k.endswith("class1")])),
                 comp_zero=s["completeness"]["zero"]["median"], comp_mneg=s["completeness"]["mean_negative"]["median"])
    else:
        r = dict(stab_zero=nn(st["A|class1"]["spearman_median"]), stab_mneg=nn(st["G|class1"]["spearman_median"]), stab_e8=nn(st["X|class1"]["spearman_median"]),
                 stab_zero_neg=nn(st["A|class0"]["spearman_median"]),
                 base=float(np.nanmedian([nn(v["median"]) for k, v in bs.items() if k.endswith("class1")])), comp_zero=None, comp_mneg=None)
    r.update(n_pos=s["n_pos"], n_win=s["n_windows"], dtop=mt["delta_top3_mean"], drand=mt["delta_random3_mean"], frac=mt["frac_top_drop_larger"], wp=mt["wilcoxon_p"],
             n_mut=mt["n"], emb=max(s["embedding_check"].values()), ko=s.get("knockout"))
    rows[t] = r
    # step checks
    nan_keys = [k for k in ("stab_zero", "stab_mneg", "stab_e8", "stab_zero_neg", "base") if not np.isfinite(nn(r[k]))]
    if nan_keys: bad.append(f"{t}: NaN in {nan_keys}")
    if r["emb"] >= 2e-3: bad.append(f"{t}: embedding check {r['emb']:.1e}")
    if r["n_pos"] < 20: notes.append(f"{t}: only {r['n_pos']} positive windows in the validation partition (all of them used)")
    if s["family"] == "PTM":
        for k in ("comp_zero", "comp_mneg"):
            if r[k] is None or r[k] > 0.05: bad.append(f"{t}: {k} = {f3(r[k])} (> 5 %)")
    if s["seeds"] != [42, 43, 44, 45, 46]: bad.append(f"{t}: seeds {s['seeds']}")
for fam, names in (("PTM", PTM), ("BP", BP)):
    lines += [f"## {fam}", "", "| task | n pos/win | seed ρ pos (zero|A) | (mneg|G) | (e8|X) | seed ρ neg | baseline ρ | compl. zero | compl. mneg | Δtop3 | Δrand3 | frac | p | n |", "|" + "---|" * 14]
    for t in names:
        if t not in rows: continue
        r = rows[t]; wp = "NA" if r["wp"] is None else "%.2g" % r["wp"]
        lines.append(f"| {t} | {r['n_pos']}/{r['n_win']} | {f3(r['stab_zero'])} | {f3(r['stab_mneg'])} | {f3(r['stab_e8'])} | {f3(r['stab_zero_neg'])} | {f3(r['base'])} | {f3(r['comp_zero'])} | {f3(r['comp_mneg'])} | "
                     f"{f3(r['dtop'])} | {f3(r['drand'])} | {f3(r['frac'])} | {wp} | {r['n_mut']} |")
    lines.append("")
    med = lambda k: float(np.nanmedian([nn(rows[t][k]) for t in names if t in rows])) if any(t in rows for t in names) else None
    sig = sum(1 for t in names if t in rows and rows[t]["wp"] is not None and rows[t]["wp"] < 0.05 and rows[t]["dtop"] < rows[t]["drand"])
    larger = sum(1 for t in names if t in rows and rows[t]["dtop"] < rows[t]["drand"])
    lines += [f"{fam}: median seed ρ (pos) {f3(med('stab_zero'))} / {f3(med('stab_mneg'))} / {f3(med('stab_e8'))}; median baseline ρ {f3(med('base'))}; "
              f"top-3 drop larger than random in {larger}/{len([t for t in names if t in rows])} tasks, p < 0.05 in {sig}.", ""]
ko = {t: r["ko"] for t, r in rows.items() if r["ko"]}
lines += ["## Known-feature knock-outs", "", "| task | rule | n | Δ prob | Δ prob random position |", "|---|---|---|---|---|"]
lines += [f"| {t} | {k['rule']} | {k['n']} | {f3(k['delta_mean'])} | {f3(k['delta_random_mean'])} |" for t, k in ko.items()]
lines += ["", "## Step checks", ""] + ([f"- FAIL {b}" for b in bad] or ["- all checks passed (embedding reproduction < 2e-3, no NaN in stability, completeness ≤ 5 %, 5 seeds)"]) + [f"- note {n}" for n in notes]
(OUT / "collect.md").write_text("\n".join(lines) + "\n")
(OUT / "collect.json").write_text(json.dumps({"rows": rows, "missing": miss, "failed_checks": bad, "notes": notes}, ensure_ascii=False, indent=1, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else float(o)))

# figure data: PTM position profiles, BP residue-type matrix, per-task stability points
fd = {"ptm_tasks": [t for t in PTM if t in Z], "bp_tasks": [t for t in BP if t in Z], "positions": list(range(-16, 17))}
fd["ptm_profile"] = {t: Z[t]["pos_profile_zero"].tolist() for t in PTM if t in Z}
fd["ptm_profile_mneg"] = {t: np.mean([Z[t][f"ig_s{sd}_mean_negative"][:S[t]["n_pos"]].mean(0) for sd in S[t]["seeds"]], axis=0).tolist() for t in PTM if t in Z}
fd["ptm_profile_e8"] = {t: np.mean([Z[t][f"ig_s{sd}_expected_8neg"][:S[t]["n_pos"]].mean(0) for sd in S[t]["seeds"]], axis=0).tolist() for t in PTM if t in Z}
fd["total_attr_median"] = {t: {b: float(np.median(np.concatenate([Z[t][f"ig_s{sd}_{b}"][:S[t]["n_pos"]].sum(1) for sd in S[t]["seeds"]]))) for b in ("zero", "mean_negative", "expected_8neg")} for t in PTM if t in Z}
fd["ptm_resmat"] = {t: (Z[t]["resmat_sum"] / np.maximum(Z[t]["resmat_count"], 1)).tolist() for t in PTM if t in Z}
bpm = {}
for t in BP:
    if t in Z:
        v, c = Z[t]["restype_sum_pos"], Z[t]["restype_count_pos"]; bpm[t] = np.where(c > 0, v / np.maximum(c, 1), np.nan).tolist()
fd["bp_restype"] = bpm
fd["stability_pairs"] = {t: S[t]["stability"][("zero|class1" if S[t]["family"] == "PTM" else "A|class1")]["spearman_pairs"] for t in S}
fd["baseline_median"] = {t: rows[t]["base"] for t in rows}
fd["mutation"] = {t: {k: rows[t][k] for k in ("dtop", "drand", "wp", "frac", "n_mut")} for t in rows}
fd["knockout"] = ko
(OUT / "fig_data.json").write_text(json.dumps(fd, ensure_ascii=False, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else float(o)))
print((OUT / "collect.md").read_text()[:3000]); print("FAILED CHECKS:", bad)
