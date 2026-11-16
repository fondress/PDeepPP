"""R7 item 29 (R4-M8): attribution reliability of the Table 2 / Table 3 PDeepPP checkpoints, one task per process.

PTM (Table 3, per-residue 33 x 1280 input): integrated gradients (zero / mean-negative / 8-negative baselines),
  seed stability, baseline sensitivity, completeness, top-3 versus random-3 mutation test, known-feature knock-outs.
BP (Table 2, whole-peptide 1 x 1280 input, so no per-residue gradient exists): single-residue replacement effects
  (Ala / Gly / X), seed stability, replacement-scheme sensitivity, top-3 versus random-3 mutation test.
Windows come from the validation partition; checkpoints are the per-seed validation-selected cells of Tables 2 and 3.

usage: interpret.py TASK [--n 200] [--steps 128] [--seeds 42 43 44 45 46] [--dry] [--out DIR]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, wilcoxon

# The fused attention kernels (flash / memory-efficient) return NaN or ~1e19 input gradients for several Table 3 checkpoints on H200
# (probe 2026-11-07: math kernel = CPU result); every run uses the math kernel.
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)

import os
G = Path(os.environ["PDEEPPP_ROOT"])  # root directory of the inputs and outputs (see README.md)
M = G
A7 = M / "r7"
SRC = Path(__file__).resolve().parents[1] / "training"
R6 = M / "r6"
STAGE = A7 / "ig_stage"
sys.path.insert(0, str(SRC)); sys.modules.pop("code", None)  # the stdlib `code` module must not shadow the project package
import esm  # noqa: E402
from code.embedding import build_bp_unidl_aligned_embeddings as BB  # noqa: E402
from code.embedding import build_pdeeppp_embeddings as PB  # noqa: E402
from code.models.pdeeppp import PDeepPPClassifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("task"); ap.add_argument("--n", type=int, default=200); ap.add_argument("--steps", type=int, default=128)
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46]); ap.add_argument("--dry", action="store_true")
ap.add_argument("--out", default=str(A7 / "ig_out"))
args = ap.parse_args()
TASK = args.task
OUT = Path(args.out) / TASK; OUT.mkdir(parents=True, exist_ok=True)
dev = torch.device("cpu" if args.dry else "cuda")
MODEL_KWARGS = dict(output_size=128, use_position_encoding=True, transformer_heads=8, transformer_hidden=256, transformer_layers=4,
                    transformer_ff_dim=512, cnn_channels=64, fusion_channels=32, dropout=0.15)
C = 16
AA = "ACDEFGHIKLMNPQRSTVWY"
SIGKO = {"N-linked_glycosylation_N": (2, "ST"), "SUMOylation_K": (2, "E"), "Hydroxyproline_P": (1, "G"), "Hydroxylysine_K": (1, "G"),
         "Phosphoserine_threonine_ST": (1, "P"), "Methylarginine_R": (1, "G")}
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

# ---------- checkpoints ----------
ptm_sel = pd.read_csv(STAGE / "ptm_selected.csv"); bp_sel = pd.read_csv(STAGE / "bp_selected.csv")
IS_PTM = TASK in set(ptm_sel.task)
sel = (ptm_sel if IS_PTM else bp_sel)
sel = sel[(sel.task == TASK) & sel.seed.isin(args.seeds)].sort_values("seed")
assert len(sel) == len(args.seeds), (TASK, len(sel))
models = {}
for r in sel.itertuples():
    path = STAGE / "ptm_ckpts" / r.ckpt if IS_PTM else Path(r.ckpt)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    m = PDeepPPClassifier(input_size=1280, **MODEL_KWARGS); m.load_state_dict(sd); m.to(dev).eval()
    models[int(r.seed)] = (m, float(r.esm_ratio), Path(path).name)
seeds = sorted(models)
alphas = sorted({a for _, a, _ in models.values()})
log("task", TASK, "family", "PTM" if IS_PTM else "BP", "seeds", {s: models[s][1:] for s in seeds})

# ---------- ESM-2 and the auxiliary projection, seeded exactly as the embedding builders ----------
random.seed(42); np.random.seed(42); torch.manual_seed(42)
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(M / "esm2_t33_650M_UR50D.pt"))
torch.load = _orig_load
esm_model = esm_model.to(dev).eval()
conv = alphabet.get_batch_converter()
if IS_PTM:
    vocab = {a: i for i, a in enumerate("XACDEFGHIKLMNPQRSTVWY-")}
    proj = PB.EmbeddingPretrainedModel(len(vocab), 128).to(dev).eval()
else:
    vocab = {a: i for i, a in enumerate("XACDEFGHIKLMNPQRSTVWY")}
    proj = BB.EmbeddingPretrainedModel(len(vocab), 128, 1280).to(dev).eval()


def ptm_embed(seqs, alpha):
    out = []
    for s in range(0, len(seqs), 128):
        sb = seqs[s:s + 128]
        _, _, tok = conv([(str(i), q.replace("-", "X")) for i, q in enumerate(sb)])
        idx = torch.tensor([[vocab.get(a, 0) for a in q] for q in sb], device=dev)
        with torch.inference_mode():
            f = esm_model(tok.to(dev), repr_layers=[33])["representations"][33][:, 1:34]
            out.append((alpha * f + (1 - alpha) * proj(idx)).float().cpu())
    return torch.cat(out).clone()


_scalers = {}


def bp_embed(seqs, alpha):
    if alpha not in _scalers:
        z = np.load(R6 / "bp_cluster_embeddings" / TASK / "split_seed_42" / str(float(alpha)) / "unidl_minmax_scaler.npz")
        scale = z["data_max"] - z["data_min"]; _scalers[alpha] = (z["data_min"], np.where(scale == 0, 1.0, scale))
    mn, sc = _scalers[alpha]
    order = np.argsort([len(q) for q in seqs]); vec = [None] * len(seqs)
    for s in range(0, len(seqs), 128):
        ids = order[s:s + 128]; sb = list(seqs[i] for i in ids)
        _, _, tok = conv([(f"s{k}", q) for k, q in enumerate(sb)])
        with torch.inference_mode():
            rep = esm_model(tok.to(dev), repr_layers=[33], return_contacts=False)["representations"][33]
            for k, q in enumerate(sb):
                idx = torch.tensor([vocab.get(a, 0) for a in q], device=dev)
                vec[ids[k]] = (alpha * rep[k, 1:len(q) + 1] + (1 - alpha) * proj(idx)).mean(0).float().cpu().numpy()
    v = (np.stack(vec) - mn) / sc
    return torch.from_numpy(v.astype(np.float32))[:, None, :]


embed = ptm_embed if IS_PTM else bp_embed


def logits(seed, x, bs=512):
    m = models[seed][0]
    with torch.no_grad():
        return torch.cat([m(x[i:i + bs].to(dev)).float().reshape(-1).cpu() for i in range(0, len(x), bs)]).numpy()


# ---------- windows (validation partition) ----------
if IS_PTM:
    SPLIT = M / "musitedeep_reproduction/outputs/splits" / TASK / "split_seed_42"; EMB = M / "results/pdeeppp_ptm_embeddings"
else:
    SPLIT = R6 / "bp_cluster_splits" / TASK / "split_seed_42"; EMB = R6 / "bp_cluster_embeddings"
va = pd.read_csv(SPLIT / "val.csv")
rng = np.random.default_rng(20261107)
y = va.label.values; pos_idx = np.flatnonzero(y == 1); neg_idx = np.flatnonzero(y == 0)
pool = neg_idx
if IS_PTM and "protein_id" in va:  # negatives from proteins that carry a positive (same matching as the R6 analysis)
    matched = neg_idx[np.isin(va.protein_id.astype(str).values[neg_idx], list(set(va.protein_id.astype(str)[y == 1])))]
    pool = matched if len(matched) >= args.n else neg_idx
pick_p = np.sort(rng.choice(pos_idx, min(args.n, len(pos_idx)), replace=False))
pick_n = np.sort(rng.choice(pool, min(args.n, len(neg_idx)), replace=False))
rows = np.r_[pick_p, pick_n]; lab = y[rows]; NP = len(pick_p)
if IS_PTM:
    seqs = [PB._prepare_fixed_length_sequence(q)[0] for q in va.sequence.values[rows]]
else:
    seqs = ["".join(r if r in BB.VALID_AA else "X" for r in "".join(str(q).strip().upper().split())) for q in va.sequence.values[rows]]
log("windows", len(rows), "pos", NP, "alphas", alphas)

# ---------- cached embeddings: only the shards that hold the selected rows are opened ----------
def read_rows(split, a, idx):
    root = EMB / TASK / "split_seed_42" / str(float(a)) / split
    man = json.load(open(root / "manifest.json")); starts = np.array([x["start"] for x in man["shards"]]); files = [x["file"] for x in man["shards"]]
    idx = np.asarray(idx); sh = np.searchsorted(starts, idx, side="right") - 1
    out = np.empty((len(idx),) + tuple(man["shape"][1:]), np.float32)
    for k in np.unique(sh):
        sel = np.flatnonzero(sh == k)
        out[sel] = np.load(root / files[k], mmap_mode="r", allow_pickle=False)[idx[sel] - starts[k]]
    return torch.from_numpy(out)


# ---------- reproduce the cached embeddings before using the re-embedding path ----------
check = {}
for a in alphas:
    k = min(16, len(rows))
    check[str(a)] = float((embed(seqs[:k], a) - read_rows("val", a, rows[:k])).abs().max())
log("re-embedding check", check)
assert max(check.values()) < 2e-3, f"re-embedding does not reproduce the cache: {check}"


def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o


def row_spearman(a, b):
    """a, b: (n, k) -> Spearman correlation of each row."""
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra = ra - ra.mean(1, keepdims=True); rb = rb - rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)


def top3_overlap(a, b):
    ta, tb = np.argsort(-np.abs(a), axis=1)[:, :3], np.argsort(-np.abs(b), axis=1)[:, :3]
    j = [len(set(u) & set(v)) / len(set(u) | set(v)) for u, v in zip(ta, tb)]
    return np.array(j)


S = {"task": TASK, "family": "PTM" if IS_PTM else "BP", "seeds": seeds, "alphas": {s: models[s][1] for s in seeds},
     "ckpt": {s: models[s][2] for s in seeds}, "n_windows": len(rows), "n_pos": int(NP), "embedding_check": check,
     "steps": args.steps if IS_PTM else None}
ARR = {"label": lab, "rows": rows}
pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]

if IS_PTM:
    nc = [i for i in range(33) if i != C]
    xs = {}
    trlab = pd.read_csv(SPLIT / "train.csv", usecols=["label"]).label.values
    trneg = np.flatnonzero(trlab == 0)
    starts_tr = np.array([x["start"] for x in json.load(open(EMB / TASK / "split_seed_42" / str(float(alphas[0])) / "train" / "manifest.json"))["shards"]])
    sh_of = np.searchsorted(starts_tr, trneg, side="right") - 1
    keep = rng.choice(np.unique(sh_of), min(60, len(np.unique(sh_of))), replace=False)   # negatives drawn from at most 60 random shards
    cand = trneg[np.isin(sh_of, keep)]
    bg = np.sort(rng.choice(cand, min(2000, len(cand)), replace=False))
    for a in alphas:
        bgx = read_rows("train", a, bg)
        xs[a] = (read_rows("val", a, rows), bgx.mean(0), bgx[:8])
    log("embeddings loaded")

    def ig(m, x, base, bs=400):
        out, gap = [], []
        for s in range(0, len(x), bs):
            xb = x[s:s + bs].to(dev); bb = (base if base.dim() == 2 else base[s:s + bs]).to(dev).expand_as(xb)
            tot = torch.zeros_like(xb)
            for k in range(1, args.steps + 1):
                z = (bb + k / args.steps * (xb - bb)).requires_grad_(True)
                g, = torch.autograd.grad(m(z).reshape(-1).sum(), z); tot += g
            att = (xb - bb) * tot / args.steps
            with torch.no_grad():
                dl = m(xb).reshape(-1) - m(bb.contiguous()).reshape(-1)
                gap.append(torch.stack([(att.sum((1, 2)) - dl).abs(), dl.abs()], 1).cpu())
            out.append(att.sum(2).detach().cpu())
        return torch.cat(out).numpy(), torch.cat(gap).numpy()

    prof, comp = {}, {}
    for s in seeds:
        m, a, _ = models[s]; x, mneg, egx = xs[a]
        for name, base in (("zero", torch.zeros(33, 1280)), ("mean_negative", mneg)):
            prof[(s, name)], g = ig(m, x, base)
            comp[(s, name)] = float(np.median(g[:, 0] / np.maximum(g[:, 1], 1e-6)))
        acc = 0
        for j in range(8):
            p, _ = ig(m, x, egx[j]); acc = acc + p
        prof[(s, "expected_8neg")] = acc / 8
        log("IG done seed", s)
    bases = ("zero", "mean_negative", "expected_8neg")
    for (s, b), v in prof.items(): ARR[f"ig_s{s}_{b}"] = v.astype(np.float32)

    def top3_of(v): return np.argsort(-np.abs(v[:, nc]), axis=1)[:, :3]
    stab = {}
    for b in bases:
        for cls in (1, 0):
            k = lab == cls
            rho = np.array([np.nanmedian(row_spearman(prof[(a, b)][k][:, nc], prof[(c, b)][k][:, nc])) for a, c in pairs])
            jac = np.array([np.mean(top3_overlap(prof[(a, b)][k][:, nc], prof[(c, b)][k][:, nc])) for a, c in pairs])
            stab[f"{b}|class{cls}"] = {"spearman_pairs": rho, "spearman_median": float(np.nanmedian(rho)), "top3_jaccard_median": float(np.median(jac))}
    base_sens = {}
    for b1, b2 in (("zero", "mean_negative"), ("zero", "expected_8neg"), ("mean_negative", "expected_8neg")):
        for cls in (1, 0):
            k = lab == cls
            r = np.array([np.nanmedian(row_spearman(prof[(s, b1)][k][:, nc], prof[(s, b2)][k][:, nc])) for s in seeds])
            base_sens[f"{b1}|{b2}|class{cls}"] = {"per_seed": r, "median": float(np.nanmedian(r))}
    S.update(stability=stab, baseline_sensitivity=base_sens,
             completeness={b: {"per_seed": [comp[(s, b)] for s in seeds], "median": float(np.median([comp[(s, b)] for s in seeds]))} for b in ("zero", "mean_negative")})
    # signed attribution profile of true positives (zero baseline, seed mean) and its residue-resolved form
    zavg = np.mean([prof[(s, "zero")] for s in seeds], axis=0)
    ARR["pos_profile_zero"] = zavg[:NP].mean(0)
    res = np.zeros((33, 20)); cnt = np.zeros((33, 20))
    for i in range(NP):
        for p, q in enumerate(seqs[i]):
            if q in AA: res[p, AA.index(q)] += zavg[i, p]; cnt[p, AA.index(q)] += 1
    ARR["resmat_sum"], ARR["resmat_count"] = res, cnt

    # mutation tests on the positives
    def mutate(q, ps):
        q = list(q)
        for p in ps:
            if q[p] != "-": q[p] = "G" if q[p] == "A" else "A"
        return "".join(q)

    def score(sq):
        ps = []
        for s in seeds:
            x = embed(sq, models[s][1]); ps.append(1 / (1 + np.exp(-logits(s, x))))
        return np.mean(ps, 0)

    pseq = seqs[:NP]
    p0 = score(pseq)
    top = [list(np.array(nc)[np.argsort(-zavg[i, nc])[:3]]) for i in range(NP)]
    pt = score([mutate(q, t) for q, t in zip(pseq, top)])
    prs = [score([mutate(q, list(np.random.default_rng(100 + d).choice(nc, 3, replace=False))) for q in pseq]) for d in range(5)]
    prand = np.mean(prs, 0)
    S["mutation_test"] = {"n": NP, "p_orig_mean": float(p0.mean()), "delta_top3_mean": float((pt - p0).mean()), "delta_random3_mean": float((prand - p0).mean()),
                          "frac_top_drop_larger": float(np.mean((pt - p0) < (prand - p0))),
                          "wilcoxon_p": float(wilcoxon(pt - p0, prand - p0).pvalue) if NP > 5 else None}
    ARR.update(p_orig=p0, p_top3=pt, p_random3=prand)
    if TASK in SIGKO:
        pos, resid = SIGKO[TASK]; kk = [i for i, q in enumerate(pseq) if q[C + pos] in resid]
        if kk:
            pk = score([mutate(pseq[i], [C + pos]) for i in kk])
            rr = []
            for d in range(5):
                r_ = np.random.default_rng(200 + d)
                rr.append(score([mutate(pseq[i], [int(r_.choice([p for p in nc if p != C + pos]))]) for i in kk]))
            S["knockout"] = {"rule": f"{pos:+d} {resid} -> A", "n": len(kk), "delta_mean": float((pk - p0[kk]).mean()),
                             "delta_random_mean": float((np.mean(rr, 0) - p0[kk]).mean())}
else:
    # ---------- BP: single-residue replacement effects on the whole-peptide model ----------
    L = np.array([len(q) for q in seqs]); off = np.r_[0, np.cumsum(L)]
    schemes = {"A": lambda r: "G" if r == "A" else "A", "G": lambda r: "A" if r == "G" else "G", "X": lambda r: "X"}
    orig = {}
    for a in alphas: orig[a] = embed(seqs, a)
    lg0 = {s: logits(s, orig[models[s][1]]) for s in seeds}
    eff = {sc: {s: np.full((len(seqs), L.max()), np.nan, np.float32) for s in seeds} for sc in schemes}
    for sc, f in schemes.items():
        mut = [q[:p] + f(q[p]) + q[p + 1:] for q in seqs for p in range(len(q))]
        E = {a: embed(mut, a) for a in alphas}
        for s in seeds:
            d = lg0[s].repeat(L) - logits(s, E[models[s][1]])   # positive = replacing the residue lowers the positive logit
            for i in range(len(seqs)): eff[sc][s][i, :L[i]] = d[off[i]:off[i + 1]]
        log("occlusion scheme", sc, "mutants", len(mut))
    for sc in schemes:
        for s in seeds: ARR[f"eff_{sc}_s{s}"] = eff[sc][s]
    ok = L >= 5
    stab = {}
    for sc in schemes:
        for cls in (1, 0):
            k = np.flatnonzero((lab == cls) & ok)
            rho = np.array([np.nanmedian([np.corrcoef(rankdata(eff[sc][a][i, :L[i]]), rankdata(eff[sc][c][i, :L[i]]))[0, 1] for i in k]) for a, c in pairs])
            stab[f"{sc}|class{cls}"] = {"spearman_pairs": rho, "spearman_median": float(np.nanmedian(rho)), "n_windows": int(len(k))}
    base_sens = {}
    for s1, s2 in (("A", "G"), ("A", "X"), ("G", "X")):
        for cls in (1, 0):
            k = np.flatnonzero((lab == cls) & ok)
            r = np.array([np.nanmedian([np.corrcoef(rankdata(eff[s1][s][i, :L[i]]), rankdata(eff[s2][s][i, :L[i]]))[0, 1] for i in k]) for s in seeds])
            base_sens[f"{s1}|{s2}|class{cls}"] = {"per_seed": r, "median": float(np.nanmedian(r))}
    S.update(stability=stab, baseline_sensitivity=base_sens)
    avg = {sc: np.mean([eff[sc][s] for s in seeds], axis=0) for sc in schemes}
    rt = {}
    for cls, name in ((1, "pos"), (0, "neg")):
        v = np.zeros(20); c = np.zeros(20)
        for i in np.flatnonzero(lab == cls):
            for p, q in enumerate(seqs[i]):
                if q in AA: v[AA.index(q)] += avg["A"][i, p]; c[AA.index(q)] += 1
        ARR[f"restype_sum_{name}"], ARR[f"restype_count_{name}"] = v, c
    # mutation test on the positives with L >= 6: rank by the X replacement, mutate to Ala
    kp = [i for i in range(NP) if L[i] >= 6]; pseq = [seqs[i] for i in kp]

    def mutate(q, ps):
        q = list(q)
        for p in ps: q[p] = "G" if q[p] == "A" else "A"
        return "".join(q)

    def score(sq):
        ps = []
        for a in alphas:
            E = embed(sq, a)
            for s in seeds:
                if models[s][1] == a: ps.append(1 / (1 + np.exp(-logits(s, E))))
        return np.mean(ps, 0)

    p0 = score(pseq)
    top = [list(np.argsort(-avg["X"][i, :L[i]])[:3]) for i in kp]
    pt = score([mutate(q, t) for q, t in zip(pseq, top)])
    prs = [score([mutate(q, list(np.random.default_rng(100 + d + i).choice(len(q), 3, replace=False))) for i, q in enumerate(pseq)]) for d in range(5)]
    prand = np.mean(prs, 0)
    S["mutation_test"] = {"n": len(kp), "p_orig_mean": float(p0.mean()), "delta_top3_mean": float((pt - p0).mean()), "delta_random3_mean": float((prand - p0).mean()),
                          "frac_top_drop_larger": float(np.mean((pt - p0) < (prand - p0))),
                          "wilcoxon_p": float(wilcoxon(pt - p0, prand - p0).pvalue) if len(kp) > 5 else None}
    ARR.update(p_orig=p0, p_top3=pt, p_random3=prand)

np.savez_compressed(OUT / f"{TASK}_arrays.npz", **ARR)
tmp = OUT / f"{TASK}_summary.json.tmp"
tmp.write_text(json.dumps(jsonable(S), ensure_ascii=False, indent=1))
os.replace(tmp, OUT / f"{TASK}_summary.json")
log("R7-INTERPRET-DONE", TASK)
