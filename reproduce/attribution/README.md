# Attribution analysis (Section 3.3, Figure 5, Supplementary Tables S25 and S26)

Run in this order. Set `PDEEPPP_ROOT` to your input and output directory. Supply the ESM model, BP partitions, embeddings and scalers under `r6/`, selected checkpoints and the `ptm_selected.csv` and `bp_selected.csv` selection tables under `r7/ig_stage/`, and PTM partitions and embeddings under `musitedeep_reproduction/outputs/splits` and `results/pdeeppp_ptm_embeddings`. The agreement analysis also needs per-task enrichment and one-hot logistic-regression tables under `r6/p1_cpu/`. Checkpoints, selection tables and statistics are not included in this repository.

1. `python interpret.py <task> [--n 200] [--steps 128]`, once per task: integrated gradients for the 13 PTM tasks, single-residue replacement for the 20 BP tasks (their models read one pooled vector per peptide), stability over seeds, baseline sensitivity, top-3 against random-3 replacement, known-feature replacement.
2. `python interpret_collect.py [out dir]`: one table, the step checks, and the figure data (`collect.json`, `fig_data.json`).
3. `python interpret_agreement.py [out dir]`: comparison with the sequence statistics of the training windows.
4. `ATTR_OUT=<out dir> python attribution_periodicity.py`: three-residue periodicity of the PTM profiles.
