# Commands of each analysis

First prepare `$SPLITS_BP` and `$SPLITS_PTM` as described in [README.md](README.md): the distributed files are compressed and need conversion to `<task>/split_seed_42/{train,val,test}.csv`. Set `$EMB_PTM` to your prepared PTM embedding cache and `$OUT` to a writable output directory. In the BP JSON configs, set `embeddings_root` and `esm_model_path` to your own paths. Run the training examples from `reproduce/training/`. Values in angle brackets are placeholders to replace, not literal shell arguments. One task and one configuration is shown; repeat as needed.

For a PTM task, the embedding builder can prepare the cache used by the standalone checkpoint runners. From `reproduce/training/`, after setting `esm_model_path` in `bp_paperloss_config.json`, run:

    python -m code.embedding.build_pdeeppp_embeddings --split-dir "$SPLITS_PTM/<task>/split_seed_42" --output-root "$EMB_PTM" --config bp_paperloss_config.json --device cuda

Replace `<task>` and choose a suitable `--device` (for example, `cpu`); the configuration's `esm_ratios` determines which caches are built. Generation requires substantial storage and compute.

**Table 2, PDeepPP (BP; α × λ grid, checkpoint and configuration selected on validation PR AUC).** Set the two paths in `bp_paperloss_config.json` first. `<task>.csv` is a CSV with a header and task row, including `task` and `task_type`; the supplied `../baselines/unidl_reproduction/config/unidl_tasks.csv` provides a BP task list.

    python -u -m code.training.run_pdeeppp_comparison --tasks <task>.csv --config bp_paperloss_config.json --splits-root $SPLITS_BP --output-root $OUT/bp_paperloss/<task>

**Table 3, PDeepPP (PTM).** For α in 0.9, 0.95, 1.0 and λ in 0.5, 1, 2; per task and seed, the run with the highest validation PR AUC is reported.

    python -u train_checkpoint.py --task <task> --task-type PTM --split-dir $SPLITS_PTM/<task>/split_seed_42 --embeddings-root $EMB_PTM --variant full --esm-ratio <α> --loss tim_paper --lambda-value <λ> --batch-size 512 --lr 1e-3 --max-epochs 100 --patience 20 --seeds 42 43 44 45 46 --tag t3_esm<α>_lam<λ> --out-root $OUT/paperloss_t3 --data-on-gpu 1

Phosphoserine/threonine uses `train_checkpoint_prefetch.py` with the same arguments (its cache is a memory-mapped file on local disk).

**Figure 2, ESM-2 controls.** BP:

    python -u -m code.training.run_pdeeppp_comparison --tasks <task>.csv --config bp_control_config.json --splits-root $SPLITS_BP --output-root $OUT/paperloss_fig2/bp/<task>

PTM, with `<variant>` = `esm_mlp` or `esm_linear`:

    python -u train_control.py --task <task> --task-type PTM --split-dir $SPLITS_PTM/<task>/split_seed_42 --embeddings-root $EMB_PTM --variant <variant> --esm-ratio 1.0 --loss tim_paper --lambda-value 2 --batch-size 512 --lr 1e-3 --max-epochs 100 --patience 20 --seeds 42 43 44 45 46 --tag f2_<variant> --out-root $OUT/paperloss_fig2/ptm --data-on-gpu 1 --center-only 1 --amp-dtype bfloat16

**Figure 4, training objectives.** `<loss>` = `bce`, `weighted_bce`, or `focal`; the TIM-inspired row is `--loss tim_paper --lambda-value 2 --max-epochs 100`. BP omits `--batch-size` and `--lr` (the defaults are the same values).

    python -u train_checkpoint.py --task <task> --task-type <BP|PTM> --split-dir <partition> --embeddings-root <embeddings> --variant full --esm-ratio 0.95 --loss <loss> --lambda-value 0.5 --batch-size 512 --lr 1e-3 --max-epochs 100 --patience 20 --seeds 42 43 44 45 46 --tag e100_<loss> --out-root $OUT/paperloss_fig4 --data-on-gpu 1 --amp-dtype bfloat16

**Figure 6, component analysis.** `<variant>` = `wo_poscnn`, `wo_pe`, `wo_attention`, or `wo_translinear`; the full model is `--variant full --max-epochs 100`.

    python -u train_checkpoint.py --task <task> --task-type <BP|PTM> --split-dir <partition> --embeddings-root <embeddings> --variant <variant> --esm-ratio 0.95 --loss tim_paper --lambda-value 2 --max-epochs 100 --patience 20 --seeds 42 43 44 45 46 --tag abl100_<variant> --out-root $OUT/paperloss_abl --data-on-gpu 1 --amp-dtype bfloat16

## Reproduced baselines

Run from `reproduce/baselines/` (MusiteDeep: from `reproduce/baselines/musitedeep_reproduction/`). The partition directories must be writable: baseline preprocessing writes generated embeddings or encodings within them.

**UniDL4BioPep (Table 2).** Set the ESM-2 path in `unidl_config.json` and put the BP partitions in `$SPLITS_BP` (`<task>/split_seed_42/`).

    python -u -m unidl_reproduction.src.train.run_repeated_seeds --tasks unidl_reproduction/config/unidl_tasks.csv --config unidl_reproduction/config/unidl_config.json --splits-root $SPLITS_BP --output-root $OUT/unidl --use-existing-splits

**MusiteDeep (Table 3).**

    python -m src.encode_sequences --tasks config/musite_tasks.csv --config config/musite_config.example.json --splits-root $SPLITS_PTM
    python -m src.run_experiments --tasks config/musite_tasks.csv --config config/musite_config.example.json --splits-root $SPLITS_PTM --output-root $OUT/musitedeep

**PhosF3C (Table 3).** Set the ESM-2 path and review `max_epochs` in `config.json`.

    python phosf3c_reproduction/run.py --tasks <PTM task list> --config phosf3c_reproduction/config.json --splits-root $SPLITS_PTM --output-root $OUT/phosf3c

