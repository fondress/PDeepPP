# Reproducing the partitions, the training runs, and Tables 2-4

Run in this order. Paths at the top of each step need to be set for your machine.

1. **Environment.** Python 3.10.20, PyTorch 2.6.0 (CUDA 12.4), NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, fair-esm 2.0.0; MMseqs2 18.8cc5c for the BP partitions.

2. **Partitions.** The supplied partitions are compressed files in `../data_revised/{BP,PTM}/<task>/{train,val,test}.csv.gz`, with checksums in `MANIFEST.tsv`. The training and baseline scripts expect uncompressed files in `<splits-root>/<task>/split_seed_42/`. Set `SPLITS_BP` and `SPLITS_PTM` to two new, writable directories, then run the following from `reproduce/`:

   ```bash
   export SPLITS_BP=''  # Set to your BP partition directory
   export SPLITS_PTM='' # Set to your PTM partition directory
   python - <<'PY'
   import gzip, os, shutil
   from pathlib import Path

   for family, variable in [('BP', 'SPLITS_BP'), ('PTM', 'SPLITS_PTM')]:
       destination = os.environ.get(variable, '').strip()
       if not destination:
           raise SystemExit(f'Set {variable} first')
       for source in (Path('../data_revised') / family).glob('*/*.csv.gz'):
           target = Path(destination) / source.parent.name / 'split_seed_42' / source.name[:-3]
           target.parent.mkdir(parents=True, exist_ok=True)
           with gzip.open(source, 'rb') as src, target.open('xb') as dst:
               shutil.copyfileobj(src, dst)
   PY
   ```

   The conversion preserves the input rows and refuses to overwrite existing files. Use new directories, or remove the generated files yourself before retrying. To rebuild partitions from original source data instead:
   - BP: `BP_SOURCE_SPLITS=<source datasets> MMSEQS=<mmseqs> python partitions/bp_cluster_split.py <output directory>` (MMseqs2 clustering and all-versus-all search, connected components, whole components allocated to 70/10/20, seed 42).
   - PTM: `python partitions/prepare_data.py --tasks partitions/musite_tasks.csv --config partitions/ptm_split_config.json --splits-root <output directory>` (33-residue windows from the annotated MusiteDeep benchmark FASTA files, split by parent protein, 70/10/20, seed 42). Set `fasta_path` in `musite_tasks.csv` to your copy of the FASTA files.

3. **Embeddings.** Obtain the ESM model separately and set `esm_model_path` and `embeddings_root` in the applicable configuration. `training/code/embedding/` builds cached esm2_t33_650M_UR50D representations. BP comparison runs can build the configured embeddings; the standalone checkpoint runners require prepared caches. Keep generated outputs outside the supplied inputs.

4. **Training.** `training/` holds the runners and the project code they import; `commands.md` gives the command of each analysis (Tables 2 and 3, Figures 2, 4, and 6), including the α × λ grid and the seeds.

5. **Tables 2-4.** `python make_tables.py <per-run-results.csv>` prints the means of Tables 2 and 3 and the paired tests of Table 4. Supply your own results CSV; results are not distributed here.

6. **Reproduced baselines.** `baselines/` holds our reproduction packages of UniDL4BioPep, MusiteDeep (a PyTorch reimplementation of the original topology), and PhosF3C; `commands.md` gives their commands. The settings are in Supplementary Table S5, and the original methods are cited in the paper. These packages run on the same partitions as PDeepPP. Review the settings in `phosf3c_reproduction/config.json` before running.
