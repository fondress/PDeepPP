# PDeepPP

PDeepPP is a sequence-based framework for bioactive-peptide (BP) classification and post-translational modification (PTM) site prediction. It trains a separate model for each task using frozen ESM-2 representations and parallel convolutional and attention branches.

This repository contains the input data and code for the revised study. Experiment results, prediction files and model checkpoints are not included. To run the workflows, supply the required models and set the input and output paths for your environment.

## Repository layout

| Path | Contents |
| --- | --- |
| [`data_revised/`](data_revised/) | BP and PTM training, validation and test partitions, with a data manifest. |
| [`reproduce/`](reproduce/) | Data preparation, embedding, training, baseline and analysis code. |
| [`reproduce/commands.md`](reproduce/commands.md) | Commands for the experiments and reproduced baselines. |
| [`reproduce/attribution/`](reproduce/attribution/) | Attribution analysis scripts and their input requirements. |
| [`legacy_audit/`](legacy_audit/) | Script for auditing the earlier dataset. |
| [`requirements.txt`](requirements.txt) | Python dependencies for reproduction. |

## Environment

The reproduction environment uses Python 3.10 and the packages in `requirements.txt` (`timm` is used by PhosF3C and `openpyxl` by the legacy Excel audit):

```bash
python -m pip install -r requirements.txt
```

GPU-enabled PyTorch and CUDA are required for the supplied GPU training configurations. MMseqs2 is needed only when rebuilding BP partitions. An ESM-2 model file must be supplied separately; the trained PDeepPP checkpoints are not included. See [`reproduce/README.md`](reproduce/README.md) for the environment and workflow details.

## Input data

[`data_revised/`](data_revised/) contains partitions for 20 BP and 13 PTM tasks:

- `BP/<task>/{train,val,test}.csv.gz`: bioactive-peptide partitions. Sequences are deduplicated and split by sequence-similarity components.
- `PTM/<task>/{train,val,test}.csv.gz`: PTM partitions. Each record contains a padded 33-residue window; proteins are disjoint between partitions, although identical window sequences may occur in different partitions.
- `MANIFEST.tsv`: row and label counts and the MD5 checksum of each uncompressed CSV.

The supplied partitions are compressed. Training scripts expect uncompressed files arranged under `<task>/split_seed_42/{train,val,test}.csv`; prepare them as described in [`reproduce/README.md`](reproduce/README.md) before running training. To rebuild the PTM partitions from annotated FASTA inputs, fill in the `fasta_path` column of [`reproduce/partitions/musite_tasks.csv`](reproduce/partitions/musite_tasks.csv).

## Running the code

1. Set up the environment, prepare the partitions and provide the ESM-2 model file.
2. Set the empty model and embedding paths in the relevant configuration files, including [`bp_paperloss_config.json`](reproduce/training/bp_paperloss_config.json), [`bp_control_config.json`](reproduce/training/bp_control_config.json), and the baseline configurations when applicable. Choose your own input, output and cache locations.
3. Follow [`reproduce/README.md`](reproduce/README.md) for the workflow and [`reproduce/commands.md`](reproduce/commands.md) for training and baseline commands. The scripts generate results and checkpoints in the output locations you choose.

The table-generation script requires a compatible per-run results CSV produced or provided by the user:

```bash
python reproduce/make_tables.py <per-run-results.csv>
```

The PTM exact-window sensitivity script is [`reproduce/analysis/ptm_window_sensitivity.py`](reproduce/analysis/ptm_window_sensitivity.py). It requires user-provided run records, partitions, audit data and predictions; run `python reproduce/analysis/ptm_window_sensitivity.py --help` for its arguments. Attribution analysis requires additional checkpoints and statistics, as described in its [README](reproduce/attribution/README.md).
