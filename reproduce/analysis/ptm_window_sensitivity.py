"""Compare exact-training-window matched/unmatched PTM test predictions (no retraining)."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


SEEDS = {42, 43, 44, 45, 46}


def selected_runs(runs_file):
    best = {}
    with gzip.open(runs_file, "rt") as handle:
        for line in handle:
            run = json.loads(line)
            if run["block"] != "t3":
                continue
            key = (run["task"], int(run["seed"]))
            # Keep the first occurrence of ties, as in r7_release_results.py.
            if key not in best or run["best_validation_pr_auc"] > best[key]["best_validation_pr_auc"]:
                best[key] = run
    tasks = {task for task, _ in best}
    assert len(best) == 65 and len(tasks) == 13, (len(best), len(tasks))
    assert all({seed for task, seed in best if task == name} == SEEDS for name in tasks)
    return best


def metrics(y, probability):
    if not len(y):
        return float("nan"), float("nan")
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    ap = float("nan")
    if n_positive:
        order = np.argsort(-probability, kind="mergesort")
        score, label = probability[order], y[order]
        ends = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
        true_positives = np.cumsum(label, dtype=np.int64)[ends]
        precision = true_positives / (ends + 1)
        ap = float(np.sum(np.diff(np.r_[0, true_positives]) * precision) / n_positive)
    if not n_positive or not n_negative:
        return ap, float("nan")
    prediction = probability >= 0.5
    tp = int(np.count_nonzero(prediction & (y == 1)))
    fp = int(np.count_nonzero(prediction & (y == 0)))
    fn, tn = n_positive - tp, n_negative - fp
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.0
    return ap, mcc


def read_partition(partitions_dir, task):
    path = partitions_dir / task
    with gzip.open(path / "train.csv.gz", "rt") as handle:
        training = {row["sequence"] for row in csv.DictReader(handle)}
    with gzip.open(path / "test.csv.gz", "rt") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int8)
    assert np.isin(labels, [0, 1]).all(), task
    assert len({row["sample_id"] for row in rows}) == len(rows), task
    # Use the unmodified padded sequence, exactly as in supp_ptm_audit_table.py.
    matched = np.array([row["sequence"] in training for row in rows], dtype=bool)
    return rows, labels, matched


def read_predictions(path, test_rows, labels):
    with path.open() as handle:
        columns = next(csv.reader(handle))
    values = np.loadtxt(path, delimiter=",", skiprows=1,
                        usecols=[columns.index("y_true"), columns.index("y_prob")], ndmin=2)
    assert values.shape == (len(labels), 2), (path.name, values.shape, len(labels))
    assert np.array_equal(values[:, 0], labels), f"Label/order mismatch: {path.name}"
    probability = values[:, 1]
    assert np.isfinite(probability).all(), f"Nonfinite predictions: {path.name}"
    assert ((probability >= 0) & (probability <= 1)).all(), f"Out-of-range predictions: {path.name}"
    if "sample_id" in columns or "sequence" in columns:
        with path.open() as handle:
            for observed, expected in zip(csv.DictReader(handle), test_rows):
                for key in ("sample_id", "sequence"):
                    if key in columns:
                        assert observed[key] == expected[key], f"{key} mismatch: {path.name}"
    return probability


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if isinstance(value, float) and math.isnan(value) else value
                             for key, value in row.items()})


def self_test():
    y = np.array([1, 0, 1, 0], dtype=np.int8)
    p = np.array([0.9, 0.8, 0.8, 0.1])
    ap, mcc = metrics(y, p)
    assert abs(ap - 5 / 6) < 1e-15, ap
    assert abs(mcc - 1 / math.sqrt(3)) < 1e-15, mcc
    assert metrics(y, np.ones(4)) == (0.5, 0.0)
    assert metrics(np.array([0, 1]), np.array([0.1, 0.5])) == (1.0, 1.0)
    assert all(math.isnan(x) for x in metrics(np.array([0, 0]), np.array([0.1, 0.2])))
    training = {"-" + "A" * 32, "B" * 33}
    observed = ["-" + "A" * 32, "A" * 33, "B" * 33, "X" + "A" * 32]
    assert [sequence in training for sequence in observed] == [True, False, True, False]


def required_path(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-file", type=required_path, required=True, help="Gzipped JSONL run-selection records")
    parser.add_argument("--partitions-dir", type=required_path, required=True, help="Directory containing PTM task partitions")
    parser.add_argument("--audit-file", type=required_path, required=True, help="PTM audit CSV")
    parser.add_argument("--predictions-dir", type=required_path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = parser.parse_args()
    self_test()
    runs = selected_runs(args.runs_file)
    with args.audit_file.open() as handle:
        audit = {row["task"]: row for row in csv.DictReader(handle)}
    per_run, crosscheck = [], []
    for task in sorted({task for task, _ in runs}):
        test_rows, labels, matched = read_partition(args.partitions_dir, task)
        assert len(labels) == int(audit[task]["test_windows"]), task
        assert abs(float(matched.mean()) - float(audit[task]["identical_window_frac"])) < 1e-12, task
        for seed in sorted(SEEDS):
            run = runs[task, seed]
            filename = f"{task}__{run['tag']}__seed{seed}.csv"
            probability = read_predictions(args.predictions_dir / filename, test_rows, labels)
            ap, mcc = metrics(labels, probability)
            assert len(labels) == int(run["n_test"]), filename
            assert abs(ap - run["pr_auc"]) <= 1e-9, (filename, "AP", ap, run["pr_auc"])
            assert abs(mcc - run["mcc"]) <= 1e-9, (filename, "MCC", mcc, run["mcc"])
            crosscheck.append(dict(task=task, seed=seed, tag=run["tag"], n=len(labels),
                                   pr_auc_recorded=run["pr_auc"], pr_auc_recomputed=ap,
                                   pr_auc_abs_error=abs(ap - run["pr_auc"]),
                                   mcc_recorded=run["mcc"], mcc_recomputed=mcc,
                                   mcc_abs_error=abs(mcc - run["mcc"]),
                                   overlap_fraction=float(matched.mean()),
                                   overlap_fraction_audit=float(audit[task]["identical_window_frac"])))
            for group, mask in (("matched", matched), ("unmatched", ~matched)):
                group_labels = labels[mask]
                group_ap, group_mcc = metrics(group_labels, probability[mask])
                n, n_positive = len(group_labels), int(group_labels.sum())
                per_run.append(dict(task=task, seed=seed, tag=run["tag"], group=group,
                                    n=n, n_positive=n_positive, n_negative=n - n_positive,
                                    positive_rate=n_positive / n if n else float("nan"),
                                    pr_auc=group_ap, mcc=group_mcc))
        print(f"{task}: {int(matched.sum())}/{len(labels)} matched; 5 runs checked", flush=True)
    summary = []
    for task in sorted({row["task"] for row in per_run}):
        for group in ("matched", "unmatched"):
            rows = [row for row in per_run if row["task"] == task and row["group"] == group]
            row = {key: rows[0][key] for key in ("task", "group", "n", "n_positive", "n_negative", "positive_rate")}
            row["n_seeds"] = len(rows)
            for metric in ("pr_auc", "mcc"):
                values = np.array([item[metric] for item in rows])
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_std"] = float(np.std(values, ddof=1))
            summary.append(row)
    assert len(per_run) == 130 and len(summary) == 26 and len(crosscheck) == 65
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_run.csv", per_run)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "crosscheck.csv", crosscheck)
    print(f"Saved 130 per-run groups, 26 summary rows and 65 full-test checks to {args.output_dir}")


if __name__ == "__main__":
    main()
