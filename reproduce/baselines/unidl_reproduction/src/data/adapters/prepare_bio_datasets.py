from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ...utils import ensure_dir, normalize_sequence, read_table, write_json


BP_SUFFIXES = ("_train", "_test")
PTM_FILENAMES = ("train.xlsx", "test.xlsx")


def _find_bp_files(task_dir: Path) -> List[Path]:
    files = []
    for p in task_dir.iterdir():
        if p.is_file() and any(p.stem.endswith(suf) for suf in BP_SUFFIXES):
            files.append(p)
    return sorted(files)


def _find_ptm_files(task_dir: Path) -> List[Path]:
    files = []
    for name in PTM_FILENAMES:
        p = task_dir / name
        if p.exists():
            files.append(p)
    return files


def _infer_label_from_name(path: Path) -> Optional[int]:
    lower = path.stem.lower()
    if "train" in lower or "pos" in lower or "positive" in lower:
        return 1
    if "test" in lower or "neg" in lower or "negative" in lower:
        return 0
    return None


def _standardize_table(df: pd.DataFrame, task: str, source_path: Path, label_value: Optional[int] = None) -> pd.DataFrame:
    cols = {str(c).lower(): c for c in df.columns}
    seq_col = None
    for cand in ["sequence", "seq", "peptide", "peptides"]:
        if cand in cols:
            seq_col = cols[cand]
            break
    label_col = None
    for cand in ["label", "labels", "y", "target"]:
        if cand in cols:
            label_col = cols[cand]
            break

    if seq_col is None and df.shape[1] >= 1:
        seq_col = df.columns[0]
    if seq_col is None:
        raise ValueError(f"Cannot infer sequence column from {source_path}")

    out = pd.DataFrame()
    out["task"] = task
    out["source_file"] = str(source_path)
    out["sequence"] = df[seq_col].map(normalize_sequence)
    if label_col is not None:
        out["label"] = pd.to_numeric(df[label_col], errors="coerce")
    elif label_value is not None:
        out["label"] = label_value
    else:
        raise ValueError(f"Cannot infer label for {source_path}; add a label column or use pos/neg filenames.")
    out = out[out["sequence"].str.len() > 0].copy()
    out = out[out["label"].isin([0, 1])].copy()
    out["label"] = out["label"].astype(int)
    out["sample_id"] = [f"{task}_{i}" for i in range(len(out))]
    out["split_group"] = out["sequence"]
    return out[["sample_id", "task", "sequence", "label", "split_group", "source_file"]].copy()


def prepare_bp_dataset(task_dir: str | Path, task_name: str, out_dir: str | Path) -> Tuple[pd.DataFrame, dict]:
    task_dir = Path(task_dir)
    out_dir = ensure_dir(out_dir)
    files = _find_bp_files(task_dir)
    if not files:
        raise FileNotFoundError(f"No BP files with _train/_test suffix found in {task_dir}")

    parts = []
    for file in files:
        df = read_table(file)
        label_value = _infer_label_from_name(file)
        parts.append(_standardize_table(df, task_name, file, label_value=label_value))
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["sequence", "label"]).reset_index(drop=True)
    combined.to_csv(out_dir / f"{task_name}.csv", index=False)
    report = {
        "task": task_name,
        "task_type": "BP",
        "task_dir": str(task_dir),
        "output_csv": str(out_dir / f"{task_name}.csv"),
        "n_files": len(files),
        "n_samples": int(len(combined)),
        "positive_n": int((combined["label"] == 1).sum()),
        "negative_n": int((combined["label"] == 0).sum()),
        "source_files": [str(p) for p in files],
    }
    write_json(report, out_dir / f"{task_name}.json")
    return combined, report


def prepare_ptm_dataset(task_dir: str | Path, task_name: str, out_dir: str | Path) -> Tuple[pd.DataFrame, dict]:
    task_dir = Path(task_dir)
    out_dir = ensure_dir(out_dir)
    files = _find_ptm_files(task_dir)
    if not files:
        raise FileNotFoundError(f"No PTM files named train.xlsx/test.xlsx found in {task_dir}")
    parts = []
    for file in files:
        df = read_table(file)
        parts.append(_standardize_table(df, task_name, file, label_value=None))
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["sequence", "label"]).reset_index(drop=True)
    combined.to_csv(out_dir / f"{task_name}.csv", index=False)
    report = {
        "task": task_name,
        "task_type": "PTM",
        "task_dir": str(task_dir),
        "output_csv": str(out_dir / f"{task_name}.csv"),
        "n_files": len(files),
        "n_samples": int(len(combined)),
        "positive_n": int((combined["label"] == 1).sum()),
        "negative_n": int((combined["label"] == 0).sum()),
        "source_files": [str(p) for p in files],
    }
    write_json(report, out_dir / f"{task_name}.json")
    return combined, report
