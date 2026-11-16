from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from ..utils import normalize_sequence


@dataclass
class BpColumns:
    sequence_col: str = "sequence"
    label_col: str = "label"
    split_group_col: Optional[str] = None


def clean_bp_dataframe(raw_df: pd.DataFrame, columns: BpColumns, task: str) -> Tuple[pd.DataFrame, dict]:
    if columns.sequence_col not in raw_df.columns:
        raise ValueError(f"{task}: missing column {columns.sequence_col}")
    if columns.label_col not in raw_df.columns:
        raise ValueError(f"{task}: missing column {columns.label_col}")

    df = pd.DataFrame()
    df["task"] = task
    df["sequence"] = raw_df[columns.sequence_col].map(normalize_sequence)
    df["label"] = pd.to_numeric(raw_df[columns.label_col], errors="coerce")
    df = df[df["sequence"].str.len() > 0].copy()
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    conflict_seqs = [seq for seq, group in df.groupby("sequence") if group["label"].nunique() > 1]
    if conflict_seqs:
        df = df[~df["sequence"].isin(conflict_seqs)].copy()

    df = df.drop_duplicates(subset=["sequence", "label"]).reset_index(drop=True)
    df["sample_id"] = [f"{task}_{i}" for i in range(len(df))]
    if columns.split_group_col and columns.split_group_col in raw_df.columns:
        df["split_group"] = raw_df.loc[df.index, columns.split_group_col].astype(str).values
    else:
        df["split_group"] = df["sequence"]

    report = {
        "task": task,
        "raw_n": int(len(raw_df)),
        "clean_n": int(len(df)),
        "positive_n": int((df["label"] == 1).sum()),
        "negative_n": int((df["label"] == 0).sum()),
        "conflict_sequence_n": int(len(conflict_seqs)),
    }
    return df[["sample_id", "task", "sequence", "label", "split_group"]].copy(), report
