from __future__ import annotations

from typing import Dict

import pandas as pd


def _overlap(a: pd.Series, b: pd.Series) -> int:
    return len(set(a.astype(str)) & set(b.astype(str)))


def build_bp_leakage_report(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict[str, int | bool]:
    report = {
        "train_val_sequence_overlap": _overlap(train["sequence"], val["sequence"]),
        "train_test_sequence_overlap": _overlap(train["sequence"], test["sequence"]),
        "val_test_sequence_overlap": _overlap(val["sequence"], test["sequence"]),
        "train_val_group_overlap": _overlap(train["split_group"], val["split_group"]),
        "train_test_group_overlap": _overlap(train["split_group"], test["split_group"]),
        "val_test_group_overlap": _overlap(val["split_group"], test["split_group"]),
    }
    report["leakage_free"] = all(v == 0 for v in report.values() if isinstance(v, int))
    return report
