"""
stage1_feature_transform.py
轻量特征变换模块，供 prepare 和 inference 共用。
"""

import numpy as np
import pandas as pd

from stage1_feature_schema import (
    RAW_ID_FEATURES,
    ORDINAL_RAW_FEATURES,
    PERIODIC_FEATURES,
    ID_CLIP_RANGES,
)


def transform_stage1_feature_array(
    df: pd.DataFrame,
    feature_cols: list,
    mean_dict: dict,
    std_dict: dict,
) -> np.ndarray:
    """
    返回 ndarray，用于推理或写入 shard。
    """
    x = df[feature_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)

    mean_s = pd.Series(mean_dict)
    std_s = pd.Series(std_dict)

    for col in feature_cols:
        if col in RAW_ID_FEATURES or col in ORDINAL_RAW_FEATURES:
            x[col] = x[col].fillna(0)

        elif col in PERIODIC_FEATURES:
            x[col] = x[col].fillna(0).clip(-1.0, 1.0)

        else:
            m = float(mean_s.get(col, 0.0))
            s = float(std_s.get(col, 1.0))
            if not np.isfinite(s) or s <= 1e-8:
                s = 1.0
            x[col] = (x[col].fillna(m) - m) / s

    for col, (lo, hi) in ID_CLIP_RANGES.items():
        if col in x.columns:
            if hi is None:
                x[col] = x[col].clip(lower=lo)
            else:
                x[col] = x[col].round().clip(lo, hi)

    return (
        x.astype("float32")
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .values
    )


def transform_stage1_feature_frame(
    df: pd.DataFrame,
    feature_cols: list,
    mean_dict: dict,
    std_dict: dict,
) -> pd.DataFrame:
    """
    返回 DataFrame，保留非特征列，用于 prepare_stage1_dataset.py。
    """
    out = df.copy()
    arr = transform_stage1_feature_array(out, feature_cols, mean_dict, std_dict)

    x_scaled = pd.DataFrame(
        arr,
        columns=feature_cols,
        index=out.index,
    )

    for col in feature_cols:
        out[col] = x_scaled[col].astype("float32")

    return out