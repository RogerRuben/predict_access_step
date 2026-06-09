"""
feature_eng.py — 修复 link_id 保留 + status=0 处理
"""
import numpy as np
import pandas as pd
import gc
from logger import get_logger

log = get_logger()

# Stage 1 模型使用的特征列
STAGE1_FEATURE_COLS = [
    "link_current_status", "link_time", "link_ratio", "out_degree",
    "pos_ratio", "link_seq", "link_count", "delta_time", "delta_status",
    "upstream_mean_status", "cum_travel_time", "downstream_cross_time",
    "sin_slice", "cos_slice", "sin_arr_slice", "cos_arr_slice",
]

STAGE1_TARGET = "link_arrival_status"

# Stage 2 额外需要保留的列
STAGE2_EXTRA_COLS = ["link_id", "link_time", "link_ratio", "downstream_cross_time"]


def build_stage1_features_batch(
    link_batch: pd.DataFrame,
    head_batch: pd.DataFrame,
    cross_batch: pd.DataFrame,
    topo: dict,
    keep_extra_for_stage2: bool = False,
) -> pd.DataFrame:
    """
    构建 Stage 1 特征。

    Parameters
    ----------
    keep_extra_for_stage2 : bool
        True = 保留 link_id 等 Stage 2 所需列（用于 predict 后的 Stage 2 计算）
        False = 仅保留训练/预测必需列（节省内存，用于 Stage 1 训练样本提取）
    """
    df = link_batch.copy()

    # 1. 合并 slice_id
    df = df.merge(head_batch[["order_id", "day", "slice_id"]],
                  on=["order_id", "day"], how="left")

    # 2. 出度
    unique_ids = df["link_id"].unique()
    deg_map = {int(lid): len(topo.get(int(lid), [])) for lid in unique_ids}
    df["out_degree"] = df["link_id"].map(deg_map).fillna(0).astype("int16")

    # 3. status=0 处理：替换为 NaN，让 LightGBM 原生处理缺失值
    df["link_current_status"] = df["link_current_status"].replace(0, np.nan).astype("float32")

    # 4. 路径序列上下文
    grp = df.groupby(["order_id", "day"], sort=False)

    df["link_seq"] = grp.cumcount().astype("int16")
    df["link_count"] = grp["link_id"].transform("count").astype("int16")
    df["pos_ratio"] = (df["link_seq"] / df["link_count"].clip(lower=1)).astype("float32")

    df["delta_time"] = grp["link_time"].diff().fillna(0).astype("float32")
    df["delta_status"] = grp["link_current_status"].diff().fillna(0).astype("float32")

    # 上游平均（前5个link滚动平均，NaN 自动跳过）
    df["upstream_mean_status"] = (
        grp["link_current_status"]
        .rolling(window=5, min_periods=1).mean()
        .reset_index(level=[0, 1], drop=True)
        .astype("float32")
    )

    df["wt"] = (df["link_time"] * df["link_ratio"]).astype("float32")
    df["cum_travel_time"] = grp["wt"].cumsum().astype("float32")

    # 5. 下游路口等待时间
    df["next_link_id"] = grp["link_id"].shift(-1)
    mask = df["next_link_id"].notna()
    df["cross_key"] = ""
    if mask.any():
        df.loc[mask, "cross_key"] = (
            df.loc[mask, "link_id"].astype(str) + "_"
            + df.loc[mask, "next_link_id"].astype("Int32").astype(str)
        )

    cross_rename = cross_batch[["order_id", "day", "cross_id", "cross_time"]].rename(
        columns={"cross_id": "cross_key", "cross_time": "downstream_cross_time"}
    )
    df = df.merge(cross_rename, on=["order_id", "day", "cross_key"], how="left")
    df["downstream_cross_time"] = df["downstream_cross_time"].fillna(0).astype("float32")

    # 6. 时空特征
    df["sin_slice"] = np.sin(2 * np.pi * df["slice_id"] / 288).astype("float32")
    df["cos_slice"] = np.cos(2 * np.pi * df["slice_id"] / 288).astype("float32")
    df["arrival_slice_est"] = (df["slice_id"] + (df["cum_travel_time"] / 300)).astype("int16")
    df["sin_arr_slice"] = np.sin(2 * np.pi * df["arrival_slice_est"] / 288).astype("float32")
    df["cos_arr_slice"] = np.cos(2 * np.pi * df["arrival_slice_est"] / 288).astype("float32")

    # 7. 裁剪列
    keep = STAGE1_FEATURE_COLS + [STAGE1_TARGET, "order_id", "day"]
    if keep_extra_for_stage2:
        keep = keep + [c for c in STAGE2_EXTRA_COLS if c not in keep]
    df = df[[c for c in keep if c in df.columns]].copy()

    del grp
    gc.collect()
    return df