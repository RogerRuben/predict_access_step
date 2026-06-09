"""
feature_eng.py — 内存优化版
"""

import numpy as np
import pandas as pd
import gc


def build_stage1_features_batch(
    link_batch: pd.DataFrame,
    head_batch: pd.DataFrame,
    cross_batch: pd.DataFrame,
    topo: dict[int, list[int]],
) -> pd.DataFrame:
    """按订单批次处理，严格控制内存"""
    df = link_batch.copy()

    # 1. 必要列合并
    df = df.merge(head_batch[["order_id", "day", "slice_id"]],
                  on=["order_id", "day"], how="left")

    # 2. 出度
    deg_map = {int(lid): len(topo.get(int(lid), [])) for lid in df["link_id"].unique()}
    df["out_degree"] = df["link_id"].map(deg_map).fillna(0).astype("int16")

    # 3. 路径上下文（轻量版）
    grp = df.groupby(["order_id", "day"], sort=False)

    df["link_seq"] = grp.cumcount().astype("int16")
    df["link_count"] = grp["link_id"].transform("count").astype("int16")
    df["pos_ratio"] = (df["link_seq"] / df["link_count"].clip(lower=1)).astype("float32")

    df["delta_time"] = grp["link_time"].diff().fillna(0).astype("float32")
    df["delta_status"] = grp["link_current_status"].diff().fillna(0).astype("float32")

    # 轻量版上游平均（前5个link滚动平均，替代expanding）
    df["upstream_mean_status"] = (
        grp["link_current_status"]
        .rolling(window=5, min_periods=1).mean()
        .reset_index(level=[0,1], drop=True)
        .astype("float32")
    )

    df["wt"] = (df["link_time"] * df["link_ratio"]).astype("float32")
    df["cum_travel_time"] = grp["wt"].cumsum().astype("float32")

    # 4. 下游路口等待时间
    df["next_link_id"] = grp["link_id"].shift(-1)
    mask = df["next_link_id"].notna()
    df["cross_key"] = ""
    df.loc[mask, "cross_key"] = (
        df.loc[mask, "link_id"].astype(str) + "_" +
        df.loc[mask, "next_link_id"].astype("Int32").astype(str)
    )

    cross_rename = cross_batch[["order_id", "day", "cross_id", "cross_time"]].rename(
        columns={"cross_id": "cross_key", "cross_time": "downstream_cross_time"}
    )
    df = df.merge(cross_rename, on=["order_id", "day", "cross_key"], how="left")
    df["downstream_cross_time"] = df["downstream_cross_time"].fillna(0).astype("float32")

    # 5. 时空特征
    df["sin_slice"] = np.sin(2 * np.pi * df["slice_id"] / 288).astype("float32")
    df["cos_slice"] = np.cos(2 * np.pi * df["slice_id"] / 288).astype("float32")

    df["arrival_slice_est"] = (df["slice_id"] + (df["cum_travel_time"] / 300)).astype("int16")
    df["sin_arr_slice"] = np.sin(2 * np.pi * df["arrival_slice_est"] / 288).astype("float32")
    df["cos_arr_slice"] = np.cos(2 * np.pi * df["arrival_slice_est"] / 288).astype("float32")

    # 只保留最终需要的列（大幅减少内存）
    final_cols = STAGE1_FEATURE_COLS + [STAGE1_TARGET, "order_id", "day"]
    df = df[final_cols].copy()

    del grp
    gc.collect()
    return df


STAGE1_FEATURE_COLS = [
    "link_current_status", "link_time", "link_ratio", "out_degree",
    "pos_ratio", "link_seq", "link_count", "delta_time", "delta_status",
    "upstream_mean_status", "cum_travel_time", "downstream_cross_time",
    "sin_slice", "cos_slice", "sin_arr_slice", "cos_arr_slice",
]

STAGE1_TARGET = "link_arrival_status"