"""
feature_eng.py — 移除 link_ratio 特征，增加拥堵相关衍生特征
"""
import numpy as np
import pandas as pd
import gc
from logger import get_logger

log = get_logger()

# ★ 移除 link_ratio（重要性=0），增加拥堵上下文特征
STAGE1_FEATURE_COLS = [
    "link_current_status", "link_time", "out_degree",
    "pos_ratio", "link_seq", "link_count",
    "delta_time", "delta_status",
    "upstream_mean_status", "cum_travel_time",
    "downstream_cross_time",
    "sin_slice", "cos_slice",
    "sin_arr_slice", "cos_arr_slice",
    # ★ 新增特征
    "upstream_cong_ratio",      # 上游拥堵占比
    "is_status_unknown",        # current_status 是否为 0（未知）
    "status_x_time",            # 状态与时间的交互
]

STAGE1_TARGET = "link_arrival_status"
STAGE2_EXTRA_COLS = ["link_id", "link_time", "link_ratio", "downstream_cross_time"]


def build_stage1_features_batch(
    link_batch: pd.DataFrame,
    head_batch: pd.DataFrame,
    cross_batch: pd.DataFrame,
    topo: dict,
    keep_extra_for_stage2: bool = False,
) -> pd.DataFrame:
    df = link_batch.copy()

    # 1. 合并 slice_id
    df = df.merge(head_batch[["order_id", "day", "slice_id"]],
                  on=["order_id", "day"], how="left")

    # 2. 出度
    unique_ids = df["link_id"].unique()
    deg_map = {int(lid): len(topo.get(int(lid), [])) for lid in unique_ids}
    df["out_degree"] = df["link_id"].map(deg_map).fillna(0).astype("int16")

    # 3. ★ status=0 处理：先记录再替换
    df["is_status_unknown"] = (df["link_current_status"] == 0).astype("float32")
    df["link_current_status"] = df["link_current_status"].replace(0, np.nan).astype("float32")

    # 4. 路径序列上下文
    grp = df.groupby(["order_id", "day"], sort=False)

    df["link_seq"] = grp.cumcount().astype("int16")
    df["link_count"] = grp["link_id"].transform("count").astype("int16")
    df["pos_ratio"] = (df["link_seq"] / df["link_count"].clip(lower=1)).astype("float32")

    df["delta_time"] = grp["link_time"].diff().fillna(0).astype("float32")
    df["delta_status"] = grp["link_current_status"].diff().fillna(0).astype("float32")

    df["upstream_mean_status"] = (
        grp["link_current_status"]
        .rolling(window=5, min_periods=1).mean()
        .reset_index(level=[0, 1], drop=True)
        .astype("float32")
    )

    # ★ 新增：上游拥堵占比（前面 link 中 status≥3 的比例）
    df["_is_cong"] = (df["link_current_status"] >= 3).astype("float32")
    df["upstream_cong_ratio"] = (
        grp["_is_cong"]
        .expanding().mean()
        .shift(1)
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype("float32")
    )
    df.drop(columns=["_is_cong"], inplace=True)

    df["wt"] = (df["link_time"] * df["link_ratio"]).astype("float32")
    df["cum_travel_time"] = grp["wt"].cumsum().astype("float32")

    # ★ 新增：状态×时间交互
    df["status_x_time"] = (df["link_current_status"].fillna(0) * df["link_time"]).astype("float32")

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
    keep = list(set(STAGE1_FEATURE_COLS + [STAGE1_TARGET, "order_id", "day"]))
    if keep_extra_for_stage2:
        keep = list(set(keep + STAGE2_EXTRA_COLS))
    df = df[[c for c in keep if c in df.columns]].copy()

    del grp
    gc.collect()
    return df