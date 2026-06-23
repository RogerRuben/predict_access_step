import numpy as np
import pandas as pd
import gc
from logger import get_logger

log = get_logger()

# ============================================================
# 特征列定义
# ============================================================
STAGE1_FEATURE_COLS = [
    # ---- 链路级特征 ----
    "link_id",  # 链路ID（离散特征，不标准化）
    "link_time",  # 平均通过时间
    "link_ratio",  # 位置指示器（头尾<1，中间=1），非速度系数
    "link_current_status",  # 上车时路况（核心特征，0=未知，1~4=有效值）
    "out_degree",  # 拓扑出度

    # ---- 序列上下文 ----
    "pos_ratio",  # 序列相对位置
    "link_seq",  # 序列绝对位置
    "link_count",  # 序列总长度
    "delta_time",  # 时间差分
    "delta_status",  # 状态差分
    "upstream_mean_status",  # 上游平均状态
    "upstream_cong_ratio",  # 上游拥堵占比

    # ---- 时空累积 ----
    "cum_travel_time",  # 累积行程时间（含路口等待）
    "downstream_cross_time",  # 下游路口等待

    # ---- 时间特征 ----
    "slice_id",  # 出发时间片（离散）
    "sin_slice",  # 出发时间周期编码
    "cos_slice",
    "arrival_slice_est",  # 到达时间片（离散）
    "sin_arr_slice",
    "cos_arr_slice",

    # ---- 行程级特征（在每个link上重复） ----
    "distance_km",  # 归一化里程 (distance/1000)
    "static_speed",  # 静态速度 (distance / simple_eta)

    # ---- 衍生特征 ----
    "is_status_unknown",  # link_current_status 是否为 0（未知）
    "status_x_time",  # 状态与时间的交互
]

STAGE1_TARGET = "link_arrival_status"

STAGE2_EXTRACOLS = ["link_id", "link_time", "link_ratio", "downstream_cross_time"]


def build_stage1_features_batch(
        link_batch: pd.DataFrame,
        head_batch: pd.DataFrame,
        cross_batch: pd.DataFrame,
        topo: dict,
        keep_extra_for_stage2: bool = False,
) -> pd.DataFrame:
    """
    构建 Stage 1 特征矩阵。

    核心设计:
        1. link_current_status 保留 0 原样（0 代表未知），不替换为 NaN
        2. cum_travel_time 包含路口等待时间 (downstream_cross_time)
        3. 增加行程级特征 distance_km 和 static_speed
        4. 增加离散时间片 slice_id 和 arrival_slice_est
    """
    df = link_batch.copy()

    # ============================================================
    # 1. 合并 head 信息
    # ============================================================
    df = df.merge(
        head_batch[["order_id", "day", "slice_id", "distance", "simple_eta"]],
        on=["order_id", "day"],
        how="left"
    )

    # 1b. 衍生行程级特征
    df["distance_km"] = (df["distance"] / 1000.0).astype("float32")
    df["static_speed"] = (df["distance"] / df["simple_eta"].clip(lower=1)).astype("float32")

    # ============================================================
    # 2. 出度（拓扑复杂度）
    # ============================================================
    unique_ids = df["link_id"].unique()
    deg_map = {int(link): len(topo.get(int(link), [])) for link in unique_ids}
    df["out_degree"] = df["link_id"].map(deg_map).fillna(0).astype("int16")

    # ============================================================
    # 3. link_current_status 处理：保留 0 原样
    # ============================================================
    # ★ 关键修改：不将 0 替换为 NaN，保留 0 作为"未知"的语义
    df["is_status_unknown"] = (df["link_current_status"] == 0).astype("float32")
    # 直接转为 float32，保留 0,1,2,3,4 原值
    df["link_current_status"] = df["link_current_status"].astype("float32")

    # ============================================================
    # 4. 路径序列上下文
    # ============================================================
    grp = df.groupby(["order_id", "day"], sort=False)
    df["link_seq"] = grp.cumcount().astype("int16")
    df["link_count"] = grp["link_id"].transform("count").astype("int16")
    df["pos_ratio"] = (df["link_seq"] / df["link_count"].clip(lower=1)).astype("float32")
    df["delta_time"] = grp["link_time"].diff().fillna(0).astype("float32")
    df["delta_status"] = grp["link_current_status"].diff().fillna(0).astype("float32")
    df["upstream_mean_status"] = (
        grp["link_current_status"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .astype("float32")
    )

    # 上游拥堵占比（当前 link 之前 status >= 3 的比例）
    df["_is_cong"] = (df["link_current_status"] >= 3).astype("float32")
    df["upstream_cong_ratio"] = (
        grp["_is_cong"]
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype("float32")
    )
    df.drop(columns=["_is_cong"], inplace=True)

    # 状态与时间交互
    df["status_x_time"] = (df["link_current_status"] * df["link_time"]).astype("float32")

    # ============================================================
    # 5. 下游路口等待时间
    # ============================================================
    df["next_link_id"] = grp["link_id"].shift(-1)
    mask = df["next_link_id"].notna()
    df["cross_key"] = ""
    if mask.any():
        df.loc[mask, "cross_key"] = (
                df.loc[mask, "link_id"].astype(str) + "_"
                + df.loc[mask, "next_link_id"].astype("int32").astype(str)
        )
    cross_rename = cross_batch[["order_id", "day", "cross_id", "cross_time"]].rename(
        columns={"cross_id": "cross_key", "cross_time": "downstream_cross_time"}
    )
    df = df.merge(cross_rename, on=["order_id", "day", "cross_key"], how="left")
    df["downstream_cross_time"] = df["downstream_cross_time"].fillna(0).astype("float32")

    # ============================================================
    # 6. 累积行程时间（包含路口等待）
    # ============================================================
    # ★ 重新创建分组对象，确保能访问最新列
    grp = df.groupby(["order_id", "day"], sort=False)
    # 有效耗时 = link_time + link_ratio + 下游路口等待
    df["wt_with_cross"] = (df["link_time"] + df["link_ratio"] + df["downstream_cross_time"]).astype("float32")
    df["cum_travel_time"] = grp["wt_with_cross"].cumsum().astype("float32")

    # 保留原始 wt（用于其他场景，如 Stage 2 权重计算）
    df["wt"] = (df["link_time"] + df["link_ratio"]).astype("float32")

    # ============================================================
    # 7. 周期特征与到达时间片
    # ============================================================
    # 出发时间周期编码
    df["sin_slice"] = np.sin(2 * np.pi * df["slice_id"] / 288).astype("float32")
    df["cos_slice"] = np.cos(2 * np.pi * df["slice_id"] / 288).astype("float32")

    # 精确计算到达时间片（取整，模 288）
    df["arrival_slice_est"] = (df["slice_id"] + (df["cum_travel_time"] / 300)).round().astype("int16") % 288
    df["sin_arr_slice"] = np.sin(2 * np.pi * df["arrival_slice_est"] / 288).astype("float32")
    df["cos_arr_slice"] = np.cos(2 * np.pi * df["arrival_slice_est"] / 288).astype("float32")

    # ============================================================
    # 8. 裁剪列
    # ============================================================
    keep = list(set(STAGE1_FEATURE_COLS + [STAGE1_TARGET, "order_id", "day"]))
    if keep_extra_for_stage2:
        keep = list(set(keep + STAGE2_EXTRACOLS))
    df = df[[c for c in keep if c in df.columns]].copy()

    del grp
    gc.collect()
    return df