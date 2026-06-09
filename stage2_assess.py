"""
stage2_assess.py
----------------
Stage 2: Dual-Component Difficulty Assessment
确定性场景难度 + 不确定性预测风险
"""

import numpy as np
import pandas as pd
from config import (
    CONGESTION_PROB_THRESHOLD,
    ENTROPY_HIGH_THRESHOLD,
    CROSS_TIME_QUANTILE,
    DEGREE_BASELINE,
    DEGREE_HIGH,
    CLUSTER_WINDOW_K,
)


def _max_consecutive_weighted(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    """mask 为 True 的连续段中 values*weights 的最大累计值。"""
    max_run = cur = 0.0
    for i in range(len(mask)):
        if mask[i]:
            cur += values[i] * weights[i]
        else:
            max_run = max(max_run, cur)
            cur = 0.0
    return max(max_run, cur)


def _max_consecutive_time(wt: np.ndarray, mask: np.ndarray) -> float:
    """mask 为 True 的连续段中 wt 的最大累计值。"""
    max_run = cur = 0.0
    for i in range(len(mask)):
        if mask[i]:
            cur += wt[i]
        else:
            max_run = max(max_run, cur)
            cur = 0.0
    return max(max_run, cur)


def _sliding_window_max(flags: np.ndarray, window: int) -> int:
    if len(flags) < window:
        return int(flags.sum())
    cs = np.concatenate([[0], np.cumsum(flags)])
    return int((cs[window:] - cs[:-window]).max())


# ================================================================
#  确定性场景难度
# ================================================================

def compute_deterministic(
    link_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    topo: dict[int, list[int]],
    cross_time_threshold: float,
) -> pd.DataFrame:
    """
    基于预测的 arrival_status 概率分布计算确定性场景难度。
    要求 link_df 已包含 pred_cong_prob, pred_omega, pred_status 列。
    """
    df = link_df.copy()
    df["wt"] = df["link_time"] * df["link_ratio"]

    # ---------- 路口聚合 ----------
    cross_agg = cross_df.groupby(["order_id", "day"], sort=False).agg(
        cross_time_sum=("cross_time", "sum"),
        cross_count=("cross_id", "size"),
        hard_cross_time=pd.NamedAgg(
            column="cross_time",
            aggfunc=lambda s: s[s > cross_time_threshold].sum(),
        ),
    )

    # ---------- Link 级聚合 ----------
    df["cong_wt"] = df["wt"] * df["pred_cong_prob"]
    df["sev_wt"] = df["wt"] * df["pred_omega"]

    # 出度
    deg_map = {int(lid): len(topo.get(int(lid), [])) for lid in df["link_id"].unique()}
    df["out_deg"] = df["link_id"].map(deg_map).fillna(0).astype("int16")
    df["excess_deg"] = np.maximum(df["out_deg"] - DEGREE_BASELINE, 0)
    df["is_high_deg"] = (df["out_deg"] >= DEGREE_HIGH)

    # --- 拥堵-路口耦合 ---
    global_mean_cross = float(cross_df["cross_time"].mean()) + 1e-6
    df["coupling_score"] = df["pred_omega"] * (df["downstream_cross_time"] / global_mean_cross)

    # ---------- 按订单聚合 ----------
    link_agg = df.groupby(["order_id", "day"], sort=False).agg(
        wt_sum=("wt", "sum"),
        cong_wt_sum=("cong_wt", "sum"),
        sev_wt_sum=("sev_wt", "sum"),
        excess_deg_sum=("excess_deg", "sum"),
        coupling_sum=("coupling_score", "sum"),
    )

    merged = link_agg.join(cross_agg, how="left")
    for c in ["cross_time_sum", "cross_count", "hard_cross_time"]:
        merged[c] = merged[c].fillna(0)

    total_time = merged["wt_sum"] + merged["cross_time_sum"]

    # D1: 预期拥堵暴露比
    merged["D1_cong_exposure"] = np.where(total_time > 0, merged["cong_wt_sum"] / total_time, 0)
    # D2: 预期拥堵严重度
    merged["D2_cong_severity"] = np.where(merged["wt_sum"] > 0, merged["sev_wt_sum"] / merged["wt_sum"], 0)
    # D5: 拥堵-路口耦合
    merged["D5_coupling"] = merged["coupling_sum"]
    # D6: 复杂路口暴露时间
    merged["D6_hard_cross"] = merged["hard_cross_time"]
    # D7: 路口决策频率
    merged["D7_cross_freq"] = np.where(total_time > 0, merged["cross_count"] / total_time, 0)
    # D8: 路口时间占比
    merged["D8_cross_share"] = np.where(total_time > 0, merged["cross_time_sum"] / total_time, 0)
    # D9: 超额出度累计
    merged["D9_topo_complex"] = merged["excess_deg_sum"]

    det_cols = [
        "D1_cong_exposure", "D2_cong_severity", "D5_coupling",
        "D6_hard_cross", "D7_cross_freq", "D8_cross_share", "D9_topo_complex",
    ]

    result = merged[det_cols].reset_index()

    # ---------- 需要保序的逐订单指标 ----------
    persist_records = []
    cluster_records = []
    jump_records = []

    for (oid, day), grp in df.groupby(["order_id", "day"], sort=False):
        wt_arr = grp["wt"].values
        cong_prob = grp["pred_cong_prob"].values
        pred_s = grp["pred_status"].values
        is_high = grp["is_high_deg"].values

        # D3: 预期最长连续拥堵段
        cong_mask = cong_prob > CONGESTION_PROB_THRESHOLD
        d3 = _max_consecutive_weighted(cong_prob, wt_arr, cong_mask)
        persist_records.append((oid, day, d3))

        # D4: 预期状态突变
        if len(pred_s) > 1:
            d4 = float(np.sum(np.abs(np.diff(pred_s))))
        else:
            d4 = 0.0
        jump_records.append((oid, day, d4))

        # D10: 高出度集中度
        d10 = _sliding_window_max(is_high, CLUSTER_WINDOW_K)
        cluster_records.append((oid, day, d10))

    persist_df = pd.DataFrame(persist_records, columns=["order_id", "day", "D3_cong_persist"])
    jump_df = pd.DataFrame(jump_records, columns=["order_id", "day", "D4_status_jump"])
    cluster_df = pd.DataFrame(cluster_records, columns=["order_id", "day", "D10_topo_cluster"])

    result = result.merge(persist_df, on=["order_id", "day"], how="left")
    result = result.merge(jump_df, on=["order_id", "day"], how="left")
    result = result.merge(cluster_df, on=["order_id", "day"], how="left")

    return result


# ================================================================
#  不确定性预测风险
# ================================================================

def compute_uncertainty(link_df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 Stage 1 预测概率分布的熵计算不确定性指标。
    要求 link_df 已包含 pred_entropy 列。
    """
    df = link_df.copy()
    df["wt"] = df["link_time"] * df["link_ratio"]
    df["entropy_wt"] = df["pred_entropy"] * df["wt"]
    df["is_high_unc"] = df["pred_entropy"] > ENTROPY_HIGH_THRESHOLD
    df["high_unc_wt"] = df["wt"] * df["is_high_unc"].astype("float32")

    # U1: 路径级预测熵
    # U2: 高不确定性路段时间占比
    agg = df.groupby(["order_id", "day"], sort=False).agg(
        wt_sum=("wt", "sum"),
        entropy_wt_sum=("entropy_wt", "sum"),
        high_unc_wt_sum=("high_unc_wt", "sum"),
    )
    agg["U1_path_entropy"] = np.where(agg["wt_sum"] > 0, agg["entropy_wt_sum"] / agg["wt_sum"], 0)
    agg["U2_high_unc_ratio"] = np.where(agg["wt_sum"] > 0, agg["high_unc_wt_sum"] / agg["wt_sum"], 0)

    result = agg[["U1_path_entropy", "U2_high_unc_ratio"]].reset_index()

    # U3: 最长连续高不确定性段
    u3_records = []
    for (oid, day), grp in df.groupby(["order_id", "day"], sort=False):
        wt_arr = grp["wt"].values
        mask = grp["is_high_unc"].values
        u3 = _max_consecutive_time(wt_arr, mask)
        u3_records.append((oid, day, u3))

    u3_df = pd.DataFrame(u3_records, columns=["order_id", "day", "U3_unc_persist"])
    result = result.merge(u3_df, on=["order_id", "day"], how="left")

    return result


# ================================================================
#  夜间感知退化 (不依赖预测)
# ================================================================

def compute_night(head_df: pd.DataFrame) -> pd.DataFrame:
    df = head_df[["order_id", "day", "slice_id"]].copy()
    conditions = [
        (df["slice_id"] >= 264) | (df["slice_id"] <= 71),
        (df["slice_id"].between(72, 83)) | (df["slice_id"].between(228, 263)),
    ]
    df["D11_night"] = np.select(conditions, [1.0, 0.3], default=0.0)
    return df[["order_id", "day", "D11_night"]]


# ================================================================
#  合并全部 Stage 2 指标
# ================================================================

# 所有 Stage 2 特征列名（供 Stage 3 使用）
STAGE2_DET_COLS = [
    "D1_cong_exposure", "D2_cong_severity", "D3_cong_persist",
    "D4_status_jump", "D5_coupling",
    "D6_hard_cross", "D7_cross_freq", "D8_cross_share",
    "D9_topo_complex", "D10_topo_cluster",
    "D11_night",
]

STAGE2_UNC_COLS = [
    "U1_path_entropy", "U2_high_unc_ratio", "U3_unc_persist",
]

STAGE2_ALL_COLS = STAGE2_DET_COLS + STAGE2_UNC_COLS