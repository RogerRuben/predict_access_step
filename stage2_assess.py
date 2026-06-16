"""
stage2_assess.py — 完整版（含 R1/R2 安全风险指标）

指标体系:
  确定性场景难度 D1-D11:
    D1  拥堵暴露比        ← pred_cong_prob
    D2  拥堵严重度        ← pred_omega
    D3  连续拥堵段        ← pred_cong_prob
    D4  状态突变          ← pred_status
    D5  拥堵-路口耦合      ← pred_omega + cross_time
    D6  复杂路口暴露      ← cross_time
    D7  路口决策频率      ← cross_count
    D8  路口时间占比      ← cross_time
    D9  拓扑复杂度        ← topology
    D10 高出度集中度      ← topology
    D11 夜间退化          ← slice_id

  不确定性预测风险 U1-U3:
    U1  路径预测熵        ← pred_entropy
    U2  高不确定性占比    ← pred_entropy
    U3  连续高不确定段    ← pred_entropy

  ★ 新增安全风险 R1-R2:
    R1  路径极端拥堵概率  ← max(pred_risk_prob) along path
    R2  reject 占比       ← pred_reject 时间加权占比
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
from logger import get_logger

log = get_logger()


# ================================================================
# 确定性场景难度 D1-D11
# ================================================================

def compute_deterministic(link_df, cross_df, topo, cross_time_threshold,
                          cross_global_mean=None):
    df = link_df.copy()
    df["wt"] = (df["link_time"] * df["link_ratio"]).astype("float32")

    # 路口聚合
    cross_agg = cross_df.groupby(["order_id", "day"], sort=False).agg(
        cross_time_sum=("cross_time", "sum"),
        cross_count=("cross_id", "size"),
        hard_cross_time=pd.NamedAgg(
            column="cross_time",
            aggfunc=lambda s: s[s > cross_time_threshold].sum(),
        ),
    )

    df["cong_wt"] = (df["wt"] * df["pred_cong_prob"]).astype("float32")
    df["sev_wt"]  = (df["wt"] * df["pred_omega"]).astype("float32")

    # 出度
    deg_map = {int(lid): len(topo.get(int(lid), []))
               for lid in df["link_id"].unique()}
    df["out_deg"]    = df["link_id"].map(deg_map).fillna(0).astype("int16")
    df["excess_deg"] = np.maximum(df["out_deg"] - DEGREE_BASELINE, 0)
    df["is_high_deg"] = (df["out_deg"] >= DEGREE_HIGH)

    if cross_global_mean is None:
        cross_global_mean = float(cross_df["cross_time"].mean())
    gmc = cross_global_mean + 1e-6

    df["coupling_score"] = (
        df["pred_omega"] * (df["downstream_cross_time"] / gmc)
    ).astype("float32")

    # 标量聚合
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

    merged["D1_cong_exposure"]  = np.where(total_time > 0, merged["cong_wt_sum"] / total_time, 0)
    merged["D2_cong_severity"]  = np.where(merged["wt_sum"] > 0, merged["sev_wt_sum"] / merged["wt_sum"], 0)
    merged["D5_coupling"]       = merged["coupling_sum"]
    merged["D6_hard_cross"]     = merged["hard_cross_time"]
    merged["D7_cross_freq"]     = np.where(total_time > 0, merged["cross_count"] / total_time, 0)
    merged["D8_cross_share"]    = np.where(total_time > 0, merged["cross_time_sum"] / total_time, 0)
    merged["D9_topo_complex"]   = merged["excess_deg_sum"]

    det_cols = [
        "D1_cong_exposure", "D2_cong_severity", "D5_coupling",
        "D6_hard_cross", "D7_cross_freq", "D8_cross_share", "D9_topo_complex",
    ]
    result = merged[det_cols].reset_index()

    # D4: 状态突变（向量化）
    df["_diff"] = df.groupby(["order_id", "day"], sort=False)["pred_status"].diff().abs()
    d4 = (
        df.groupby(["order_id", "day"], sort=False)["_diff"]
        .sum().fillna(0).rename("D4_status_jump").reset_index()
    )
    result = result.merge(d4, on=["order_id", "day"], how="left")

    # D3: 连续拥堵段
    df["_cong_flag"] = (df["pred_cong_prob"] > CONGESTION_PROB_THRESHOLD).astype("int8")
    df["_cong_wt"]   = (df["pred_cong_prob"] * df["wt"]).astype("float32")

    def _max_cong_run(grp):
        flags = grp["_cong_flag"].values
        vals  = grp["_cong_wt"].values
        mx = cur = 0.0
        for i in range(len(flags)):
            if flags[i]:
                cur += vals[i]
            else:
                mx = max(mx, cur); cur = 0.0
        return max(mx, cur)

    d3 = (
        df.groupby(["order_id", "day"], sort=False)
        .apply(_max_cong_run).rename("D3_cong_persist").reset_index()
    )
    result = result.merge(d3, on=["order_id", "day"], how="left")

    # D10: 高出度集中度
    def _max_deg_window(grp):
        flags = grp["is_high_deg"].values.astype("int8")
        w = CLUSTER_WINDOW_K
        if len(flags) < w:
            return int(flags.sum())
        cs = np.concatenate([[0], np.cumsum(flags)])
        return int((cs[w:] - cs[:-w]).max())

    d10 = (
        df.groupby(["order_id", "day"], sort=False)
        .apply(_max_deg_window).rename("D10_topo_cluster").reset_index()
    )
    result = result.merge(d10, on=["order_id", "day"], how="left")

    for c in ["D3_cong_persist", "D4_status_jump", "D10_topo_cluster"]:
        result[c] = result[c].fillna(0)

    return result


# ================================================================
# 不确定性预测风险 U1-U3
# ================================================================

def compute_uncertainty(link_df):
    df = link_df.copy()
    df["wt"]         = (df["link_time"] * df["link_ratio"]).astype("float32")
    df["entropy_wt"] = (df["pred_entropy"] * df["wt"]).astype("float32")
    df["is_high_unc"] = df["pred_entropy"] > ENTROPY_HIGH_THRESHOLD
    df["high_unc_wt"] = (df["wt"] * df["is_high_unc"].astype("float32")).astype("float32")

    agg = df.groupby(["order_id", "day"], sort=False).agg(
        wt_sum=("wt", "sum"),
        entropy_wt_sum=("entropy_wt", "sum"),
        high_unc_wt_sum=("high_unc_wt", "sum"),
    )
    agg["U1_path_entropy"]    = np.where(agg["wt_sum"] > 0, agg["entropy_wt_sum"] / agg["wt_sum"], 0)
    agg["U2_high_unc_ratio"]  = np.where(agg["wt_sum"] > 0, agg["high_unc_wt_sum"] / agg["wt_sum"], 0)
    result = agg[["U1_path_entropy", "U2_high_unc_ratio"]].reset_index()

    # U3: 连续高不确定性段
    df["_unc_flag"] = df["is_high_unc"].astype("int8")

    def _max_unc_run(grp):
        flags = grp["_unc_flag"].values
        wts   = grp["wt"].values
        mx = cur = 0.0
        for i in range(len(flags)):
            if flags[i]:
                cur += wts[i]
            else:
                mx = max(mx, cur); cur = 0.0
        return max(mx, cur)

    u3 = (
        df.groupby(["order_id", "day"], sort=False)
        .apply(_max_unc_run).rename("U3_unc_persist").reset_index()
    )
    result = result.merge(u3, on=["order_id", "day"], how="left")
    result["U3_unc_persist"] = result["U3_unc_persist"].fillna(0)

    return result


# ================================================================
# ★ 新增安全风险 R1-R2
# ================================================================

def compute_safety_risk(link_df):
    """
    R1: 路径上最大极端拥堵概率 max(pred_risk_prob)
        ← 直接捕捉"路径上最危险的一个点"
        ← pred_risk_prob = P(s4) = p_cong * p_severe

    R2: 路径上 reject 标记的 link 时间占比
        ← 捕捉"模型对这条路径有多不确定"
        ← pred_reject = 信息熵超过阈值的 link
    """
    df = link_df.copy()
    df["wt"] = (df["link_time"] * df["link_ratio"]).astype("float32")

    # R1: max P(s4) along path
    if "pred_risk_prob" in df.columns:
        r1 = (
            df.groupby(["order_id", "day"], sort=False)["pred_risk_prob"]
            .max().rename("R1_max_p4").reset_index()
        )
    else:
        # 兼容旧模型：用 pred_p4 替代
        if "pred_p4" in df.columns:
            r1 = (
                df.groupby(["order_id", "day"], sort=False)["pred_p4"]
                .max().rename("R1_max_p4").reset_index()
            )
        else:
            r1 = (
                df[["order_id", "day"]].drop_duplicates()
                .assign(R1_max_p4=0.0)
            )

    # R2: reject ratio (时间加权)
    if "pred_reject" in df.columns:
        df["reject_wt"] = df["wt"] * df["pred_reject"].astype("float32")
        r2_num = df.groupby(["order_id", "day"], sort=False)["reject_wt"].sum()
        wt_sum = df.groupby(["order_id", "day"], sort=False)["wt"].sum()
        r2_ratio = (r2_num / wt_sum.clip(lower=1e-6)).rename("R2_reject_ratio").reset_index()
    else:
        r2_ratio = (
            df[["order_id", "day"]].drop_duplicates()
            .assign(R2_reject_ratio=0.0)
        )

    result = r1.merge(r2_ratio, on=["order_id", "day"], how="outer")
    result["R1_max_p4"]       = result["R1_max_p4"].fillna(0)
    result["R2_reject_ratio"] = result["R2_reject_ratio"].fillna(0)

    return result


# ================================================================
# 夜间感知退化 D11
# ================================================================

def compute_night(head_df):
    df = head_df[["order_id", "day", "slice_id"]].copy()
    conditions = [
        (df["slice_id"] >= 264) | (df["slice_id"] <= 71),
        (df["slice_id"].between(72, 83)) | (df["slice_id"].between(228, 263)),
    ]
    df["D11_night"] = np.select(conditions, [1.0, 0.3], default=0.0)
    return df[["order_id", "day", "D11_night"]]


# ================================================================
# 特征列定义（Stage 3 使用）
# ================================================================

STAGE2_DET_COLS = [
    "D1_cong_exposure", "D2_cong_severity", "D3_cong_persist",
    "D4_status_jump", "D5_coupling",
    "D6_hard_cross", "D7_cross_freq", "D8_cross_share",
    "D9_topo_complex", "D10_topo_cluster", "D11_night",
]

STAGE2_UNC_COLS = [
    "U1_path_entropy", "U2_high_unc_ratio", "U3_unc_persist",
]

STAGE2_RISK_COLS = [
    "R1_max_p4", "R2_reject_ratio",
]

STAGE2_ALL_COLS = STAGE2_DET_COLS + STAGE2_UNC_COLS + STAGE2_RISK_COLS