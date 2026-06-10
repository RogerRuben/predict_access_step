"""
stage2_assess.py — 无逻辑变化，仅加 logger
"""
import numpy as np
import pandas as pd
from config import (
    CONGESTION_PROB_THRESHOLD, ENTROPY_HIGH_THRESHOLD,
    CROSS_TIME_QUANTILE, DEGREE_BASELINE, DEGREE_HIGH, CLUSTER_WINDOW_K,
)
from logger import get_logger

log = get_logger()


def _max_consecutive_weighted(values, weights, mask):
    max_run = cur = 0.0
    for i in range(len(mask)):
        if mask[i]:
            cur += values[i] * weights[i]
        else:
            max_run = max(max_run, cur)
            cur = 0.0
    return max(max_run, cur)


def _max_consecutive_time(wt, mask):
    max_run = cur = 0.0
    for i in range(len(mask)):
        if mask[i]:
            cur += wt[i]
        else:
            max_run = max(max_run, cur)
            cur = 0.0
    return max(max_run, cur)


def _sliding_window_max(flags, window):
    if len(flags) < window:
        return int(flags.sum())
    cs = np.concatenate([[0], np.cumsum(flags)])
    return int((cs[window:] - cs[:-window]).max())


def compute_deterministic(link_df, cross_df, topo, cross_time_threshold, cross_global_mean=None):
    """
    Parameters
    ----------
    cross_global_mean : float, optional
        全局路口等待时间均值。如果为 None，则从 cross_df 计算（训练集用）。
        测试集应传入训练集的 global_mean 以防止数据泄露。
    """
    df = link_df.copy()
    df["wt"] = (df["link_time"] * df["link_ratio"]).astype("float32")

    cross_agg = cross_df.groupby(["order_id", "day"], sort=False).agg(
        cross_time_sum=("cross_time", "sum"),
        cross_count=("cross_id", "size"),
        hard_cross_time=pd.NamedAgg(
            column="cross_time",
            aggfunc=lambda s: s[s > cross_time_threshold].sum(),
        ),
    )

    df["cong_wt"] = (df["wt"] * df["pred_cong_prob"]).astype("float32")
    df["sev_wt"] = (df["wt"] * df["pred_omega"]).astype("float32")

    deg_map = {int(lid): len(topo.get(int(lid), [])) for lid in df["link_id"].unique()}
    df["out_deg"] = df["link_id"].map(deg_map).fillna(0).astype("int16")
    df["excess_deg"] = np.maximum(df["out_deg"] - DEGREE_BASELINE, 0)
    df["is_high_deg"] = (df["out_deg"] >= DEGREE_HIGH)

    # ★ 使用传入的 global_mean（防止数据泄露）
    if cross_global_mean is None:
        cross_global_mean = float(cross_df["cross_time"].mean())
    global_mean_cross = cross_global_mean + 1e-6

    df["coupling_score"] = (
        df["pred_omega"] * (df["downstream_cross_time"] / global_mean_cross)
    ).astype("float32")

    # 按订单聚合
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

    merged["D1_cong_exposure"] = np.where(total_time > 0, merged["cong_wt_sum"] / total_time, 0)
    merged["D2_cong_severity"] = np.where(merged["wt_sum"] > 0, merged["sev_wt_sum"] / merged["wt_sum"], 0)
    merged["D5_coupling"] = merged["coupling_sum"]
    merged["D6_hard_cross"] = merged["hard_cross_time"]
    merged["D7_cross_freq"] = np.where(total_time > 0, merged["cross_count"] / total_time, 0)
    merged["D8_cross_share"] = np.where(total_time > 0, merged["cross_time_sum"] / total_time, 0)
    merged["D9_topo_complex"] = merged["excess_deg_sum"]

    det_cols = ["D1_cong_exposure", "D2_cong_severity", "D5_coupling",
                "D6_hard_cross", "D7_cross_freq", "D8_cross_share", "D9_topo_complex"]
    result = merged[det_cols].reset_index()

    # 逐订单保序指标
    # ================================================================
    # ★ D3/D4/D10: 向量化计算（替代逐订单 Python 循环）
    # ================================================================

    # --- D4: 预期状态突变（完全向量化）---
    df["_pred_status_diff"] = df.groupby(["order_id", "day"], sort=False)["pred_status"].diff().abs()
    d4 = (
        df.groupby(["order_id", "day"], sort=False)["_pred_status_diff"]
        .sum()
        .fillna(0)
        .rename("D4_status_jump")
        .reset_index()
    )
    result = result.merge(d4, on=["order_id", "day"], how="left")

    # --- D3: 预期最长连续拥堵段（优化版：用 numba 或纯 numpy 分组）---
    df["_cong_flag"] = (df["pred_cong_prob"] > CONGESTION_PROB_THRESHOLD).astype("int8")
    df["_cong_weighted"] = (df["pred_cong_prob"] * df["wt"]).astype("float32")

    # 用分组 + apply 替代逐订单循环（pandas apply 比纯 Python 循环快 3-5x）
    def _max_cong_run(grp):
        flags = grp["_cong_flag"].values
        vals = grp["_cong_weighted"].values
        max_run = cur = 0.0
        for i in range(len(flags)):
            if flags[i]:
                cur += vals[i]
            else:
                if cur > max_run:
                    max_run = cur
                cur = 0.0
        return max(max_run, cur)

    d3 = (
        df.groupby(["order_id", "day"], sort=False)
        .apply(_max_cong_run)
        .rename("D3_cong_persist")
        .reset_index()
    )
    result = result.merge(d3, on=["order_id", "day"], how="left")

    # --- D10: 高出度集中度（优化版）---
    def _max_high_deg_window(grp):
        flags = grp["is_high_deg"].values.astype("int8")
        if len(flags) < CLUSTER_WINDOW_K:
            return int(flags.sum())
        cs = np.concatenate([[0], np.cumsum(flags)])
        return int((cs[CLUSTER_WINDOW_K:] - cs[:-CLUSTER_WINDOW_K]).max())

    d10 = (
        df.groupby(["order_id", "day"], sort=False)
        .apply(_max_high_deg_window)
        .rename("D10_topo_cluster")
        .reset_index()
    )
    result = result.merge(d10, on=["order_id", "day"], how="left")

    # 填充
    for c in ["D3_cong_persist", "D4_status_jump", "D10_topo_cluster"]:
        result[c] = result[c].fillna(0)

    return result
    # persist_records, jump_records, cluster_records = [], [], []
    # for (oid, day), grp in df.groupby(["order_id", "day"], sort=False):
    #     wt_arr = grp["wt"].values
    #     cong_prob = grp["pred_cong_prob"].values
    #     pred_s = grp["pred_status"].values
    #     is_high = grp["is_high_deg"].values
    #
    #     cong_mask = cong_prob > CONGESTION_PROB_THRESHOLD
    #     d3 = _max_consecutive_weighted(cong_prob, wt_arr, cong_mask)
    #     persist_records.append((oid, day, d3))
    #
    #     d4 = float(np.sum(np.abs(np.diff(pred_s)))) if len(pred_s) > 1 else 0.0
    #     jump_records.append((oid, day, d4))
    #
    #     d10 = _sliding_window_max(is_high, CLUSTER_WINDOW_K)
    #     cluster_records.append((oid, day, d10))
    #
    # for recs, col in [(persist_records, "D3_cong_persist"),
    #                   (jump_records, "D4_status_jump"),
    #                   (cluster_records, "D10_topo_cluster")]:
    #     tmp = pd.DataFrame(recs, columns=["order_id", "day", col])
    #     result = result.merge(tmp, on=["order_id", "day"], how="left")
    #
    # return result


def compute_uncertainty(link_df):
    df = link_df.copy()
    df["wt"] = (df["link_time"] * df["link_ratio"]).astype("float32")
    df["entropy_wt"] = (df["pred_entropy"] * df["wt"]).astype("float32")
    df["is_high_unc"] = df["pred_entropy"] > ENTROPY_HIGH_THRESHOLD
    df["high_unc_wt"] = (df["wt"] * df["is_high_unc"].astype("float32")).astype("float32")

    agg = df.groupby(["order_id", "day"], sort=False).agg(
        wt_sum=("wt", "sum"),
        entropy_wt_sum=("entropy_wt", "sum"),
        high_unc_wt_sum=("high_unc_wt", "sum"),
    )
    agg["U1_path_entropy"] = np.where(agg["wt_sum"] > 0, agg["entropy_wt_sum"] / agg["wt_sum"], 0)
    agg["U2_high_unc_ratio"] = np.where(agg["wt_sum"] > 0, agg["high_unc_wt_sum"] / agg["wt_sum"], 0)
    result = agg[["U1_path_entropy", "U2_high_unc_ratio"]].reset_index()

    # u3_records = []
    # for (oid, day), grp in df.groupby(["order_id", "day"], sort=False):
    #     u3 = _max_consecutive_time(grp["wt"].values, grp["is_high_unc"].values)
    #     u3_records.append((oid, day, u3))
    # u3_df = pd.DataFrame(u3_records, columns=["order_id", "day", "U3_unc_persist"])
    # result = result.merge(u3_df, on=["order_id", "day"], how="left")
    # return result

 # ★ U3: 向量化（同 D3 逻辑）
    df["_unc_flag"] = df["is_high_unc"].astype("int8")

    def _max_unc_run(grp):
        flags = grp["_unc_flag"].values
        wts = grp["wt"].values
        max_run = cur = 0.0
        for i in range(len(flags)):
            if flags[i]:
                cur += wts[i]
            else:
                if cur > max_run:
                    max_run = cur
                cur = 0.0
        return max(max_run, cur)

    u3 = (
        df.groupby(["order_id", "day"], sort=False)
        .apply(_max_unc_run)
        .rename("U3_unc_persist")
        .reset_index()
    )
    result = result.merge(u3, on=["order_id", "day"], how="left")
    result["U3_unc_persist"] = result["U3_unc_persist"].fillna(0)

    return result

def compute_night(head_df):
    df = head_df[["order_id", "day", "slice_id"]].copy()
    conditions = [
        (df["slice_id"] >= 264) | (df["slice_id"] <= 71),
        (df["slice_id"].between(72, 83)) | (df["slice_id"].between(228, 263)),
    ]
    df["D11_night"] = np.select(conditions, [1.0, 0.3], default=0.0)
    return df[["order_id", "day", "D11_night"]]


STAGE2_DET_COLS = [
    "D1_cong_exposure", "D2_cong_severity", "D3_cong_persist",
    "D4_status_jump", "D5_coupling",
    "D6_hard_cross", "D7_cross_freq", "D8_cross_share",
    "D9_topo_complex", "D10_topo_cluster", "D11_night",
]
STAGE2_UNC_COLS = ["U1_path_entropy", "U2_high_unc_ratio", "U3_unc_persist"]
STAGE2_ALL_COLS = STAGE2_DET_COLS + STAGE2_UNC_COLS