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
from collections import defaultdict
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




# ============================================================
# 天气严重程度有序映射（值越大越恶劣）
# ============================================================
WEATHER_SEVERITY_MAP = {
    'cloudy': 0,
    'showers': 1,
    'moderate rain': 2,
    'heavy rain': 3,
    'rainstorm': 4,
}
EXTREME_WEATHER_SET = {'heavy rain', 'rainstorm'}

# ============================================================
# Stage 2 全部输出列（含新增特征）
# ============================================================
# ============================================================
# Stage 2 完整列清单（所有可能输出的列）
# ============================================================
STAGE2_ALL_COLS = [
    # 确定性指标
    'D1_cong_exposure', 'D2_cong_severity', 'D3_cong_persist',
    'D4_status_jump', 'D5_coupling', 'D6_hard_cross',
    'D7_cross_freq', 'D8_cross_share', 'D9_topo_complex',
    'D10_topo_cluster', 'D11_night',
    # 不确定性指标
    'U1_path_entropy', 'U2_high_unc_ratio', 'U3_unc_persist',
    # 安全风险指标（旧 + 新）
    "R1_max_p4",
    "R1_p4_p95",
    "R1_p4_top10_mean",
    "R1_p4_exposure",
    "R1_cong_exposure",
    "R1_p4_tail_mass_030",
    "R1_p4_tail_mass_020",
    "R1_p4_std",
    "R2_reject_ratio",
    # Markov 链特征
    'D12_transition_entropy', 'D13_risk_exposure',
    'D14_first_hitting_time', 'D15_stationary_risk',
    # 天气特征
    'weather_severity', 'temp_avg', 'temp_range',
    'is_extreme_weather', 'is_high_temp', 'is_low_temp',

]



STAGE3_REALIZED_TARGET_COLS = [
    "target_actual_s4_any",
    "target_actual_s4_exposure",
    "target_actual_cong_exposure",
    "target_actual_max_status",
    "target_serious_underestimate",
    "target_s4_underestimate",
    "target_under_score",
    "target_path_nll",
    "target_path_len",
]

log = get_logger()


def _top_frac_mean(x, frac=0.10):
    arr = np.asarray(x, dtype="float32")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(arr) * frac)))
    return float(np.sort(arr)[-k:].mean())

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

    d3_slim = df[["order_id", "day", "_cong_flag", "_cong_wt"]].copy()

    d3 = (
        d3_slim.groupby(["order_id", "day"], sort=False)
        .apply(_max_cong_run)
        .rename("D3_cong_persist")
        .reset_index()
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

    d10_slim = df[["order_id", "day", "is_high_deg"]].copy()

    d10 = (
        d10_slim.groupby(["order_id", "day"], sort=False)
        .apply(_max_deg_window)
        .rename("D10_topo_cluster")
        .reset_index()
    )
    result = result.merge(d10, on=["order_id", "day"], how="left")

    for c in ["D3_cong_persist", "D4_status_jump", "D10_topo_cluster"]:
        result[c] = result[c].fillna(0)

    return result


# ================================================================
# 不确定性预测风险 U1-U3
# ================================================================

def compute_uncertainty(link_df):
    """
    计算不确定性指标 U1/U2/U3。

    只取必要列，避免 pandas 在宽表上 groupby.apply 时复制 30+ 列。
    """
    # 只取必要列
    df = link_df[
        ["order_id", "day", "link_time", "link_ratio", "pred_entropy"]
    ].copy()

    df["wt"] = (df["link_time"] * df["link_ratio"]).astype("float32")
    df["entropy_wt"] = (df["pred_entropy"] * df["wt"]).astype("float32")
    df["_unc_flag"] = (df["pred_entropy"] > ENTROPY_HIGH_THRESHOLD).astype("bool")
    df["high_unc_wt"] = (df["wt"] * df["_unc_flag"].astype("float32")).astype("float32")

    agg = df.groupby(["order_id", "day"], sort=False).agg(
        wt_sum=("wt", "sum"),
        entropy_wt_sum=("entropy_wt", "sum"),
        high_unc_wt_sum=("high_unc_wt", "sum"),
    )

    agg["U1_path_entropy"] = np.where(
        agg["wt_sum"] > 0,
        agg["entropy_wt_sum"] / agg["wt_sum"],
        0.0,
    )
    agg["U2_high_unc_ratio"] = np.where(
        agg["wt_sum"] > 0,
        agg["high_unc_wt_sum"] / agg["wt_sum"],
        0.0,
    )

    result = agg[["U1_path_entropy", "U2_high_unc_ratio"]].reset_index()

    # U3: 连续高不确定性段（窄表 apply）
    slim = df[["order_id", "day", "_unc_flag", "wt"]].copy()

    def _max_unc_run(grp):
        flags = grp["_unc_flag"].values
        wts = grp["wt"].values
        mx = cur = 0.0
        for i in range(len(flags)):
            if flags[i]:
                cur += wts[i]
            else:
                mx = max(mx, cur)
                cur = 0.0
        return max(mx, cur)

    u3 = (
        slim.groupby(["order_id", "day"], sort=False)
        .apply(_max_unc_run)
        .rename("U3_unc_persist")
        .reset_index()
    )

    result = result.merge(u3, on=["order_id", "day"], how="left")
    result["U3_unc_persist"] = result["U3_unc_persist"].fillna(0).astype("float32")

    return result


# ================================================================
# ★ 新增安全风险 R1-R2
# ================================================================

def compute_safety_risk(link_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算安全风险指标（分布型版本）

    新增指标：
        R1_p4_p95: p4 的 95% 分位数
        R1_p4_top10_mean: p4 最高的 10% 的均值
        R1_p4_exposure: 时间加权 p4 暴露
        R1_cong_exposure: 时间加权拥堵暴露
        R1_p4_tail_mass_030: p4 > 0.30 的比例
        R1_p4_tail_mass_020: p4 > 0.20 的比例
        R1_p4_std: p4 的标准差
    """
    df = link_df.copy()

    if "pred_p4" not in df.columns:
        raise ValueError("compute_safety_risk requires pred_p4.")

    df["pred_p4"] = df["pred_p4"].fillna(0).clip(0.0, 1.0)

    if "pred_cong_prob" not in df.columns:
        if "pred_p3" in df.columns:
            df["pred_cong_prob"] = df["pred_p3"] + df["pred_p4"]
        else:
            df["pred_cong_prob"] = df["pred_p4"]
    df["pred_cong_prob"] = df["pred_cong_prob"].fillna(0).clip(0.0, 1.0)

    wt = df.get("link_time", pd.Series(1.0, index=df.index)).fillna(1.0).clip(lower=1.0)
    if "downstream_cross_time" in df.columns:
        wt = wt + df["downstream_cross_time"].fillna(0).clip(lower=0.0)
    df["_risk_wt"] = wt.astype("float32")

    rows = []
    for (oid, day), g in df.groupby(["order_id", "day"], sort=False):
        p4 = g["pred_p4"].values.astype("float32")
        cong = g["pred_cong_prob"].values.astype("float32")
        w = g["_risk_wt"].values.astype("float32")
        wsum = float(np.maximum(w.sum(), 1e-6))

        rows.append({
            "order_id": oid,
            "day": day,
            "R1_max_p4": float(np.nanmax(p4)) if len(p4) else 0.0,
            "R1_p4_p95": float(np.nanpercentile(p4, 95)) if len(p4) else 0.0,
            "R1_p4_top10_mean": _top_frac_mean(p4, 0.10),
            "R1_p4_exposure": float(np.nansum(p4 * w) / wsum),
            "R1_cong_exposure": float(np.nansum(cong * w) / wsum),
            "R1_p4_tail_mass_030": float(np.nanmean(p4 > 0.30)) if len(p4) else 0.0,
            "R1_p4_tail_mass_020": float(np.nanmean(p4 > 0.20)) if len(p4) else 0.0,
            "R1_p4_std": float(np.nanstd(p4)) if len(p4) else 0.0,
            "R2_reject_ratio": float(g.get("pred_reject", pd.Series(False, index=g.index)).mean()),
        })

    return pd.DataFrame(rows)


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


def compute_markov_features(link_df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 Markov 链理论计算路径风险特征。

    学术依据：
        将路径路况序列视为有限状态离散时间 Markov 链，
        状态空间 S = {1, 2, 3, 4} (s1~s4)。

    输出特征：
        D12_transition_entropy: 状态转移熵，反映路况波动程度
        D13_risk_exposure: 风险暴露积分 Σ r(s_t)·Δt_t
        D14_first_hitting_time: 首次进入拥堵状态 (≥3) 的时间
        D15_stationary_risk: 平稳分布下的拥堵概率
    """
    df = link_df.copy()

    # 定义风险函数 r(s): 状态越拥堵，风险越高
    def risk_function(s):
        # s 为 1~4，映射到 [0, 1]
        return (s - 1) / 3.0

    df['risk_weight'] = df['pred_status'].apply(risk_function).astype('float32')
    df['wt'] = (df['link_time'] + df['link_ratio']).astype('float32')

    # ---- D13: 风险暴露积分 ----
    # 在每个 link 上，风险 = risk_weight * wt
    df['risk_exposure'] = (df['risk_weight'] * df['wt']).astype('float32')
    risk_exposure = df.groupby(['order_id', 'day'])['risk_exposure'].sum().rename('D13_risk_exposure').reset_index()

    # ---- D14: 首次恶化时间 ----
    # 定义"恶化"为状态 >= 3 (拥堵)
    def first_hitting_time(grp):
        statuses = grp['pred_status'].values
        cum_times = grp['cum_travel_time'].values
        for i, s in enumerate(statuses):
            if s >= 3:
                return cum_times[i]
        return cum_times[-1] if len(cum_times) > 0 else 0.0

    first_hit = df.groupby(['order_id', 'day']).apply(first_hitting_time).rename('D14_first_hitting_time').reset_index()

    # ---- D12: 状态转移熵 ----
    # 计算每个订单内状态转移矩阵的熵
    def transition_entropy(grp):
        statuses = grp['pred_status'].values
        if len(statuses) < 2:
            return 0.0
        # 统计转移频次
        trans_count = defaultdict(int)
        for i in range(len(statuses) - 1):
            trans_count[(statuses[i], statuses[i + 1])] += 1
        total = sum(trans_count.values())
        # 计算熵
        entropy = 0.0
        for count in trans_count.values():
            p = count / total
            entropy -= p * np.log(p + 1e-10)
        return entropy

    trans_entropy = df.groupby(['order_id', 'day']).apply(transition_entropy).rename(
        'D12_transition_entropy').reset_index()

    # ---- D15: 平稳分布风险 ----
    # 估计该路径的平稳分布，计算拥堵状态 (≥3) 的概率
    def stationary_risk(grp):
        statuses = grp['pred_status'].values
        if len(statuses) == 0:
            return 0.0
        # 用经验分布近似平稳分布
        counts = np.bincount(statuses.astype(int), minlength=5)[1:5]  # 1~4
        if counts.sum() == 0:
            return 0.0
        stationary = counts / counts.sum()
        # 拥堵概率 = P(s3) + P(s4)
        return stationary[2] + stationary[3]

    stat_risk = df.groupby(['order_id', 'day']).apply(stationary_risk).rename('D15_stationary_risk').reset_index()

    # ---- 合并 ----
    result = risk_exposure.merge(first_hit, on=['order_id', 'day'], how='outer')
    result = result.merge(trans_entropy, on=['order_id', 'day'], how='outer')
    result = result.merge(stat_risk, on=['order_id', 'day'], how='outer')

    # 填充缺失值
    for col in ['D13_risk_exposure', 'D14_first_hitting_time', 'D12_transition_entropy', 'D15_stationary_risk']:
        result[col] = result[col].fillna(0)

    return result


# ============================================================
# ★ 新增：天气特征
# ============================================================

def compute_weather_features(head_batch: pd.DataFrame) -> pd.DataFrame:
    """
    从 head 数据中提取天气特征（有序编码）。

    学术依据：
        天气严重程度是有序类别（Ordinal），而非无序类别（Nominal）。
        使用有序编码保留"暴雨 > 大雨 > 中雨 > 阵雨 > 多云"的单调约束。
    """
    df = head_batch[['order_id', 'day', 'weather', 'hightemp', 'lowtemp']].copy()

    # 1. 天气严重程度有序编码
    df['weather_severity'] = df['weather'].map(WEATHER_SEVERITY_MAP).fillna(0).astype(np.float32)

    # 2. 极端天气标志（用于 Known-Group 校准，不直接进模型）
    df['is_extreme_weather'] = df['weather'].isin(EXTREME_WEATHER_SET).astype(np.float32)

    # 3. 温度特征
    df['temp_avg'] = ((df['hightemp'] + df['lowtemp']) / 2).astype(np.float32)
    df['temp_range'] = (df['hightemp'] - df['lowtemp']).astype(np.float32)

    # 4. 极端温度标志
    df['is_high_temp'] = (df['hightemp'] > 35).astype(np.float32)
    df['is_low_temp'] = (df['lowtemp'] < 0).astype(np.float32)

    return df[['order_id', 'day', 'weather_severity', 'is_extreme_weather',
               'temp_avg', 'temp_range', 'is_high_temp', 'is_low_temp']]
##
import numpy as np
import pandas as pd


def build_realized_stage3_targets(link_level_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct order-level realized targets for Stage3-v2.

    Required:
      order_id, day, link_arrival_status

    Optional but strongly recommended:
      pred_p1, pred_p2, pred_p3, pred_p4, pred_status

    Output:
      one row per (order_id, day)
    """

    df = link_level_df.copy()

    if "order_id" not in df.columns or "day" not in df.columns:
        raise ValueError("link_level_df must contain order_id and day.")

    actual_col = None
    for c in ["link_arrival_status", "actual_status", "y_true", "status_true"]:
        if c in df.columns:
            actual_col = c
            break

    if actual_col is None:
        raise ValueError(
            "Cannot build realized targets: missing link_arrival_status / actual_status."
        )

    y = pd.to_numeric(df[actual_col], errors="coerce")

    # 兼容 0/1/2/3 和 1/2/3/4 两种编码
    valid = y.notna()
    if valid.sum() == 0:
        raise ValueError("No valid actual status values.")

    y_valid_min = int(y[valid].min())
    y_valid_max = int(y[valid].max())

    if y_valid_min >= 0 and y_valid_max <= 3:
        y_status = y + 1
    else:
        y_status = y

    y_status = y_status.clip(lower=1, upper=4).astype("float32")
    df["_actual_status_1to4"] = y_status

    # ---- predicted probabilities ----
    pcols = ["pred_p1", "pred_p2", "pred_p3", "pred_p4"]
    has_probs = all(c in df.columns for c in pcols)

    if has_probs:
        p = df[pcols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype("float64")
        p = np.clip(p, 1e-8, 1.0)
        p = p / p.sum(axis=1, keepdims=True).clip(min=1e-8)

        df["_pred_expected_status"] = (
            1.0 * p[:, 0] +
            2.0 * p[:, 1] +
            3.0 * p[:, 2] +
            4.0 * p[:, 3]
        ).astype("float32")

        df["_pred_status_1to4"] = (np.argmax(p, axis=1) + 1).astype("int16")

        actual_idx = (df["_actual_status_1to4"].astype(int).values - 1).clip(0, 3)
        df["_link_nll"] = (-np.log(p[np.arange(len(df)), actual_idx].clip(min=1e-8))).astype("float32")
    else:
        df["_pred_expected_status"] = np.nan
        df["_link_nll"] = np.nan

        if "pred_status" in df.columns:
            pred_status = pd.to_numeric(df["pred_status"], errors="coerce")
            if pred_status.min() >= 0 and pred_status.max() <= 3:
                pred_status = pred_status + 1
            df["_pred_status_1to4"] = pred_status.clip(1, 4)
        else:
            df["_pred_status_1to4"] = np.nan

    # ---- realized severe exposure ----
    df["_actual_s4"] = (df["_actual_status_1to4"] >= 4).astype("float32")
    df["_actual_cong"] = (df["_actual_status_1to4"] >= 3).astype("float32")

    # ---- underestimation targets ----
    if df["_pred_expected_status"].notna().any():
        df["_under_gap"] = np.maximum(
            df["_actual_status_1to4"] - df["_pred_expected_status"],
            0.0,
        ).astype("float32")
    else:
        df["_under_gap"] = np.nan

    df["_serious_underestimate"] = (
        (df["_actual_status_1to4"] >= 4) &
        (df["_pred_status_1to4"].notna()) &
        (df["_pred_status_1to4"] <= 2)
    ).astype("float32")

    df["_s4_underestimate"] = (
        (df["_actual_status_1to4"] >= 4) &
        (df["_pred_status_1to4"].notna()) &
        (df["_pred_status_1to4"] < 4)
    ).astype("float32")

    agg = df.groupby(["order_id", "day"], sort=False).agg(
        target_actual_s4_any=("_actual_s4", "max"),
        target_actual_s4_exposure=("_actual_s4", "mean"),
        target_actual_cong_exposure=("_actual_cong", "mean"),
        target_actual_max_status=("_actual_status_1to4", "max"),
        target_serious_underestimate=("_serious_underestimate", "max"),
        target_s4_underestimate=("_s4_underestimate", "max"),
        target_under_score=("_under_gap", "mean"),
        target_path_nll=("_link_nll", "mean"),
        target_path_len=("_actual_status_1to4", "count"),
    ).reset_index()

    # 避免全 NaN 影响后续模型
    for c in [
        "target_under_score",
        "target_path_nll",
    ]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0.0)

    return agg
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
STAGE2_WEATHER_COLS = [
    # ★ 新增天气特征
    'weather_severity', 'temp_avg', 'temp_range',
    'is_extreme_weather', 'is_high_temp', 'is_low_temp',
]

# STAGE2_ALL_COLS = STAGE2_DET_COLS + STAGE2_UNC_COLS + STAGE2_RISK_COLS + STAGE2_WEATHER_COLS