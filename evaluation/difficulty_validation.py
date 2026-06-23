"""
evaluation/difficulty_validation.py — 难度有效性验证（论文级）

包含：
    1. 单调性检验（Monotonicity Check）
    2. 校准曲线（Calibration Curve）
    3. 对比实验（Ablation Study）
    4. 极端天气分位数对比
    5. 可视化输出
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score
from config import FIGURES_DIR, RESULTS_DIR
from logger import get_logger

log = get_logger()


# ============================================================
# 1. 单调性检验
# ============================================================

def monotonicity_check(df: pd.DataFrame, score_col: str = 'difficulty', target_col: str = 'ata_over_eta'):
    log.info("\n" + "=" * 70)
    log.info("MONOTONICITY CHECK")
    log.info("=" * 70)

    if target_col not in df.columns:
        df['ata_over_eta'] = df['ata'] / df['simple_eta'].clip(lower=1)
        target_col = 'ata_over_eta'

    # ★ 防御：如果 unique 值太少，跳过
    if df[score_col].nunique() < 10:
        log.warning(f"{score_col} has fewer than 10 unique values. Skipping monotonicity.")
        return {}

    df['decile'] = pd.qcut(df[score_col], q=10, labels=False, duplicates='drop')

    # ★ 整理 agg_dict
    agg_dict = {
        target_col: ["mean", "std", "count"],
    }

    safety_cols = [
        "R1_max_p4",
        "R1_p4_p95",
        "R1_p4_exposure",
        "R1_p4_top10_mean",
        "R1_cong_exposure",
        "R2_reject_ratio",
    ]

    for col in safety_cols:
        if col in df.columns:
            agg_dict[col] = "mean"

    stats = df.groupby('decile').agg(agg_dict).round(4)

    decile_means = df.groupby("decile")[target_col].mean()
    rho, p_value = spearmanr(np.arange(len(decile_means)), decile_means.values)

    if len(decile_means) < 2:
        log.warning("Not enough deciles for Spearman correlation.")
        return {}

    log.info(f"Spearman Correlation (decile means): ρ={rho:.4f} (p={p_value:.4f})")
    log.info("\nDecile Statistics:")
    log.info(stats.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(range(len(decile_means)), decile_means, 'o-', color='#2E86AB', linewidth=2)
    ax.set_xlabel('Difficulty Decile (0=Easy, 9=Hard)')
    ax.set_ylabel(f'Mean {target_col}')
    ax.set_title(f'Monotonicity Check (ρ={rho:.3f})')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    r1_means = df.groupby('decile')['R1_max_p4'].mean().values
    ax.plot(range(len(r1_means)), r1_means, 'o-', color='#A23B72', linewidth=2)
    ax.set_xlabel('Difficulty Decile')
    ax.set_ylabel('Mean R1_max_p4 (Safety Risk)')
    ax.set_title('Safety Risk vs Difficulty')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    r2_means = df.groupby('decile')['R2_reject_ratio'].mean().values
    ax.plot(range(len(r2_means)), r2_means, 'o-', color='#F18F01', linewidth=2)
    ax.set_xlabel('Difficulty Decile')
    ax.set_ylabel('Mean R2_reject_ratio')
    ax.set_title('Model Uncertainty vs Difficulty')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, 'monotonicity_check.png'), dpi=150)
    plt.close()
    log.info(f"Monotonicity plot saved to {FIGURES_DIR}/monotonicity_check.png")
    # 构建可序列化的 stats
    stats_dict = {}
    for decile, row in stats.iterrows():
        decile = int(decile)
        row_dict = {}
        for col in stats.columns:
            if isinstance(col, tuple):
                key = f"{col[0]}_{col[1]}"
            else:
                key = str(col)
            value = row[col]
            if isinstance(value, (np.integer, np.floating)):
                value = float(value)
            row_dict[key] = value
        stats_dict[decile] = row_dict
    return {
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "p_value": float(p_value) if np.isfinite(p_value) else None,
        "n_deciles": int(len(decile_means)),
        "decile_stats": stats.reset_index().to_dict(orient="records"),
    }


# ============================================================
# 2. 校准曲线（修复后）
# ============================================================

def calibration_curve(
    df: pd.DataFrame,
    pred_col: str = "bti_score",
    n_bins: int = 20,
):
    """
    校准曲线：只校准效率风险 bti_score 与真实相对 ETA 偏差。

    target = (ata - simple_eta) / simple_eta
    pred_col = bti_score，当前 Stage 3 中应为同量纲相对偏差预测。
    """
    log.info("\n" + "=" * 70)
    log.info("CALIBRATION CURVE")
    log.info("=" * 70)

    required = {"ata", "simple_eta", pred_col}
    missing = required - set(df.columns)
    if missing:
        log.warning(f"Missing columns for calibration: {missing}. Skipping.")
        return {}

    tmp = df[["order_id", "ata", "simple_eta", pred_col]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    if tmp.empty:
        log.warning("No valid rows for calibration.")
        return {}

    tmp["_target_dev"] = (
        (tmp["ata"] - tmp["simple_eta"].clip(lower=1))
        / tmp["simple_eta"].clip(lower=1)
    ).clip(-0.5, 2.0)

    tmp["_pred"] = tmp[pred_col].clip(-0.5, 2.0)

    try:
        tmp["_bin"] = pd.qcut(tmp["_pred"], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        log.warning("qcut failed for calibration. Skipping.")
        return {}

    cal = (
        tmp.groupby("_bin")
        .agg(
            pred_mean=("_pred", "mean"),
            target_mean=("_target_dev", "mean"),
            count=("order_id", "count"),
        )
        .reset_index()
    )
    cal = cal[cal["count"] >= 30]

    if len(cal) < 2:
        log.warning("Not enough calibration bins.")
        return {}

    mae_cal = mean_absolute_error(cal["target_mean"], cal["pred_mean"])
    r2_cal = r2_score(cal["target_mean"], cal["pred_mean"])

    log.info(f"Calibration: MAE={mae_cal:.4f}, R²={r2_cal:.4f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(cal["pred_mean"], cal["target_mean"],
               s=cal["count"] / 10, alpha=0.6, color="#2E86AB")
    ax.plot([-0.5, 2.0], [-0.5, 2.0], "--", color="red", label="Perfect Calibration")
    ax.set_xlabel("Predicted Deviation")
    ax.set_ylabel("Actual Deviation")
    ax.set_title(f"Calibration Curve (R²={r2_cal:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "calibration_curve.png"), dpi=150)
    plt.close()

    return {
        "mae_cal": float(mae_cal),
        "r2_cal": float(r2_cal),
        "n_bins": int(len(cal)),
    }


# ============================================================
# 3. 对比实验（Ablation Study）
# ============================================================

def _safe_corr(a, b):
    """安全的相关系数计算，处理常数向量和 NaN"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan

    a = a[mask]
    b = b[mask]

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def build_safety_proxy_target(df: pd.DataFrame) -> np.ndarray:
    """
    构建新的 safety proxy target，替代 R1_max_p4

    使用 decile 表中有梯度的指标：
        R1_p4_p95: 25%
        R1_p4_exposure: 30%
        R1_p4_top10_mean: 20%
        R1_cong_exposure: 15%
        R2_reject_ratio: 10%
    """
    cols = {
        "R1_p4_p95": 0.25,
        "R1_p4_exposure": 0.30,
        "R1_p4_top10_mean": 0.20,
        "R1_cong_exposure": 0.15,
        "R2_reject_ratio": 0.10,
    }

    score = np.zeros(len(df), dtype="float64")
    weight_sum = 0.0

    for col, w in cols.items():
        if col in df.columns:
            x = df[col].fillna(0).values.astype("float64")
            # 鲁棒归一化
            lo, hi = np.nanquantile(x, [0.02, 0.98])
            if hi - lo > 1e-8:
                x = np.clip((x - lo) / (hi - lo), 0, 1)
                score += w * x
                weight_sum += w

    if weight_sum <= 0:
        # 如果没有新指标，回退到 R1_max_p4
        log.warning("No new safety indicators found, falling back to R1_max_p4")
        return df.get("R1_max_p4", pd.Series(0, index=df.index)).fillna(0).values

    return score / weight_sum

def _norm01_array(x):
    x = np.asarray(x, dtype="float64")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.nanquantile(x, [0.02, 0.98])
    if hi - lo <= 1e-8:
        return np.zeros_like(x, dtype="float64")

    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _get_score_col(df, candidates, fallback=None):
    for col in candidates:
        if col in df.columns:
            return df[col].fillna(0).values.astype("float64")

    if fallback is not None:
        return np.asarray(fallback, dtype="float64")

    return np.zeros(len(df), dtype="float64")


def ablation_study(df: pd.DataFrame, stage4_results=None) -> pd.DataFrame:
    log.info("\n" + "=" * 70)
    log.info("ABLATION STUDY")
    log.info("=" * 70)

    # 真实效率目标：ETA 偏差
    if "ata_over_eta" in df.columns:
        real_deviation = df["ata_over_eta"].fillna(1.0).values.astype("float64") - 1.0
    elif {"ata", "simple_eta"}.issubset(df.columns):
        real_deviation = (
            (df["ata"].values.astype("float64") - df["simple_eta"].values.astype("float64"))
            / np.maximum(df["simple_eta"].values.astype("float64"), 1.0)
        )
        real_deviation = np.clip(real_deviation, -0.5, 2.0)
    else:
        real_deviation = np.zeros(len(df), dtype="float64")

    # 新 safety proxy target
    safety_target = build_safety_proxy_target(df)

    # 旧 R1_max_p4 作为参考
    if "R1_max_p4" in df.columns:
        r1_max_p4 = df["R1_max_p4"].fillna(0).values.astype("float64")
    else:
        r1_max_p4 = np.zeros(len(df), dtype="float64")

    # 各策略分数
    rng = np.random.default_rng(42)

    eta_score = _get_score_col(
        df,
        ["eta_risk_score", "bti_score", "difficulty"],
        fallback=real_deviation,
    )

    safety_score = _get_score_col(
        df,
        ["safety_risk_score", "safety_score"],
        fallback=safety_target,
    )

    uncertainty_score = _get_score_col(
        df,
        ["uncertainty_score", "U1_path_entropy"],
        fallback=np.zeros(len(df), dtype="float64"),
    )

    composite_score = _get_score_col(
        df,
        ["av_unsuitability_score", "difficulty"],
        fallback=0.4 * _norm01_array(safety_score)
        + 0.35 * _norm01_array(uncertainty_score)
        + 0.25 * _norm01_array(eta_score),
    )

    strategies = {
        "Random": rng.random(len(df)),
        "Efficiency Only": _norm01_array(eta_score),
        "Safety Only": _norm01_array(safety_score),
        "Uncertainty Only": _norm01_array(uncertainty_score),
        "Current Composite": _norm01_array(composite_score),
    }

    rows = []
    for name, score in strategies.items():
        score = np.asarray(score, dtype="float64")

        rows.append({
            "Strategy": name,
            "Corr with Deviation": _safe_corr(score, real_deviation),
            "Corr with Safety Proxy": _safe_corr(score, safety_target),
            "Corr with R1_max_p4 (ref)": _safe_corr(score, r1_max_p4),
            "Score Std": float(np.nanstd(score)),
        })

    results = pd.DataFrame(rows)

    log.info("\nAblation Results:")
    log.info(results.to_string(index=False))

    if not results.empty:
        best_dev = results.loc[
            results["Corr with Deviation"].idxmax(), "Strategy"
        ]
        best_safety = results.loc[
            results["Corr with Safety Proxy"].idxmax(), "Strategy"
        ]

        composite_row = results[results["Strategy"] == "Current Composite"]
        if not composite_row.empty:
            comp_dev = float(composite_row["Corr with Deviation"].iloc[0])
            comp_safe = float(composite_row["Corr with Safety Proxy"].iloc[0])
        else:
            comp_dev = np.nan
            comp_safe = np.nan

        log.info("")
        log.info("Ablation Summary:")
        log.info(f"  Best deviation correlation: {best_dev}")
        log.info(f"  Best safety-proxy correlation: {best_safety}")
        log.info(
            f"  Current Composite: deviation_corr={comp_dev:.4f}, "
            f"safety_proxy_corr={comp_safe:.4f}"
        )
        log.info(
            "  Interpretation: Current Composite is evaluated as a compromise "
            "between efficiency, safety proxy, and uncertainty, not as a "
            "winner-takes-all baseline."
        )

    return results

def build_safety_proxy_target(df: pd.DataFrame) -> np.ndarray:
    """构建 safety proxy target，替代 R1_max_p4"""
    cols = {
        "R1_p4_p95": 0.25,
        "R1_p4_exposure": 0.30,
        "R1_p4_top10_mean": 0.20,
        "R1_cong_exposure": 0.15,
        "R2_reject_ratio": 0.10,
    }

    score = np.zeros(len(df), dtype="float64")
    weight_sum = 0.0

    for col, w in cols.items():
        if col in df.columns:
            x = df[col].fillna(0).values.astype("float64")
            lo, hi = np.nanquantile(x, [0.02, 0.98])
            if hi - lo > 1e-8:
                x = np.clip((x - lo) / (hi - lo), 0, 1)
                score += w * x
                weight_sum += w

    if weight_sum <= 0:
        log.warning("No new safety indicators found, falling back to R1_max_p4")
        return df.get("R1_max_p4", pd.Series(0, index=df.index)).fillna(0).values

    return score / weight_sum
# ============================================================
# 4. 完整验证报告
# ============================================================

def generate_validation_report(df: pd.DataFrame, stage4_results: dict = None):
    log.info("\n" + "=" * 70)
    log.info("DIFFICULTY VALIDATION REPORT")
    log.info("=" * 70)

    mono_results = monotonicity_check(df)
    cal_results = calibration_curve(df)
    ablation_results = ablation_study(df, stage4_results)

    # ★ 修复 summary
    log.info("\n" + "=" * 70)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 70)

    if mono_results:
        log.info(f"Monotonicity: Spearman ρ={mono_results.get('spearman_rho', 'N/A')}")
    if cal_results:
        log.info(f"Calibration: MAE={cal_results.get('mae_cal', 'N/A'):.4f}, R²={cal_results.get('r2_cal', 'N/A'):.4f}")

    # ★ 不再使用 "5/3 metrics"
    if isinstance(ablation_results, pd.DataFrame) and len(ablation_results) > 0:
        safety_col = (
            "Corr with Safety Proxy"
            if "Corr with Safety Proxy" in ablation_results.columns
            else "Corr with Safety"
        )
        best_dev = ablation_results.loc[ablation_results["Corr with Deviation"].idxmax(), "Strategy"]
        best_safety = ablation_results.loc[ablation_results[safety_col].idxmax(), "Strategy"]
        log.info("Ablation summary:")
        log.info(f"  best deviation correlation: {best_dev}")
        log.info(f"  best safety-proxy correlation: {best_safety}")
        log.info("  composite score is reported as a compromise indicator, not a winner-takes-all metric.")

    # ★ 清洗 ablation_results（如果是 DataFrame）
    if isinstance(ablation_results, pd.DataFrame):
        ablation_records = ablation_results.replace({np.nan: None}).to_dict(orient="records")
    else:
        ablation_records = ablation_results

    # ★ 清洗 mono_results 和 cal_results（确保 numpy 类型被转换）
    def _clean_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): _clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.replace({np.nan: None}).to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.replace({np.nan: None}).tolist()
        else:
            return obj

    report = {
        'monotonicity': _clean_for_json(mono_results),
        'calibration': _clean_for_json(cal_results),
        'ablation': _clean_for_json(ablation_records),
        'timestamp': pd.Timestamp.now().isoformat(),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'validation_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Validation report saved to {RESULTS_DIR}/validation_report.json")

    return report


# ============================================================
# 5. 极端天气分位数对比
# ============================================================

def extreme_weather_validation(df: pd.DataFrame):
    """验证极端天气下难度分数的分布差异"""
    log.info("\n" + "=" * 70)
    log.info("EXTREME WEATHER VALIDATION")
    log.info("=" * 70)

    if 'is_extreme_weather' not in df.columns:
        log.warning("'is_extreme_weather' column not found. Skipping.")
        return None

    extreme = df[df['is_extreme_weather'] == 1]
    normal = df[df['is_extreme_weather'] == 0]

    diff_cols = ['difficulty', 'bti_score', 'safety_score'] if 'bti_score' in df.columns else ['difficulty']
    results = {}
    for col in diff_cols:
        if col not in df.columns:
            continue
        results[col] = {
            'extreme_mean': extreme[col].mean() if len(extreme) > 0 else 0,
            'normal_mean': normal[col].mean() if len(normal) > 0 else 0,
            'extreme_median': extreme[col].median() if len(extreme) > 0 else 0,
            'normal_median': normal[col].median() if len(normal) > 0 else 0,
        }
        if len(extreme) > 0 and len(normal) > 0:
            log.info(f"{col}: Extreme={extreme[col].mean():.4f}, Normal={normal[col].mean():.4f}")
        else:
            log.warning(f"No extreme weather samples available for {col}.")

    return results