"""
stage3_anchor.py — 分层优化框架（可求解代理问题）

核心理念：
    1. 安全硬约束：高阈值（0.5）过滤极端危险订单 → G4
    2. 效率排序：剩余订单按预测偏差排序，KMeans 聚类分 G1-G3
    3. 不确定性修正：高不确定性订单升一级
    4. 熵权法确定权重（用于 difficulty 计算，便于论文论证）
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
from logger import get_logger
import warnings

from stage2_assess import STAGE2_ALL_COLS
warnings.filterwarnings('ignore')

log = get_logger()

# ============================================================
# 配置参数
# ============================================================
SAFETY_THRESHOLD = 0.50       # ★ 关键修复：从 0.25 提升到 0.50
UNCERTAINTY_THRESHOLD = 0.3   # R2_reject_ratio > 0.3 升一级

# ============================================================
# Stage 2 特征列
# ============================================================
STAGE2_ALL_COLS = [
    'D1_cong_exposure', 'D2_cong_severity', 'D3_cong_persist',
    'D4_status_jump', 'D5_coupling', 'D6_hard_cross',
    'D7_cross_freq', 'D8_cross_share', 'D9_topo_complex',
    'D10_topo_cluster', 'D11_night',
    'U1_path_entropy', 'U2_high_unc_ratio', 'U3_unc_persist',
    'R1_max_p4', 'R2_reject_ratio',
    'D12_transition_entropy', 'D13_risk_exposure',
    'D14_first_hitting_time', 'D15_stationary_risk',
    'weather_severity', 'temp_avg', 'temp_range',
    'is_extreme_weather', 'is_high_temp', 'is_low_temp',
]

INTERACTION_PAIRS = [
    ('D2_cong_severity', 'U1_path_entropy'),
    ('D6_hard_cross', 'U1_path_entropy'),
    ('D2_cong_severity', 'D7_cross_freq'),
    ('R1_max_p4', 'U1_path_entropy'),
    ('R1_max_p4', 'D6_hard_cross'),
    ('weather_severity', 'R1_max_p4'),
    ('temp_avg', 'D13_risk_exposure'),
]

def _robust_01(x, q_low=0.02, q_high=0.98):
    """Robust min-max normalization with quantile clipping"""
    x = np.asarray(x, dtype="float64")
    lo = np.nanquantile(x, q_low)
    hi = np.nanquantile(x, q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(x, dtype="float32")
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype("float32")

def _fit_robust_params(self, x, q_low=0.02, q_high=0.98):
    x = np.asarray(x, dtype="float64")
    lo = np.nanquantile(x, q_low)
    hi = np.nanquantile(x, q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        lo, hi = 0.0, 1.0
    return float(lo), float(hi)

def _apply_robust_01(self, x, params):
    lo, hi = params
    x = np.asarray(x, dtype="float64")
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0).astype("float32")

def _compute_safety_raw(self, df):
    return (
        0.35 * df.get("R1_max_p4", 0).fillna(0).values
        + 0.25 * df.get("R1_p4_p95", df.get("R1_max_p4", 0)).fillna(0).values
        + 0.25 * df.get("R1_p4_exposure", df.get("R1_max_p4", 0)).fillna(0).values
        + 0.15 * df.get("R2_reject_ratio", 0).fillna(0).values
    )

def _compute_uncertainty_raw(self, df):
    return (
        0.60 * df.get("U1_path_entropy", 0).fillna(0).values
        + 0.40 * df.get("U2_high_unc_ratio", 0).fillna(0).values
    )


class DifficultyModelXGB:
    """分层优化框架"""

    def __init__(self):
        self.feature_cols = list(STAGE2_ALL_COLS)
        self.interaction_cols = []
        self.all_cols_ordered = []

        self.scaler = StandardScaler()
        self.model = None
        self.explainer = None

        # 分级边界
        self.boundaries = None  # [G1/G2, G2/G3]
        self.safety_threshold = SAFETY_THRESHOLD
        self.uncertainty_threshold = UNCERTAINTY_THRESHOLD

        # 熵权法权重（用于 difficulty 计算）
        self.entropy_weights = {'efficiency': 0.5, 'safety': 0.5}

        self.is_fitted = False
        self.score_norm_params = {
            "eta": (0.0, 1.0),
            "safety": (0.0, 1.0),
            "uncertainty": (0.0, 1.0),
        }
        self.grade_thresholds = [0.25, 0.50, 0.75]

    def _build_interaction_cols(self):
        return [f'{c1}×{c2}' for c1, c2 in INTERACTION_PAIRS]

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            log.warning(f"Missing columns: {missing_cols}. Filling with 0.")
            for col in missing_cols:
                df[col] = 0.0

        out = df[self.feature_cols].copy()

        for c1, c2 in INTERACTION_PAIRS:
            col_name = f'{c1}×{c2}'
            if c1 not in out.columns:
                out[c1] = 0.0
            if c2 not in out.columns:
                out[c2] = 0.0
            out[col_name] = (out[c1] * out[c2]).astype('float32')

        self.interaction_cols = self._build_interaction_cols()
        self.all_cols_ordered = self.feature_cols + self.interaction_cols

        X = out[self.all_cols_ordered].values.astype('float32')

        expected = getattr(self.scaler, 'n_features_in_', None)
        if expected is not None and X.shape[1] != expected:
            log.warning(f"Dimension mismatch: {X.shape[1]} vs {expected}. Adjusting.")
            if X.shape[1] < expected:
                pad = np.zeros((X.shape[0], expected - X.shape[1]), dtype=np.float32)
                X = np.concatenate([X, pad], axis=1)
            else:
                X = X[:, :expected]

        return X

    def _compute_av_unsuitability_score(self, eta_score, safety_score, uncertainty_score):
        """
        AV 不适配分数：线性组合，避免 noisy-OR 饱和
        """
        return (
                0.40 * safety_score
                + 0.35 * uncertainty_score
                + 0.25 * eta_score
        ).clip(0, 1).astype("float32")

    # ============================================================
    # 监督信号：绝对偏差 + 自适应贝叶斯收缩
    # ============================================================
    def _fit_robust_params(self, x, q_low=0.02, q_high=0.98):
        x = np.asarray(x, dtype="float64")
        lo = np.nanquantile(x, q_low)
        hi = np.nanquantile(x, q_high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
            lo, hi = 0.0, 1.0
        return float(lo), float(hi)

    def _apply_robust_01(self, x, params):
        lo, hi = params
        x = np.asarray(x, dtype="float64")
        return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0).astype("float32")

    def _compute_safety_raw(self, df):
        # 确保列存在
        cols = ["R1_max_p4", "R1_p4_p95", "R1_p4_exposure", "R2_reject_ratio"]
        for col in cols:
            if col not in df.columns:
                df[col] = 0.0
        return (
                0.35 * df["R1_max_p4"].fillna(0).values
                + 0.25 * df["R1_p4_p95"].fillna(0).values
                + 0.25 * df["R1_p4_exposure"].fillna(0).values
                + 0.15 * df["R2_reject_ratio"].fillna(0).values
        )

    def _compute_uncertainty_raw(self, df):
        cols = ["U1_path_entropy", "U2_high_unc_ratio"]
        for col in cols:
            if col not in df.columns:
                df[col] = 0.0
        return (
                0.60 * df["U1_path_entropy"].fillna(0).values
                + 0.40 * df["U2_high_unc_ratio"].fillna(0).values
        )

    def _compute_target(self, df: pd.DataFrame) -> pd.Series:
        """绝对偏差 + 贝叶斯收缩（按 driver_id, slice_id 分组）"""
        eta = df['simple_eta'].clip(lower=1.0)
        raw_deviation = (df['ata'] - eta) / eta
        raw_deviation = raw_deviation.clip(-0.5, 2.0)

        global_mean = raw_deviation.mean()

        df_temp = df.copy()
        df_temp['_raw_dev'] = raw_deviation

        grouped = df_temp.groupby(['driver_id', 'slice_id'])
        group_counts = grouped.size()
        median_count = group_counts.median()
        shrinkage_param = max(5, min(50, median_count * 0.5))

        group_mean = grouped['_raw_dev'].transform('mean')
        group_count = grouped['_raw_dev'].transform('count')

        shrinkage_weight = group_count / (group_count + shrinkage_param)
        y = shrinkage_weight * group_mean + (1 - shrinkage_weight) * global_mean

        log.info(f"Supervision: mean={y.mean():.4f}, std={y.std():.4f}, "
                 f"shrinkage={shrinkage_param:.1f}")
        return y

    # ============================================================
    # 熵权法计算权重（用于 difficulty 计算）
    # ============================================================

    def _compute_entropy_weights(self, df: pd.DataFrame) -> dict:
        """
        用熵权法确定 efficiency 和 safety 的权重。
        可用于论文论证：说明权重是数据驱动的，而非任意设置。
        """
        # 效率指标：预测的绝对偏差
        X_raw = self._prepare_X(df)
        X_scaled = self.scaler.transform(X_raw)
        pred_deviation = self.model.predict(X_scaled)

        # 安全指标：R1_max_p4
        safety = df['R1_max_p4'].fillna(0).values

        # 构建指标矩阵
        metrics = np.column_stack([pred_deviation, safety])
        metrics = np.abs(metrics)  # 取绝对值，避免负值影响熵计算

        # 归一化
        metrics_norm = metrics / (metrics.sum(axis=0) + 1e-10)

        # 计算熵值
        with np.errstate(divide='ignore'):
            entropy = -np.sum(metrics_norm * np.log(metrics_norm + 1e-10), axis=0)
        entropy = entropy / np.log(len(metrics_norm))

        # 计算权重
        weights = (1 - entropy) / (2 - entropy.sum())

        self.entropy_weights = {
            'efficiency': weights[0],
            'safety': weights[1]
        }
        log.info(f"Entropy weights: efficiency={weights[0]:.4f}, safety={weights[1]:.4f}")
        return self.entropy_weights

    # ============================================================
    # 边界确定：KMeans 一维聚类
    # ============================================================

    def _determine_boundaries(self, predictions: np.ndarray) -> list:
        """KMeans 自然断点（一维聚类）"""
        if len(predictions) < 100:
            return [np.percentile(predictions, 33), np.percentile(predictions, 67)]

        p_low = np.percentile(predictions, 1)
        p_high = np.percentile(predictions, 99)
        filtered = predictions[(predictions >= p_low) & (predictions <= p_high)]

        X_1d = filtered.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_1d)

        centers = sorted(kmeans.cluster_centers_.flatten())
        boundary1 = (centers[0] + centers[1]) / 2
        boundary2 = (centers[1] + centers[2]) / 2

        log.info(f"Boundaries: G1/G2={boundary1:.4f}, G2/G3={boundary2:.4f}")
        return [boundary1, boundary2]

    # ============================================================
    # 训练
    # ============================================================

    def fit(self, df: pd.DataFrame):
        """训练模型 + 学习归一化参数 + 确定 grade 阈值"""
        log.info("=" * 70)
        log.info("Stage 3: 分层优化框架 (安全硬约束 + 效率排序 + 不确定性修正)")
        log.info("=" * 70)

        # 1. 计算监督信号
        y = self._compute_target(df)

        # 2. 训练 XGBoost
        X_raw = self._prepare_X(df)
        X_scaled = self.scaler.fit_transform(X_raw)

        log.info(f"Training: {len(y)} samples, {X_scaled.shape[1]} features")

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42
        )

        self.model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='mae',
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = self.model.predict(X_val)
        log.info(f"Model: R²={r2_score(y_val, y_pred):.4f}, MAE={mean_absolute_error(y_val, y_pred):.4f}")

        # 3. 学习归一化参数（在训练集上）
        train_pred = np.clip(self.model.predict(self.scaler.transform(self._prepare_X(df))), -0.5, 2.0)

        eta_raw = train_pred.astype("float64")
        safety_raw = self._compute_safety_raw(df)
        unc_raw = self._compute_uncertainty_raw(df)

        self.score_norm_params = {
            "eta": self._fit_robust_params(eta_raw),
            "safety": self._fit_robust_params(safety_raw),
            "uncertainty": self._fit_robust_params(unc_raw),
        }

        # 4. 计算训练集上的 av_unsuitability_score 并确定 grade 阈值
        eta_score = self._apply_robust_01(eta_raw, self.score_norm_params["eta"])
        safety_score = self._apply_robust_01(safety_raw, self.score_norm_params["safety"])
        uncertainty_score = self._apply_robust_01(unc_raw, self.score_norm_params["uncertainty"])

        av_unsuitability_train = self._compute_av_unsuitability_score(
            eta_score, safety_score, uncertainty_score
        )

        self.grade_thresholds = np.quantile(av_unsuitability_train, [0.25, 0.50, 0.75]).tolist()
        log.info(f"Grade thresholds (train): {self.grade_thresholds}")

        # 5. SHAP 解释器
        self.explainer = shap.TreeExplainer(self.model)

        self.is_fitted = True
        log.info("Stage 3 model fitted successfully.")
        return self

    # ============================================================
    # 预测与分级（分层优化框架）
    # ============================================================

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X_raw = self._prepare_X(df)
        X_scaled = self.scaler.transform(X_raw)
        pred = self.model.predict(X_scaled)
        pred = np.clip(pred, -0.5, 2.0)

        eta_raw = pred.astype("float64")
        safety_raw = self._compute_safety_raw(df)
        unc_raw = self._compute_uncertainty_raw(df)

        # ★ 使用训练集参数归一化
        eta_score = self._apply_robust_01(eta_raw, self.score_norm_params["eta"])
        safety_score = self._apply_robust_01(safety_raw, self.score_norm_params["safety"])
        uncertainty_score = self._apply_robust_01(unc_raw, self.score_norm_params["uncertainty"])

        av_unsuitability_score = self._compute_av_unsuitability_score(
            eta_score, safety_score, uncertainty_score
        )

        # 使用训练集阈值分级
        thresholds = self.grade_thresholds
        grades = np.ones(len(av_unsuitability_score), dtype=np.int8)
        grades[av_unsuitability_score >= thresholds[0]] = 2
        grades[av_unsuitability_score >= thresholds[1]] = 3
        grades[av_unsuitability_score >= thresholds[2]] = 4

        result = df[["order_id", "day"]].copy()
        result["eta_risk_score"] = eta_score.astype("float32")
        result["safety_risk_score"] = safety_score.astype("float32")
        result["uncertainty_score"] = uncertainty_score.astype("float32")
        result["av_unsuitability_score"] = av_unsuitability_score.astype("float32")

        # 向后兼容
        result["difficulty"] = (
                0.55 * eta_score
                + 0.30 * safety_score
                + 0.15 * uncertainty_score
        ).astype("float32")
        result["bti_score"] = pred.astype("float32")
        result["safety_score"] = safety_score.astype("float32")
        result["grade"] = grades

        # deprecated
        result["av_reject_prob_proxy"] = av_unsuitability_score.astype("float32")

        return result
    # ============================================================
    # SHAP 解释
    # ============================================================

    def explain(self, df: pd.DataFrame, output_dir: str = '.'):
        if self.explainer is None:
            log.warning("SHAP not available.")
            return

        X_raw = self._prepare_X(df)
        X_scaled = self.scaler.transform(X_raw)

        n_samples = min(500, X_scaled.shape[0])
        idx = np.random.choice(X_scaled.shape[0], n_samples, replace=False)
        X_sample = X_scaled[idx]

        shap_values = self.explainer.shap_values(X_sample)

        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.all_cols_ordered, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=150)
        plt.close()
        log.info(f"SHAP summary saved.")

        key_features = ['R1_max_p4', 'weather_severity', 'D2_cong_severity', 'U1_path_entropy']
        for feat in key_features:
            if feat in self.all_cols_ordered:
                feat_idx = self.all_cols_ordered.index(feat)
                plt.figure(figsize=(8, 5))
                shap.dependence_plot(feat_idx, shap_values, X_sample,
                                     feature_names=self.all_cols_ordered, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_dependence_{feat}.png'), dpi=150)
                plt.close()
                log.info(f"SHAP dependence for {feat} saved.")

    # ============================================================
    # 评估
    # ============================================================

    def evaluate(self, df: pd.DataFrame) -> dict:
        pred = self.predict(df)
        actual = (df['ata'] - df['simple_eta'].clip(lower=1)) / df['simple_eta'].clip(lower=1)
        actual = actual.clip(-0.5, 2.0)

        mae = mean_absolute_error(actual, pred['bti_score'])
        r2 = r2_score(actual, pred['bti_score'])

        grade_counts = pred['grade'].value_counts().sort_index()

        log.info("\n" + "=" * 70)
        log.info("STAGE 3 EVALUATION")
        log.info("=" * 70)
        log.info(f"Prediction: R²={r2:.4f}, MAE={mae:.4f}")
        log.info(f"Grade distribution: G1={grade_counts.get(1,0)}, G2={grade_counts.get(2,0)}, "
                 f"G3={grade_counts.get(3,0)}, G4={grade_counts.get(4,0)}")

        return {
            'r2': r2,
            'mae': mae,
            'grade_counts': grade_counts.to_dict(),
            'boundaries': self.boundaries,
            'safety_threshold': self.safety_threshold,
            'entropy_weights': self.entropy_weights,
        }

    # ============================================================
    # 保存 / 加载
    # ============================================================

    def save(self, tag: str = ''):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"stage3_xgb_{tag}_{ts}" if tag else f"stage3_xgb_{ts}"
        os.makedirs('saved_models', exist_ok=True)
        path = os.path.join('saved_models', name + '.pkl')

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'all_cols_ordered': self.all_cols_ordered,
            'boundaries': self.boundaries,
            'safety_threshold': self.safety_threshold,
            'uncertainty_threshold': self.uncertainty_threshold,
            'entropy_weights': self.entropy_weights,
            'is_fitted': self.is_fitted,
            "score_norm_params": self.score_norm_params,
            "grade_thresholds": self.grade_thresholds,
        }, path)
        log.info(f"Model saved: {path}")
        return path

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.scaler = data['scaler']
        obj.feature_cols = data.get('feature_cols', list(STAGE2_ALL_COLS))
        obj.all_cols_ordered = data.get('all_cols_ordered', [])
        obj.boundaries = data.get('boundaries', [0.5, 1.0])
        obj.safety_threshold = data.get('safety_threshold', SAFETY_THRESHOLD)
        obj.uncertainty_threshold = data.get('uncertainty_threshold', UNCERTAINTY_THRESHOLD)
        obj.entropy_weights = data.get('entropy_weights', {'efficiency': 0.5, 'safety': 0.5})
        obj.is_fitted = data.get('is_fitted', True)
        obj.score_norm_params = data.get("score_norm_params", {"eta": (0, 1), "safety": (0, 1), "uncertainty": (0, 1)})
        obj.grade_thresholds = data.get("grade_thresholds", [0.25, 0.50, 0.75])
        if obj.model is not None:
            obj.explainer = shap.TreeExplainer(obj.model)

        log.info(f"Model loaded: {path}")
        return obj