"""
stage3_anchor.py
----------------
Stage 3: 数据驱动的难度锚定

学术架构:
  - 主模型 (Baseline): ElasticNetCV — 可解释性 + 固定效应控制
  - 稳健性检验 (Robustness Check): LightGBM 回归 — 验证非线性效应
"""
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from datetime import datetime
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from config import DRIVER_SHRINKAGE_PRIOR, MODEL_DIR, Y_TILDE_WINSORIZE
from stage2_assess import STAGE2_ALL_COLS
from logger import get_logger

log = get_logger()


def build_supervision_signal(head_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建去驾驶员固定效应的 ata 偏差率 y_tilde。
    使用贝叶斯收缩处理低频驾驶员。
    """
    df = head_df.copy()
    df["y_raw"] = (df["ata"] - df["simple_eta"]) / df["simple_eta"].clip(lower=1)
    global_mean = float(df["y_raw"].mean())

    driver_stats = df.groupby("driver_id")["y_raw"].agg(["mean", "count"]).reset_index()
    driver_stats.columns = ["driver_id", "driver_mean", "driver_count"]
    k = DRIVER_SHRINKAGE_PRIOR
    driver_stats["shrinkage"] = driver_stats["driver_count"] / (driver_stats["driver_count"] + k)
    driver_stats["alpha_driver"] = (
        driver_stats["shrinkage"] * driver_stats["driver_mean"]
        + (1 - driver_stats["shrinkage"]) * global_mean
    )

    df = df.merge(driver_stats[["driver_id", "alpha_driver"]], on="driver_id", how="left")
    df["alpha_driver"] = df["alpha_driver"].fillna(global_mean)
    df["y_tilde"] = df["y_raw"] - df["alpha_driver"]

    # Winsorize
    lo, hi = Y_TILDE_WINSORIZE
    q_lo = float(df["y_tilde"].quantile(lo))
    q_hi = float(df["y_tilde"].quantile(hi))
    n_clipped = int(((df["y_tilde"] < q_lo) | (df["y_tilde"] > q_hi)).sum())
    df["y_tilde"] = df["y_tilde"].clip(q_lo, q_hi)

    log.info(f"Supervision signal:")
    log.info(f"  y_raw    mean={df['y_raw'].mean():.4f}  std={df['y_raw'].std():.4f}")
    log.info(f"  y_tilde  mean={df['y_tilde'].mean():.4f}  std={df['y_tilde'].std():.4f}")
    log.info(f"  Winsorized [{q_lo:.4f}, {q_hi:.4f}], {n_clipped:,} clipped")
    log.info(f"  Unique drivers: {df['driver_id'].nunique():,}")

    return df[["order_id", "day", "y_raw", "y_tilde"]]


class DifficultyModel:
    """
    学术架构的难度评分模型:

    主模型 (Baseline):
        ElasticNetCV 回归 — 系数直接揭示各难度维度的边际贡献
        控制了驾驶员固定效应后 R²=0.03-0.05 在交通领域完全合理

    稳健性检验 (Robustness Check):
        LightGBM 回归 — 验证是否存在显著的非线性效应
        若 R² 显著提升，则说明难度维度间存在非线性交互
    """

    # ★ 精心设计的交互项（使用连续变量，避免稀疏导致归零）
    INTERACTION_PAIRS = [
        # 核心交互：拥堵严重度 × 预测不确定性
        # D2 是连续加权值（不像 D1 大量为0），U1 是路径级熵
        ("D2_cong_severity", "U1_path_entropy"),
        # 路口复杂度 × 预测不确定性
        # D6 是累计时间（连续），捕捉"复杂路口+不可预测"的复合风险
        ("D6_hard_cross", "U1_path_entropy"),
        # 拥堵严重度 × 路口决策频率
        # 两个连续变量，捕捉"拥堵中频繁过路口"的复合场景
        ("D2_cong_severity", "D7_cross_freq"),
    ]

    def __init__(self):
        self.feature_cols = STAGE2_ALL_COLS
        self.scaler = StandardScaler()
        self.elastic_model = None    # 主模型
        self.lgbm_model = None       # 稳健性检验
        self.interaction_cols = []

    def _add_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建交互特征。"""
        df = df.copy()
        self.interaction_cols = []
        for c1, c2 in self.INTERACTION_PAIRS:
            col_name = f"{c1}_x_{c2}"
            df[col_name] = (df[c1] * df[c2]).astype("float64")
            self.interaction_cols.append(col_name)
        return df

    def _all_cols(self) -> list[str]:
        return self.feature_cols + self.interaction_cols

    def fit(self, df: pd.DataFrame, y: pd.Series):
        df = self._add_interactions(df)
        all_cols = self._all_cols()
        X_raw = df[all_cols].values.astype("float64")

        # 数据诊断
        log.info(f"Stage 3 fitting: {len(y):,} samples, {len(all_cols)} features")
        log.info(f"Feature variance check:")
        for i, col in enumerate(all_cols):
            var = np.var(X_raw[:, i])
            pct_zero = np.mean(X_raw[:, i] == 0) * 100
            log.info(f"  {col:40s}  var={var:.6f}  zeros={pct_zero:.1f}%")

        # ==============================================================
        # 主模型: ElasticNetCV（交叉验证自动选择最优正则化参数）
        # ==============================================================
        log.info("\n" + "-" * 60)
        log.info("Baseline Model: ElasticNetCV")
        log.info("-" * 60)

        X_scaled = self.scaler.fit_transform(X_raw)

        self.elastic_model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-5, -1, 30),    # 更宽的搜索范围
            cv=5,
            max_iter=10000,
            random_state=42,
        )
        self.elastic_model.fit(X_scaled, y.values)

        y_pred_enet = self.elastic_model.predict(X_scaled)
        r2_enet = r2_score(y, y_pred_enet)
        mae_enet = mean_absolute_error(y, y_pred_enet)

        log.info(f"  Best alpha = {self.elastic_model.alpha_:.6f}")
        log.info(f"  Best l1_ratio = {self.elastic_model.l1_ratio_:.2f}")
        log.info(f"  Train R² = {r2_enet:.4f}")
        log.info(f"  Train MAE = {mae_enet:.4f}")

        # 系数表（核心输出：每个维度的边际贡献）
        coef_df = pd.DataFrame({
            "feature": all_cols,
            "coefficient": self.elastic_model.coef_,
            "abs_coef": np.abs(self.elastic_model.coef_),
        }).sort_values("abs_coef", ascending=False)
        coef_df["significant"] = coef_df["abs_coef"] > 1e-6

        log.info(f"  Intercept = {self.elastic_model.intercept_:.6f}")
        log.info(f"  Coefficients (sorted by |β|):")
        for _, row in coef_df.iterrows():
            marker = "★" if row["significant"] else " "
            log.info(f"    {marker} {row['feature']:45s}  β = {row['coefficient']:+.6f}")

        n_active = int(coef_df["significant"].sum())
        log.info(f"  Active features: {n_active}/{len(all_cols)}")

        # ==============================================================
        # 稳健性检验: LightGBM 回归
        # ==============================================================
        log.info("\n" + "-" * 60)
        log.info("Robustness Check: LightGBM Regressor")
        log.info("-" * 60)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_raw, y.values, test_size=0.15, random_state=42,
        )

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=all_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=all_cols, reference=dtrain)

        lgbm_params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

        self.lgbm_model = lgb.train(
            lgbm_params, dtrain,
            num_boost_round=500,
            valid_sets=[dval], valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        y_pred_lgbm_val = self.lgbm_model.predict(X_val)
        r2_lgbm = r2_score(y_val, y_pred_lgbm_val)
        mae_lgbm = mean_absolute_error(y_val, y_pred_lgbm_val)

        log.info(f"  LightGBM Val R² = {r2_lgbm:.4f}")
        log.info(f"  LightGBM Val MAE = {mae_lgbm:.4f}")
        log.info(f"  R² uplift over ElasticNet: {r2_lgbm - r2_enet:+.4f}")

        if r2_lgbm > r2_enet * 1.5:
            log.info("  → Significant nonlinear effects detected.")
        else:
            log.info("  → Nonlinear uplift is modest; linear baseline is adequate.")

        # LightGBM 特征重要性（辅助分析）
        imp = pd.DataFrame({
            "feature": all_cols,
            "importance": self.lgbm_model.feature_importance("gain"),
        }).sort_values("importance", ascending=False)
        log.info(f"  LightGBM Feature Importance:\n{imp.head(10).to_string(index=False)}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """主模型预测（ElasticNet）。"""
        df = self._add_interactions(df)
        X = df[self._all_cols()].values.astype("float64")
        return self.elastic_model.predict(self.scaler.transform(X))

    def predict_robustness(self, df: pd.DataFrame) -> np.ndarray:
        """稳健性检验预测（LightGBM）。"""
        df = self._add_interactions(df)
        X = df[self._all_cols()].values.astype("float64")
        return self.lgbm_model.predict(X)

    def evaluate(self, df: pd.DataFrame, y: pd.Series) -> dict:
        log.info("\n" + "-" * 60)
        log.info("Test Set Evaluation")
        log.info("-" * 60)

        # 主模型
        y_pred = self.predict(df)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        corr = float(np.corrcoef(y, y_pred)[0, 1])
        log.info(f"  Baseline (ElasticNet):  R²={r2:.4f}  MAE={mae:.4f}  Corr={corr:.4f}")

        # 稳健性检验
        y_pred_r = self.predict_robustness(df)
        r2_r = r2_score(y, y_pred_r)
        mae_r = mean_absolute_error(y, y_pred_r)
        corr_r = float(np.corrcoef(y, y_pred_r)[0, 1])
        log.info(f"  Robustness (LightGBM):  R²={r2_r:.4f}  MAE={mae_r:.4f}  Corr={corr_r:.4f}")
        log.info(f"  Nonlinear R² uplift: {r2_r - r2:+.4f}")

        return {
            "baseline": {"r2": r2, "mae": mae, "corr": corr},
            "robustness": {"r2": r2_r, "mae": mae_r, "corr": corr_r},
        }

    def save(self, tag: str = ""):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"stage3_difficulty_{tag}_{ts}" if tag else f"stage3_difficulty_{ts}"
        path = os.path.join(MODEL_DIR, name + ".pkl")
        joblib.dump({
            "scaler": self.scaler,
            "elastic_model": self.elastic_model,
            "lgbm_model": self.lgbm_model,
            "interaction_pairs": self.INTERACTION_PAIRS,
            "feature_cols": self.feature_cols,
        }, path)
        log.info(f"Stage 3 models saved: {path}")
        return path

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.elastic_model = data["elastic_model"]
        obj.lgbm_model = data["lgbm_model"]
        if "interaction_pairs" in data:
            obj.INTERACTION_PAIRS = data["interaction_pairs"]
        if "feature_cols" in data:
            obj.feature_cols = data["feature_cols"]
        log.info(f"Stage 3 models loaded: {path}")
        return obj