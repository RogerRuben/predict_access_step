"""
stage3_anchor.py — 完整最终版
- ElasticNetCV 主模型 + LightGBM robustness check
- 统一的 _prepare_X 保证 fit/predict 维度一致
- schema 版本校验
- 缓存兼容
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


    INTERACTION_PAIRS = [
        ("D2_cong_severity", "U1_path_entropy"),
        ("D6_hard_cross", "U1_path_entropy"),
        ("D2_cong_severity", "D7_cross_freq"),
        # ★ 新增
        ("R1_max_p4", "U1_path_entropy"),
        ("R1_max_p4", "D6_hard_cross"),
    ]
    SCHEMA_VERSION = "stage3_v3"

    def __init__(self):
        self.feature_cols = list(STAGE2_ALL_COLS)
        self.scaler = StandardScaler()
        self.elastic_model = None
        self.lgbm_model = None
        self.interaction_cols = []
        self._all_cols_ordered = []

    # ----------------------------------------------------------------
    # 特征准备（统一入口，fit 和 predict 都用这个）
    # ----------------------------------------------------------------

    def _build_interaction_cols(self):
        """根据 INTERACTION_PAIRS 生成交互列名列表。"""
        return [f"{c1}_x_{c2}" for c1, c2 in self.INTERACTION_PAIRS]

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        """
        统一的特征构建流程:
        1. 取原始特征列
        2. 构建交互项
        3. 按固定列顺序输出
        """
        out = df[self.feature_cols].copy()

        for c1, c2 in self.INTERACTION_PAIRS:
            col_name = f"{c1}_x_{c2}"
            out[col_name] = (out[c1] * out[c2]).astype("float32")

        self.interaction_cols = self._build_interaction_cols()

        if self._all_cols_ordered:
            cols = self._all_cols_ordered
        else:
            cols = self.feature_cols + self.interaction_cols

        X = out[cols].values.astype("float32")

        # 维度校验
        expected = getattr(self.scaler, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(
                f"Feature dimension mismatch: code produces {X.shape[1]} features, "
                f"but loaded scaler expects {expected}. "
                f"Please retrain: python main.py train"
            )
        return X

    # ----------------------------------------------------------------
    # 训练
    # ----------------------------------------------------------------

    def fit(self, df: pd.DataFrame, y: pd.Series):
        # 确定列顺序
        self.interaction_cols = self._build_interaction_cols()
        self._all_cols_ordered = self.feature_cols + self.interaction_cols
        all_cols = self._all_cols_ordered

        X_raw = self._prepare_X(df)

        log.info(f"Stage 3 fitting: {len(y):,} samples, {X_raw.shape[1]} features")
        log.info(f"  Columns: {all_cols}")

        # ==== 主模型: ElasticNetCV ====
        log.info("\n" + "-" * 60)
        log.info("Baseline Model: ElasticNetCV")
        log.info("-" * 60)

        X_scaled = self.scaler.fit_transform(X_raw)

        self.elastic_model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-5, -1, 30),
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

        coef_df = pd.DataFrame({
            "feature": all_cols,
            "coefficient": self.elastic_model.coef_,
            "abs_coef": np.abs(self.elastic_model.coef_),
        }).sort_values("abs_coef", ascending=False)
        coef_df["significant"] = coef_df["abs_coef"] > 1e-6

        log.info(f"  Intercept = {self.elastic_model.intercept_:.6f}")
        log.info(f"  Coefficients:")
        for _, row in coef_df.iterrows():
            marker = "★" if row["significant"] else " "
            log.info(f"    {marker} {row['feature']:45s}  β = {row['coefficient']:+.6f}")
        log.info(f"  Active: {int(coef_df['significant'].sum())}/{len(all_cols)}")

        # ==== 稳健性检验: LightGBM ====
        log.info("\n" + "-" * 60)
        log.info("Robustness Check: LightGBM Regressor")
        log.info("-" * 60)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_raw, y.values, test_size=0.15, random_state=42,
        )
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=all_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=all_cols, reference=dtrain)

        self.lgbm_model = lgb.train(
            {
                "objective": "regression", "metric": "mae",
                "num_leaves": 63, "learning_rate": 0.03,
                "feature_fraction": 0.8, "bagging_fraction": 0.8,
                "bagging_freq": 5, "verbose": -1, "seed": 42,
            },
            dtrain,
            num_boost_round=500,
            valid_sets=[dval], valid_names=["valid"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        y_lgbm_val = self.lgbm_model.predict(X_val)
        r2_lgbm = r2_score(y_val, y_lgbm_val)
        mae_lgbm = mean_absolute_error(y_val, y_lgbm_val)

        log.info(f"  LightGBM Val R² = {r2_lgbm:.4f}")
        log.info(f"  LightGBM Val MAE = {mae_lgbm:.4f}")
        log.info(f"  R² uplift vs ElasticNet: {r2_lgbm - r2_enet:+.4f}")

        imp = pd.DataFrame({
            "feature": all_cols,
            "importance": self.lgbm_model.feature_importance("gain"),
        }).sort_values("importance", ascending=False)
        log.info(f"  Feature Importance:\n{imp.to_string(index=False)}")

    # ----------------------------------------------------------------
    # 预测
    # ----------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X_raw = self._prepare_X(df)
        X_scaled = self.scaler.transform(X_raw)
        return self.elastic_model.predict(X_scaled)

    def predict_robustness(self, df: pd.DataFrame) -> np.ndarray:
        X_raw = self._prepare_X(df)
        return self.lgbm_model.predict(X_raw)

    # ----------------------------------------------------------------
    # 评估
    # ----------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame, y: pd.Series) -> dict:
        log.info("\n" + "-" * 60)
        log.info("Test Set Evaluation")
        log.info("-" * 60)

        y_pred = self.predict(df)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        corr = float(np.corrcoef(y, y_pred)[0, 1])
        log.info(f"  Baseline (ElasticNet):  R²={r2:.4f}  MAE={mae:.4f}  Corr={corr:.4f}")

        result = {"baseline": {"r2": r2, "mae": mae, "corr": corr}}

        if self.lgbm_model is not None:
            y_r = self.predict_robustness(df)
            r2_r = r2_score(y, y_r)
            mae_r = mean_absolute_error(y, y_r)
            corr_r = float(np.corrcoef(y, y_r)[0, 1])
            log.info(f"  Robustness (LightGBM):  R²={r2_r:.4f}  MAE={mae_r:.4f}  Corr={corr_r:.4f}")
            log.info(f"  Nonlinear R² uplift: {r2_r - r2:+.4f}")
            result["robustness"] = {"r2": r2_r, "mae": mae_r, "corr": corr_r}
        else:
            log.info("  Robustness (LightGBM): not available")
            result["robustness"] = None

        return result

    # ----------------------------------------------------------------
    # 保存 / 加载
    # ----------------------------------------------------------------

    def save(self, tag: str = ""):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"stage3_difficulty_{tag}_{ts}" if tag else f"stage3_difficulty_{ts}"
        path = os.path.join(MODEL_DIR, name + ".pkl")
        joblib.dump({
            "schema_version": self.SCHEMA_VERSION,
            "scaler": self.scaler,
            "elastic_model": self.elastic_model,
            "lgbm_model": self.lgbm_model,
            "interaction_pairs": self.INTERACTION_PAIRS,
            "feature_cols": self.feature_cols,
            "all_cols_ordered": self._all_cols_ordered,
        }, path)
        log.info(f"Stage 3 saved: {path}")
        log.info(f"  schema: {self.SCHEMA_VERSION}, features: {len(self._all_cols_ordered)}")
        return path

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls()

        schema = data.get("schema_version", "unknown")

        # ★ 严格校验：拒绝旧模型
        if schema not in ("stage3_v3",):
            raise ValueError(
                f"Incompatible Stage 3 model: schema='{schema}' in {path}.\n"
                f"Current code requires schema='stage3_v3'.\n"
                f"Please delete old models and rerun: python main.py train"
            )

        obj.scaler = data["scaler"]
        obj.elastic_model = data["elastic_model"]
        obj.lgbm_model = data.get("lgbm_model", None)
        obj.feature_cols = data.get("feature_cols", list(STAGE2_ALL_COLS))
        obj.INTERACTION_PAIRS = data.get("interaction_pairs", cls.INTERACTION_PAIRS)
        obj._all_cols_ordered = data.get("all_cols_ordered", [])
        obj.interaction_cols = obj._build_interaction_cols()

        # 如果旧文件没存 all_cols_ordered，重建
        if not obj._all_cols_ordered:
            obj._all_cols_ordered = obj.feature_cols + obj.interaction_cols

        log.info(f"Stage 3 loaded: {path}")
        log.info(f"  schema: {schema}")
        log.info(f"  scaler expects: {getattr(obj.scaler, 'n_features_in_', '?')} features")
        log.info(f"  all_cols: {len(obj._all_cols_ordered)}")
        log.info(f"  ElasticNet: {'✓' if obj.elastic_model else '✗'}")
        log.info(f"  LightGBM:   {'✓' if obj.lgbm_model else '✗'}")

        return obj