"""
stage3_anchor.py
----------------
Stage 3: 数据驱动的难度锚定
- 构建监督信号: 控制驾驶员效应后的 ata 偏差
- 训练 ElasticNet 回归模型学习特征权重
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from config import DRIVER_SHRINKAGE_PRIOR, LASSO_ALPHA
from stage2_assess import STAGE2_ALL_COLS


def build_supervision_signal(head_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建去驾驶员效应的 ata 偏差率 y_tilde。
    使用贝叶斯收缩处理低频驾驶员。
    """
    df = head_df.copy()

    # 原始偏差率
    df["y_raw"] = (df["ata"] - df["simple_eta"]) / df["simple_eta"].clip(lower=1)

    # 全局均值
    global_mean = float(df["y_raw"].mean())

    # 驾驶员统计
    driver_stats = df.groupby("driver_id")["y_raw"].agg(["mean", "count"]).reset_index()
    driver_stats.columns = ["driver_id", "driver_mean", "driver_count"]

    # 贝叶斯收缩
    k = DRIVER_SHRINKAGE_PRIOR
    driver_stats["shrinkage"] = driver_stats["driver_count"] / (driver_stats["driver_count"] + k)
    driver_stats["alpha_driver"] = (
        driver_stats["shrinkage"] * driver_stats["driver_mean"]
        + (1 - driver_stats["shrinkage"]) * global_mean
    )

    df = df.merge(driver_stats[["driver_id", "alpha_driver"]], on="driver_id", how="left")
    df["alpha_driver"] = df["alpha_driver"].fillna(global_mean)
    df["y_tilde"] = df["y_raw"] - df["alpha_driver"]

    print(f"[stage3] supervision signal built:")
    print(f"         y_raw   mean={df['y_raw'].mean():.4f}  std={df['y_raw'].std():.4f}")
    print(f"         y_tilde mean={df['y_tilde'].mean():.4f}  std={df['y_tilde'].std():.4f}")

    return df[["order_id", "day", "y_raw", "y_tilde"]]


class DifficultyModel:
    """
    难度回归模型：从 Stage 2 特征 → 难度评分。
    使用 ElasticNet 保证可解释性和特征选择。
    """

    def __init__(self, alpha: float = LASSO_ALPHA, l1_ratio: float = 0.5):
        self.scaler = StandardScaler()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
        self.feature_cols = STAGE2_ALL_COLS
        self.interaction_col = "D1xU1"

    def _add_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加拥堵×不确定性交互项。"""
        df = df.copy()
        df[self.interaction_col] = df["D1_cong_exposure"] * df["U1_path_entropy"]
        return df

    def fit(self, df: pd.DataFrame, y: pd.Series):
        """
        训练难度模型。

        Parameters
        ----------
        df : 包含 STAGE2_ALL_COLS 的 DataFrame
        y  : 监督信号 (y_tilde)
        """
        df = self._add_interaction(df)
        all_cols = self.feature_cols + [self.interaction_col]

        X = df[all_cols].values.astype("float64")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.values)

        # 输出系数
        coef_df = pd.DataFrame({
            "feature": all_cols,
            "coefficient": self.model.coef_,
        }).sort_values("coefficient", key=abs, ascending=False)

        print(f"\n[stage3] ElasticNet coefficients (intercept={self.model.intercept_:.4f}):")
        print(coef_df.to_string(index=False))

        # 训练集 R²
        y_pred = self.model.predict(X_scaled)
        print(f"\n[stage3] Train R²={r2_score(y, y_pred):.4f}  MAE={mean_absolute_error(y, y_pred):.4f}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """预测难度评分。"""
        df = self._add_interaction(df)
        all_cols = self.feature_cols + [self.interaction_col]
        X = df[all_cols].values.astype("float64")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, df: pd.DataFrame, y: pd.Series):
        """在测试集上评估。"""
        y_pred = self.predict(df)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        corr = np.corrcoef(y, y_pred)[0, 1]
        print(f"\n[stage3] Test R²={r2:.4f}  MAE={mae:.4f}  Corr={corr:.4f}")
        return {"r2": r2, "mae": mae, "corr": corr}