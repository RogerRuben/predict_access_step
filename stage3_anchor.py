"""
stage3_anchor.py — 增加模型保存/加载
"""
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from config import DRIVER_SHRINKAGE_PRIOR, LASSO_ALPHA, MODEL_DIR
from stage2_assess import STAGE2_ALL_COLS
from logger import get_logger

log = get_logger()


def build_supervision_signal(head_df):
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

    log.info(f"Supervision signal: y_raw mean={df['y_raw'].mean():.4f} std={df['y_raw'].std():.4f}")
    log.info(f"                    y_tilde mean={df['y_tilde'].mean():.4f} std={df['y_tilde'].std():.4f}")
    return df[["order_id", "day", "y_raw", "y_tilde"]]


class DifficultyModel:
    def __init__(self, alpha=LASSO_ALPHA, l1_ratio=0.5):
        self.scaler = StandardScaler()
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)
        self.feature_cols = STAGE2_ALL_COLS
        self.interaction_col = "D1xU1"

    def _add_interaction(self, df):
        df = df.copy()
        df[self.interaction_col] = df["D1_cong_exposure"] * df["U1_path_entropy"]
        return df

    def fit(self, df, y):
        df = self._add_interaction(df)
        all_cols = self.feature_cols + [self.interaction_col]
        X = df[all_cols].values.astype("float64")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y.values)

        coef_df = pd.DataFrame({
            "feature": all_cols, "coefficient": self.model.coef_,
        }).sort_values("coefficient", key=abs, ascending=False)
        log.info(f"ElasticNet coefficients (intercept={self.model.intercept_:.4f}):\n"
                 f"{coef_df.to_string(index=False)}")

        y_pred = self.model.predict(X_scaled)
        log.info(f"Train R²={r2_score(y, y_pred):.4f} MAE={mean_absolute_error(y, y_pred):.4f}")

    def predict(self, df):
        df = self._add_interaction(df)
        all_cols = self.feature_cols + [self.interaction_col]
        X = df[all_cols].values.astype("float64")
        return self.model.predict(self.scaler.transform(X))

    def evaluate(self, df, y):
        y_pred = self.predict(df)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        corr = float(np.corrcoef(y, y_pred)[0, 1])
        log.info(f"Test R²={r2:.4f} MAE={mae:.4f} Corr={corr:.4f}")
        return {"r2": r2, "mae": mae, "corr": corr}

    def save(self, tag=""):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"stage3_difficulty_{tag}_{ts}" if tag else f"stage3_difficulty_{ts}"
        path = os.path.join(MODEL_DIR, name + ".pkl")
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)
        log.info(f"Stage 3 model saved: {path}")
        return path

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.model = data["model"]
        log.info(f"Stage 3 model loaded: {path}")
        return obj