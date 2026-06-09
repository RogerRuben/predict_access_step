"""
stage1_predict.py — 模型保存/加载 + 改进评估指标
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score,
    confusion_matrix, classification_report,
)
from config import (
    LGBM_PARAMS, LGBM_NUM_ROUNDS, LGBM_EARLY_STOPPING,
    STATUS_CLASSES, NUM_CLASSES, MODEL_DIR,
)
from feature_eng import STAGE1_FEATURE_COLS, STAGE1_TARGET
from logger import get_logger

log = get_logger()


def prepare_training_data(df: pd.DataFrame):
    """过滤有效样本: arrival_status ∈ {1,2,3,4}（排除 0=未知）"""
    valid = df[df[STAGE1_TARGET].isin(STATUS_CLASSES)].copy()
    valid["label"] = valid[STAGE1_TARGET] - 1

    total = len(df)
    n_valid = len(valid)
    n_status0 = int((df[STAGE1_TARGET] == 0).sum())

    log.info(f"Training data: {n_valid:,}/{total:,} valid "
             f"({n_status0:,} arrival_status=0 excluded)")
    log.info(f"Label distribution:\n{valid['label'].value_counts().sort_index().to_string()}")

    # 统计 current_status=0（NaN）的比例
    n_cur0 = int(df["link_current_status"].isna().sum())
    log.info(f"current_status=0 (→NaN): {n_cur0:,}/{total:,} "
             f"({n_cur0/total*100:.1f}%) — handled as missing by LightGBM")

    X = valid[STAGE1_FEATURE_COLS].values.astype("float32")
    y = valid["label"].values.astype("int32")
    return X, y, valid


def train_model(X: np.ndarray, y: np.ndarray) -> lgb.Booster:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y,
    )

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=STAGE1_FEATURE_COLS)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=STAGE1_FEATURE_COLS, reference=dtrain)

    model = lgb.train(
        LGBM_PARAMS, dtrain,
        num_boost_round=LGBM_NUM_ROUNDS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING),
            lgb.log_evaluation(period=100),
        ],
    )

    # ---- 评估 ----
    y_val_pred = model.predict(X_val)
    y_val_cls = y_val_pred.argmax(axis=1)

    acc = accuracy_score(y_val, y_val_cls)
    ll = log_loss(y_val, y_val_pred)
    wf1 = f1_score(y_val, y_val_cls, average="weighted")
    mf1 = f1_score(y_val, y_val_cls, average="macro")

    # 安全关键指标: 将高拥堵(3/4)误判为低拥堵(1/2)的比例
    high_actual = (y_val >= 2)  # label 2,3 = status 3,4
    if high_actual.sum() > 0:
        dangerous_miss = ((y_val_cls < 2) & high_actual).sum() / high_actual.sum()
    else:
        dangerous_miss = 0.0

    log.info(f"=== Stage 1 Validation ===")
    log.info(f"  Accuracy:       {acc:.4f}")
    log.info(f"  Weighted F1:    {wf1:.4f}")
    log.info(f"  Macro F1:       {mf1:.4f}")
    log.info(f"  Log Loss:       {ll:.4f}")
    log.info(f"  Dangerous Miss Rate (status≥3 predicted as ≤2): {dangerous_miss:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_val, y_val_cls)
    labels = [f"s{k}" for k in STATUS_CLASSES]
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                         columns=[f"pred_{l}" for l in labels])
    log.info(f"  Confusion Matrix:\n{cm_df.to_string()}")

    # 分类报告
    report = classification_report(y_val, y_val_cls, target_names=labels, digits=4)
    log.info(f"  Classification Report:\n{report}")

    # 特征重要性
    imp = pd.DataFrame({
        "feature": STAGE1_FEATURE_COLS,
        "importance": model.feature_importance("gain"),
    }).sort_values("importance", ascending=False)
    log.info(f"  Feature Importance:\n{imp.to_string(index=False)}")

    return model


def save_stage1_model(model: lgb.Booster, tag: str = ""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"stage1_lgbm_{tag}_{ts}" if tag else f"stage1_lgbm_{ts}"
    path = os.path.join(MODEL_DIR, name + ".txt")
    model.save_model(path)
    log.info(f"Stage 1 model saved: {path}")
    return path


def load_stage1_model(path: str) -> lgb.Booster:
    model = lgb.Booster(model_file=path)
    log.info(f"Stage 1 model loaded: {path}")
    return model


def predict_proba(model: lgb.Booster, df: pd.DataFrame) -> pd.DataFrame:
    X = df[STAGE1_FEATURE_COLS].values.astype("float32")
    proba = model.predict(X)

    df = df.copy()
    for k in range(NUM_CLASSES):
        df[f"pred_p{k+1}"] = proba[:, k].astype("float32")

    df["pred_status"] = (proba * np.array([[1, 2, 3, 4]])).sum(axis=1).astype("float32")

    eps = 1e-10
    df["pred_entropy"] = -(proba * np.log(proba + eps)).sum(axis=1).astype("float32")
    df["pred_cong_prob"] = (proba[:, 2] + proba[:, 3]).astype("float32")
    df["pred_omega"] = (proba[:, 2] * 1 + proba[:, 3] * 3).astype("float32")

    return df