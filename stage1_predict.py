"""
stage1_predict.py
-----------------
Stage 1: Link-level Traffic State Prediction
使用 LightGBM 多分类模型预测每个 link 的 arrival_status 概率分布。
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from config import (
    LGBM_PARAMS, LGBM_NUM_ROUNDS, LGBM_EARLY_STOPPING,
    STATUS_CLASSES, NUM_CLASSES,
)
from feature_eng import STAGE1_FEATURE_COLS, STAGE1_TARGET


def prepare_training_data(df: pd.DataFrame):
    """
    过滤有效样本（arrival_status ∈ {1,2,3,4}），构建特征和标签。
    """
    valid = df[df[STAGE1_TARGET].isin(STATUS_CLASSES)].copy()
    # 标签转为 0-indexed
    valid["label"] = valid[STAGE1_TARGET] - 1
    print(f"[stage1] valid samples: {len(valid):,} / {len(df):,} "
          f"({len(valid)/len(df)*100:.1f}%)")
    print(f"[stage1] label distribution:\n{valid['label'].value_counts().sort_index().to_string()}")

    X = valid[STAGE1_FEATURE_COLS].values.astype("float32")
    y = valid["label"].values.astype("int32")
    return X, y, valid


def train_model(X: np.ndarray, y: np.ndarray) -> lgb.Booster:
    """训练 LightGBM 多分类模型。"""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y,
    )

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=STAGE1_FEATURE_COLS)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=STAGE1_FEATURE_COLS, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=LGBM_NUM_ROUNDS,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # 验证集评估
    y_val_pred = model.predict(X_val)
    y_val_cls = y_val_pred.argmax(axis=1)
    acc = accuracy_score(y_val, y_val_cls)
    ll = log_loss(y_val, y_val_pred)

    # adjacent accuracy: 预测与真实差 ≤ 1
    adj_acc = np.mean(np.abs(y_val_cls - y_val) <= 1)

    print(f"\n[stage1] Validation Results:")
    print(f"         Accuracy:          {acc:.4f}")
    print(f"         Adjacent Accuracy: {adj_acc:.4f}")
    print(f"         Log Loss:          {ll:.4f}")

    # 特征重要性
    imp = pd.DataFrame({
        "feature": STAGE1_FEATURE_COLS,
        "importance": model.feature_importance("gain"),
    }).sort_values("importance", ascending=False)
    print(f"\n[stage1] Feature Importance (top 10):")
    print(imp.head(10).to_string(index=False))

    return model


def predict_proba(model: lgb.Booster, df: pd.DataFrame) -> pd.DataFrame:
    """
    对 DataFrame 中每行预测 arrival_status 的概率分布。
    返回添加了 pred_p1/p2/p3/p4 和 pred_status、entropy 列的 DataFrame。
    """
    X = df[STAGE1_FEATURE_COLS].values.astype("float32")
    proba = model.predict(X)  # shape: (n, 4)

    df = df.copy()
    for k in range(NUM_CLASSES):
        df[f"pred_p{k+1}"] = proba[:, k].astype("float32")

    # 期望状态 (1-indexed)
    df["pred_status"] = (proba * np.array([[1, 2, 3, 4]])).sum(axis=1).astype("float32")

    # 预测熵
    eps = 1e-10
    df["pred_entropy"] = -(proba * np.log(proba + eps)).sum(axis=1).astype("float32")

    # 拥堵概率
    df["pred_cong_prob"] = (proba[:, 2] + proba[:, 3]).astype("float32")

    # 期望 omega
    df["pred_omega"] = (proba[:, 2] * 1 + proba[:, 3] * 3).astype("float32")

    return df