"""
stage1_predict.py — 类别加权 + 改进评估
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
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


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    ★ 计算 inverse-frequency 样本权重。
    少数类获得更高权重，使 LightGBM 关注拥堵类别。
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    # sklearn 风格: w_c = total / (n_classes * count_c)
    weight_map = {c: total / (n_classes * cnt) for c, cnt in zip(classes, counts)}

    log.info(f"Class weights: {', '.join(f's{c+1}={w:.2f}' for c, w in sorted(weight_map.items()))}")

    return np.array([weight_map[yi] for yi in y], dtype="float32")


def prepare_training_data(df: pd.DataFrame):
    valid = df[df[STAGE1_TARGET].isin(STATUS_CLASSES)].copy()
    valid["label"] = valid[STAGE1_TARGET] - 1

    total = len(df)
    n_valid = len(valid)
    n_arr0 = int((df[STAGE1_TARGET] == 0).sum())
    n_cur0 = int(df["is_status_unknown"].sum()) if "is_status_unknown" in df.columns else 0

    log.info(f"Training data: {n_valid:,}/{total:,} valid ({n_arr0:,} arrival=0 excluded)")
    log.info(f"current_status=0: {n_cur0:,}/{total:,} ({n_cur0/max(total,1)*100:.1f}%)")
    log.info(f"Label distribution:\n{valid['label'].value_counts().sort_index().to_string()}")

    X = valid[STAGE1_FEATURE_COLS].values.astype("float32")
    y = valid["label"].values.astype("int32")
    return X, y, valid


def train_model(X: np.ndarray, y: np.ndarray) -> lgb.Booster:
    # ★ 计算样本权重
    sample_weights = compute_sample_weights(y)

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X, y, sample_weights,
        test_size=0.15, random_state=42, stratify=y,
    )

    # ★ 传入权重
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=STAGE1_FEATURE_COLS)
    dval = lgb.Dataset(X_val, label=y_val, weight=w_val,
                       feature_name=STAGE1_FEATURE_COLS, reference=dtrain)

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

    # ---- 评估（不使用权重，反映真实性能）----
    y_val_pred = model.predict(X_val)
    y_val_cls = y_val_pred.argmax(axis=1)

    acc = accuracy_score(y_val, y_val_cls)
    ll = log_loss(y_val, y_val_pred)
    wf1 = f1_score(y_val, y_val_cls, average="weighted")
    mf1 = f1_score(y_val, y_val_cls, average="macro")

    # ★ 安全关键指标
    # 危险漏判: 真实 status≥3 (label≥2) 被预测为 status≤2 (label≤1)
    high_actual = (y_val >= 2)
    if high_actual.sum() > 0:
        dangerous_miss = ((y_val_cls < 2) & high_actual).sum() / high_actual.sum()
    else:
        dangerous_miss = 0.0

    # ★ 4→3 混淆率: 真实 status=4 被预测为 status≤3
    s4_actual = (y_val == 3)
    if s4_actual.sum() > 0:
        s4_underestimate = ((y_val_cls < 3) & s4_actual).sum() / s4_actual.sum()
    else:
        s4_underestimate = 0.0

    log.info(f"=== Stage 1 Validation ===")
    log.info(f"  Accuracy:                {acc:.4f}")
    log.info(f"  Weighted F1:             {wf1:.4f}")
    log.info(f"  Macro F1:                {mf1:.4f}")
    log.info(f"  Log Loss:                {ll:.4f}")
    log.info(f"  Dangerous Miss (≥3→≤2):  {dangerous_miss:.4f}")
    log.info(f"  s4 Underestimate (4→≤3): {s4_underestimate:.4f}")

    # 每个类的 recall（最关键指标）
    cm = confusion_matrix(y_val, y_val_cls)
    labels = [f"s{k}" for k in STATUS_CLASSES]
    for i, label in enumerate(labels):
        if cm[i].sum() > 0:
            recall_i = cm[i, i] / cm[i].sum()
            log.info(f"  Recall {label}: {recall_i:.4f} ({cm[i,i]:,}/{cm[i].sum():,})")

    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                         columns=[f"pred_{l}" for l in labels])
    log.info(f"  Confusion Matrix:\n{cm_df.to_string()}")

    report = classification_report(y_val, y_val_cls, target_names=labels, digits=4)
    log.info(f"  Classification Report:\n{report}")

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