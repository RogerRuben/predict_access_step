"""
pipeline.py — 完整最终版
- Stage 2 缓存机制
- train/test 双模式
- Stage 1 模型路径追踪
"""
import gc
import os
import glob
import hashlib
import pandas as pd
import numpy as np
from config import (
    TRAIN_DAYS, TEST_DAYS, CROSS_TIME_QUANTILE,
    TRAIN_SAMPLE_RATE, BATCH_SIZE_ORDERS, MODEL_DIR, CACHE_DIR,
)
from loader import load_day, load_split_files, load_topology
from feature_eng import (
    build_stage1_features_batch,
    STAGE1_FEATURE_COLS, STAGE1_TARGET,
)
from stage1_predict import (
    prepare_training_data, train_model, predict_proba,
    save_stage1_model, load_stage1_model,
)
from stage2_assess import (
    compute_deterministic, compute_uncertainty, compute_night,
    STAGE2_ALL_COLS,
)
from stage3_anchor import build_supervision_signal, DifficultyModel
from logger import get_logger

log = get_logger()


# ================================================================
# 工具函数
# ================================================================

def _find_latest_model(prefix: str, ext: str) -> str:
    pattern = os.path.join(MODEL_DIR, f"{prefix}*{ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No saved model: {pattern}")
    latest = files[-1]
    log.info(f"Found latest model: {latest}")
    return latest


def _make_cache_key(day: str, s1_path: str, ct_thresh: float, cross_mean: float) -> str:
    raw = f"{day}|{os.path.basename(s1_path)}|{ct_thresh:.4f}|{cross_mean:.4f}|schema_v3"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_paths(cache_key: str):
    return (
        os.path.join(CACHE_DIR, f"s2_{cache_key}.pkl"),
        os.path.join(CACHE_DIR, f"head_{cache_key}.pkl"),
    )


# ================================================================
# Stage 2: 单天处理（带缓存）
# ================================================================

def _process_day_stage2(day, topo, model, ct_threshold, cross_global_mean,
                        s1_model_path=""):
    """
    单天 Stage 2 处理。
    自动检查缓存：命中则直接读取，否则全量计算后写入缓存。
    """
    # ---- 缓存检查 ----
    cache_key = _make_cache_key(day, s1_model_path, ct_threshold, cross_global_mean)
    s2_cache, head_cache = _cache_paths(cache_key)

    if os.path.exists(s2_cache) and os.path.exists(head_cache):
        log.info(f"  Day {day}: cache HIT ({cache_key})")
        s2_all = pd.read_pickle(s2_cache)
        head_slim = pd.read_pickle(head_cache)
        return s2_all, head_slim

    log.info(f"  Day {day}: cache MISS, computing ...")

    # ---- 全量计算 ----
    head, link, cross = load_day(day)

    unique_orders = link["order_id"].unique()
    s2_parts = []
    det_unc_cols = [c for c in STAGE2_ALL_COLS if c != "D11_night"]

    for i in range(0, len(unique_orders), BATCH_SIZE_ORDERS):
        batch_orders = unique_orders[i:i + BATCH_SIZE_ORDERS]
        link_b = link[link["order_id"].isin(batch_orders)]
        head_b = head[head["order_id"].isin(batch_orders)]
        cross_b = cross[cross["order_id"].isin(batch_orders)]

        feat = build_stage1_features_batch(
            link_b, head_b, cross_b, topo,
            keep_extra_for_stage2=True,
        )
        pred = predict_proba(model, feat)

        det = compute_deterministic(pred, cross_b, topo, ct_threshold, cross_global_mean)
        unc = compute_uncertainty(pred)

        batch_s2 = det.merge(unc, on=["order_id", "day"], how="outer")
        for c in det_unc_cols:
            if c in batch_s2.columns:
                batch_s2[c] = batch_s2[c].fillna(0)
        s2_parts.append(batch_s2)

        del feat, pred, det, unc, batch_s2, link_b, head_b, cross_b
        gc.collect()

    s2_all = pd.concat(s2_parts, ignore_index=True)
    del s2_parts, link, cross
    gc.collect()

    # night
    night = compute_night(head)
    s2_all = s2_all.merge(night, on=["order_id", "day"], how="left")
    s2_all["D11_night"] = s2_all["D11_night"].fillna(0)

    head_slim = head[["order_id", "day", "ata", "simple_eta", "driver_id"]].copy()

    del head, night
    gc.collect()

    # ---- 写缓存 ----
    s2_all.to_pickle(s2_cache)
    head_slim.to_pickle(head_cache)
    log.info(f"  Day {day}: cached ({cache_key})")

    return s2_all, head_slim


# ================================================================
# Stage 1: 训练
# ================================================================

def _run_stage1_train(topo):
    log.info("=" * 70)
    log.info("STAGE 1: Traffic State Prediction (Training)")
    log.info("=" * 70)

    X_parts, y_parts = [], []
    for day in TRAIN_DAYS:
        try:
            log.info(f"Extracting day {day} ...")
            head, link, cross = load_day(day)

            unique_orders = link["order_id"].unique()
            day_samples = 0
            for i in range(0, len(unique_orders), BATCH_SIZE_ORDERS):
                batch_orders = unique_orders[i:i + BATCH_SIZE_ORDERS]
                link_b = link[link["order_id"].isin(batch_orders)]
                head_b = head[head["order_id"].isin(batch_orders)]
                cross_b = cross[cross["order_id"].isin(batch_orders)]

                feat = build_stage1_features_batch(
                    link_b, head_b, cross_b, topo,
                    keep_extra_for_stage2=False,
                )
                valid = feat[feat[STAGE1_TARGET].isin([1, 2, 3, 4])]
                if len(valid) > 0:
                    X_parts.append(valid[STAGE1_FEATURE_COLS].values.astype("float32"))
                    y_parts.append((valid[STAGE1_TARGET].values - 1).astype("int32"))
                    day_samples += len(valid)

                del feat, valid, link_b, head_b, cross_b
                gc.collect()

            del head, link, cross
            gc.collect()
            log.info(f"  Day {day}: {day_samples:,} valid samples")

        except FileNotFoundError:
            log.warning(f"Day {day} missing, skipped.")

    if not X_parts:
        raise RuntimeError("No training data loaded for Stage 1")

    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(y_parts)
    del X_parts, y_parts
    gc.collect()

    log.info(f"Total training samples: {len(y_all):,}")

    if TRAIN_SAMPLE_RATE < 1.0:
        n = int(len(y_all) * TRAIN_SAMPLE_RATE)
        idx = np.random.RandomState(42).choice(len(y_all), n, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]
        log.info(f"Sampled to {n:,} ({TRAIN_SAMPLE_RATE * 100:.0f}%)")

    log.info("Training LightGBM ...")
    model = train_model(X_all, y_all)
    model_path = save_stage1_model(model, tag="v3")

    del X_all, y_all
    gc.collect()

    return model, model_path


# ================================================================
# 全局统计量（训练集）
# ================================================================

def _compute_global_cross_stats(train_days):
    log.info("Computing global cross_time statistics (Train set only) ...")
    ct_vals = []
    for day in train_days:
        try:
            c = load_split_files("cross", day)
            ct_vals.append(c["cross_time"].values)
            del c
        except FileNotFoundError:
            pass

    if ct_vals:
        all_ct = np.concatenate(ct_vals)
        ct_threshold = float(np.percentile(all_ct, CROSS_TIME_QUANTILE * 100))
        cross_global_mean = float(np.mean(all_ct))
        del all_ct
    else:
        ct_threshold = 30.0
        cross_global_mean = 25.0

    del ct_vals
    gc.collect()

    log.info(f"  cross_time P{int(CROSS_TIME_QUANTILE*100)} = {ct_threshold:.2f}s")
    log.info(f"  cross_time global_mean = {cross_global_mean:.2f}s")

    return ct_threshold, cross_global_mean


# ================================================================
# 主管道
# ================================================================

def run_pipeline(mode: str = "train"):
    """
    Parameters
    ----------
    mode : "train" — 完整训练 + 测试
           "test"  — 加载已保存模型，利用 Stage 2 缓存快速测试
    """
    topo = load_topology()

    # ---- Stage 1 ----
    if mode == "train":
        s1_model, s1_model_path = _run_stage1_train(topo)
    else:
        s1_model_path = _find_latest_model("stage1_lgbm", ".txt")
        s1_model = load_stage1_model(s1_model_path)

    # ---- 全局统计量 ----
    ct_threshold, cross_global_mean = _compute_global_cross_stats(TRAIN_DAYS)

    # ---- Stage 2: Train ----
    train_final = None
    if mode == "train":
        log.info("=" * 70)
        log.info("STAGE 2: Difficulty Assessment (Train)")
        log.info("=" * 70)

        train_s2_list, train_head_list = [], []
        for day in TRAIN_DAYS:
            try:
                s2, head_slim = _process_day_stage2(
                    day, topo, s1_model, ct_threshold, cross_global_mean,
                    s1_model_path=s1_model_path,
                )
                train_s2_list.append(s2)
                train_head_list.append(head_slim)
                del s2, head_slim
                gc.collect()
            except FileNotFoundError:
                log.warning(f"Day {day} missing, skipped.")

        train_s2 = pd.concat(train_s2_list, ignore_index=True)
        train_head_slim = pd.concat(train_head_list, ignore_index=True)
        del train_s2_list, train_head_list
        gc.collect()

    # ---- Stage 2: Test ----
    log.info("=" * 70)
    log.info("STAGE 2: Difficulty Assessment (Test)")
    log.info("=" * 70)

    test_s2_list, test_head_list = [], []
    for day in TEST_DAYS:
        try:
            s2, head_slim = _process_day_stage2(
                day, topo, s1_model, ct_threshold, cross_global_mean,
                s1_model_path=s1_model_path,
            )
            test_s2_list.append(s2)
            test_head_list.append(head_slim)
        except FileNotFoundError:
            log.warning(f"Day {day} missing, skipped.")

    test_s2 = pd.concat(test_s2_list, ignore_index=True)
    test_head_slim = pd.concat(test_head_list, ignore_index=True)
    del test_s2_list, test_head_list
    gc.collect()

    # ---- Stage 3 ----
    log.info("=" * 70)
    log.info("STAGE 3: Difficulty Anchoring")
    log.info("=" * 70)

    if mode == "train":
        train_signal = build_supervision_signal(train_head_slim)
        train_final = train_s2.merge(train_signal, on=["order_id", "day"], how="inner")
        del train_s2, train_head_slim
        gc.collect()

        log.info(f"Train samples for Stage 3: {len(train_final):,}")

        diff_model = DifficultyModel()
        diff_model.fit(train_final, train_final["y_tilde"])
        diff_model.save(tag="v3")

        train_final["difficulty"] = diff_model.predict(train_final)
    else:
        s3_path = _find_latest_model("stage3_difficulty", ".pkl")
        diff_model = DifficultyModel.load(s3_path)

    test_signal = build_supervision_signal(test_head_slim)
    test_final = test_s2.merge(test_signal, on=["order_id", "day"], how="inner")
    del test_s2, test_head_slim
    gc.collect()

    log.info(f"Test samples: {len(test_final):,}")

    metrics = diff_model.evaluate(test_final, test_final["y_tilde"])
    test_final["difficulty"] = diff_model.predict(test_final)

    # ---- 分级 + 输出 ----
    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)

    if train_final is not None:
        q30 = train_final["difficulty"].quantile(0.30)
        q70 = train_final["difficulty"].quantile(0.70)
        q90 = train_final["difficulty"].quantile(0.90)
    else:
        q30 = test_final["difficulty"].quantile(0.30)
        q70 = test_final["difficulty"].quantile(0.70)
        q90 = test_final["difficulty"].quantile(0.90)

    dfs_to_grade = []
    if train_final is not None:
        dfs_to_grade.append((train_final, "Train"))
    dfs_to_grade.append((test_final, "Test"))

    for df, name in dfs_to_grade:
        df["grade"] = pd.cut(
            df["difficulty"],
            bins=[-np.inf, q30, q70, q90, np.inf],
            labels=["G1_AV_Easy", "G2_AV_Moderate", "G3_AV_Hard", "G4_HV_Only"],
        )
        log.info(f"[{name}] Grade distribution:\n{df['grade'].value_counts().sort_index().to_string()}")

    log.info("[Test] Mean y_tilde by grade:")
    grade_stats = test_final.groupby("grade", observed=False)["y_tilde"].agg(["mean", "std", "count"])
    log.info(f"\n{grade_stats.to_string()}")

    test_final.to_csv("difficulty_test_day16.csv", index=False)
    log.info("Saved: difficulty_test_day16.csv")

    return s1_model, diff_model, train_final, test_final