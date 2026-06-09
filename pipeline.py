"""
pipeline.py — 修复报错 + 适配新特征
"""
import gc
import os
import glob
import pandas as pd
import numpy as np
from config import (
    TRAIN_DAYS, TEST_DAYS, CROSS_TIME_QUANTILE,
    TRAIN_SAMPLE_RATE, BATCH_SIZE_ORDERS, MODEL_DIR,
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


def _find_latest_model(prefix: str, ext: str) -> str:
    pattern = os.path.join(MODEL_DIR, f"{prefix}*{ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No saved model: {pattern}")
    return files[-1]


def _process_day_stage2(day, topo, model, ct_threshold):
    """单天 Stage 2: 分批构建特征(保留 link_id) → 预测 → 聚合难度"""
    head, link, cross = load_day(day)

    unique_orders = link["order_id"].unique()
    pred_parts = []

    for i in range(0, len(unique_orders), BATCH_SIZE_ORDERS):
        batch_orders = unique_orders[i:i + BATCH_SIZE_ORDERS]
        link_b = link[link["order_id"].isin(batch_orders)]
        head_b = head[head["order_id"].isin(batch_orders)]
        cross_b = cross[cross["order_id"].isin(batch_orders)]

        feat = build_stage1_features_batch(
            link_b, head_b, cross_b, topo,
            keep_extra_for_stage2=True,  # ★ 保留 link_id 等列
        )
        pred = predict_proba(model, feat)
        pred_parts.append(pred)

        del feat, link_b, head_b, cross_b
        gc.collect()

    link_pred = pd.concat(pred_parts, ignore_index=True)
    del pred_parts, link
    gc.collect()

    det = compute_deterministic(link_pred, cross, topo, ct_threshold)
    unc = compute_uncertainty(link_pred)
    night = compute_night(head)

    s2 = det.merge(unc, on=["order_id", "day"], how="outer")
    s2 = s2.merge(night, on=["order_id", "day"], how="outer")
    s2[STAGE2_ALL_COLS] = s2[STAGE2_ALL_COLS].fillna(0)

    del link_pred, cross, det, unc, night
    gc.collect()

    return s2, head


def run_pipeline(mode: str = "train"):
    topo = load_topology()

    if mode == "train":
        s1_model = _run_stage1_train(topo)
    else:
        s1_path = _find_latest_model("stage1_lgbm", ".txt")
        s1_model = load_stage1_model(s1_path)

    # ---- cross_time 阈值 ----
    log.info("Computing global cross_time threshold ...")
    ct_vals = []
    for day in TRAIN_DAYS:
        try:
            c = load_split_files("cross", day)
            ct_vals.append(c["cross_time"].values)
            del c
        except FileNotFoundError:
            pass
    if ct_vals:
        ct_threshold = float(np.percentile(np.concatenate(ct_vals), CROSS_TIME_QUANTILE * 100))
    else:
        ct_threshold = 30.0
    del ct_vals
    gc.collect()
    log.info(f"cross_time P{int(CROSS_TIME_QUANTILE*100)} = {ct_threshold:.2f}s")

    # ---- Stage 2: Train ----
    if mode == "train":
        log.info("=" * 70)
        log.info("STAGE 2: Difficulty Assessment (Train)")
        log.info("=" * 70)
        train_s2_list, train_head_list = [], []
        for day in TRAIN_DAYS:
            try:
                log.info(f"Stage 2 train day {day} ...")
                s2, head = _process_day_stage2(day, topo, s1_model, ct_threshold)
                train_s2_list.append(s2)
                train_head_list.append(head[["order_id", "day", "ata", "simple_eta", "driver_id"]])
                del s2, head
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
            log.info(f"Stage 2 test day {day} ...")
            s2, head = _process_day_stage2(day, topo, s1_model, ct_threshold)
            test_s2_list.append(s2)
            test_head_list.append(head[["order_id", "day", "ata", "simple_eta", "driver_id"]])
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

        diff_model = DifficultyModel()
        diff_model.fit(train_final, train_final["y_tilde"])
        diff_model.save(tag="v2")

        train_final["difficulty"] = diff_model.predict(train_final)
    else:
        s3_path = _find_latest_model("stage3_difficulty", ".pkl")
        diff_model = DifficultyModel.load(s3_path)
        train_final = None

    test_signal = build_supervision_signal(test_head_slim)
    test_final = test_s2.merge(test_signal, on=["order_id", "day"], how="inner")
    del test_s2, test_head_slim
    gc.collect()

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

    dfs = []
    if train_final is not None:
        dfs.append((train_final, "Train"))
    dfs.append((test_final, "Test"))

    for df, name in dfs:
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
    save_stage1_model(model, tag="v2")

    del X_all, y_all
    gc.collect()

    return model