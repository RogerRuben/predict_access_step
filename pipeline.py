"""
pipeline.py — 批次处理版（解决内存问题）
"""

import gc
import pandas as pd
import numpy as np
from config import (
    TRAIN_DAYS, TEST_DAYS, CROSS_TIME_QUANTILE,
    TRAIN_SAMPLE_RATE, BATCH_SIZE_ORDERS
)
from loader import load_day, load_heads_only, load_topology, load_split_files
from feature_eng import build_stage1_features_batch, STAGE1_FEATURE_COLS, STAGE1_TARGET
from stage1_predict import prepare_training_data, train_model, predict_proba
from stage2_assess import (
    compute_deterministic, compute_uncertainty, compute_night,
    STAGE2_ALL_COLS,
)
from stage3_anchor import build_supervision_signal, DifficultyModel


def run_pipeline():
    topo = load_topology()

    # ===================== Stage 1: 逐批提取训练样本 =====================
    print("\n" + "="*80)
    print(" STAGE 1: Traffic State Prediction (Batch by Orders)")
    print("="*80)

    X_parts, y_parts = [], []
    for day in TRAIN_DAYS:
        try:
            print(f"\n[stage1] Processing day {day} ...")
            head, link, cross = load_day(day)

            # 按 order_id 分批处理
            unique_orders = link["order_id"].unique()
            for i in range(0, len(unique_orders), BATCH_SIZE_ORDERS):
                batch_orders = unique_orders[i:i + BATCH_SIZE_ORDERS]
                link_batch = link[link["order_id"].isin(batch_orders)]
                head_batch = head[head["order_id"].isin(batch_orders)]
                cross_batch = cross[cross["order_id"].isin(batch_orders)]

                batch_feat = build_stage1_features_batch(link_batch, head_batch, cross_batch, topo)
                valid = batch_feat[batch_feat[STAGE1_TARGET].isin([1,2,3,4])]

                if len(valid) > 0:
                    X_parts.append(valid[STAGE1_FEATURE_COLS].values.astype("float32"))
                    y_parts.append((valid[STAGE1_TARGET].values - 1).astype("int32"))

                del batch_feat, valid, link_batch, head_batch, cross_batch
                gc.collect()

            del head, link, cross
            gc.collect()
            print(f"         day {day} processed.")

        except FileNotFoundError:
            print(f"[stage1] day {day} missing, skipped.")

    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(y_parts)
    del X_parts, y_parts
    gc.collect()

    print(f"\n[stage1] Total training samples: {len(y_all):,}")

    if TRAIN_SAMPLE_RATE < 1.0:
        n = int(len(y_all) * TRAIN_SAMPLE_RATE)
        idx = np.random.RandomState(42).choice(len(y_all), n, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]
        print(f"[stage1] Sampled to {n:,} samples ({TRAIN_SAMPLE_RATE*100:.0f}%)")

    print("[stage1] Training LightGBM...")
    model = train_model(X_all, y_all)
    del X_all, y_all
    gc.collect()

    # ===================== Stage 2 & 3 =====================
    print("\n" + "="*80)
    print(" STAGE 2 & 3: Difficulty Assessment + Anchoring")
    print("="*80)

    # 计算全局 cross_time 阈值
    all_ct = []
    for day in TRAIN_DAYS:
        try:
            c = load_split_files("cross", day)   # 使用 loader 中的函数
            all_ct.append(c["cross_time"].values)
        except:
            pass
    ct_threshold = float(np.percentile(np.concatenate(all_ct), CROSS_TIME_QUANTILE*100))
    print(f"[stage2] cross_time P{int(CROSS_TIME_QUANTILE*100)} = {ct_threshold:.2f}s")

    # 训练集 Stage 2
    train_s2_list, train_head_list = [], []
    for day in TRAIN_DAYS:
        try:
            s2, head = _process_day_stage2(day, topo, model, ct_threshold)
            train_s2_list.append(s2)
            train_head_list.append(head[["order_id", "day", "ata", "simple_eta", "driver_id"]])
        except FileNotFoundError:
            continue

    train_s2 = pd.concat(train_s2_list, ignore_index=True)
    train_head_slim = pd.concat(train_head_list, ignore_index=True)

    # 测试集
    test_s2_list, test_head_list = [], []
    for day in TEST_DAYS:
        try:
            s2, head = _process_day_stage2(day, topo, model, ct_threshold)
            test_s2_list.append(s2)
            test_head_list.append(head[["order_id", "day", "ata", "simple_eta", "driver_id"]])
        except FileNotFoundError:
            continue

    test_s2 = pd.concat(test_s2_list, ignore_index=True)
    test_head_slim = pd.concat(test_head_list, ignore_index=True)

    # Stage 3
    train_signal = build_supervision_signal(train_head_slim)
    test_signal = build_supervision_signal(test_head_slim)

    train_final = train_s2.merge(train_signal, on=["order_id", "day"], how="inner")
    test_final = test_s2.merge(test_signal, on=["order_id", "day"], how="inner")

    diff_model = DifficultyModel()
    diff_model.fit(train_final, train_final["y_tilde"])
    diff_model.evaluate(test_final, test_final["y_tilde"])

    train_final["difficulty"] = diff_model.predict(train_final)
    test_final["difficulty"] = diff_model.predict(test_final)

    # 分级
    q30 = train_final["difficulty"].quantile(0.30)
    q70 = train_final["difficulty"].quantile(0.70)
    q90 = train_final["difficulty"].quantile(0.90)

    for df, name in [(train_final, "Train"), (test_final, "Test")]:
        df["grade"] = pd.cut(
            df["difficulty"],
            bins=[-np.inf, q30, q70, q90, np.inf],
            labels=["G1_AV_Easy", "G2_AV_Moderate", "G3_AV_Hard", "G4_HV_Only"],
        )
        print(f"\n[{name}] Grade distribution:\n{df['grade'].value_counts().sort_index()}")

    test_final.to_csv("difficulty_test_day16.csv", index=False)
    print("\nSaved: difficulty_test_day16.csv")
    return model, diff_model, train_final, test_final


def _process_day_stage2(day, topo, model, ct_threshold):
    """单日 Stage 2 处理"""
    head, link, cross = load_day(day)
    link_feat = build_stage1_features_batch(link, head, cross, topo)   # 使用 batch 版
    link_pred = predict_proba(model, link_feat)

    det = compute_deterministic(link_pred, cross, topo, ct_threshold)
    unc = compute_uncertainty(link_pred)
    night = compute_night(head)

    s2 = det.merge(unc, on=["order_id", "day"], how="outer")
    s2 = s2.merge(night, on=["order_id", "day"], how="outer")
    s2[STAGE2_ALL_COLS] = s2[STAGE2_ALL_COLS].fillna(0)

    del link, cross, link_feat, link_pred
    gc.collect()
    return s2, head