"""
pipeline.py — 完整版
集成功能：
1. Stage 1 训练 / 加载
2. ★ 新增 Stage 1 test evaluation（train/test 模式都可打印）
3. Stage 2 缓存 + 订单级难度特征
4. Stage 3 难度锚定与评估
"""

import gc
import os
import glob
import hashlib
import pandas as pd
import numpy as np

from config import (
    TRAIN_DAYS, TEST_DAYS,
    CROSS_TIME_QUANTILE,
    TRAIN_SAMPLE_RATE,
    BATCH_SIZE_ORDERS,
    MODEL_DIR, CACHE_DIR,
    STAGE1_MODEL,
    WRC_HIDDEN_DIM, WRC_NUM_LAYERS, WRC_BATCH_SIZE,
    WRC_EPOCHS, WRC_LR, WRC_MAX_SEQ_LEN,
)

from loader import load_day, load_split_files, load_topology
from feature_eng import (
    build_stage1_features_batch,
    STAGE1_FEATURE_COLS, STAGE1_TARGET,
)
from stage2_assess import (
    compute_deterministic,
    compute_uncertainty,
    compute_safety_risk,
    compute_night,
    STAGE2_ALL_COLS,
)
from stage3_anchor import build_supervision_signal, DifficultyModel
from logger import get_logger

log = get_logger()


# ============================================================
# 工具函数
# ============================================================

def _find_latest_model(prefix: str, ext: str) -> str:
    pattern = os.path.join(MODEL_DIR, f"{prefix}*{ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No saved model: {pattern}")
    latest = files[-1]
    log.info(f"Found model: {latest}")
    return latest


def _make_cache_key(day, s1_path, ct_thresh, cross_mean):
    """
    Stage 2 缓存 key。
    注意：如果你改了 Stage1 的推理逻辑但没改模型路径，请手动清 cache。
    """
    raw = f"{day}|{os.path.basename(s1_path)}|{ct_thresh:.4f}|{cross_mean:.4f}|v7_stage1_eval"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _cache_paths(cache_key):
    return (
        os.path.join(CACHE_DIR, f"s2_{cache_key}.pkl"),
        os.path.join(CACHE_DIR, f"head_{cache_key}.pkl"),
    )


# ============================================================
# Stage 2: 单天处理（带缓存）
# ============================================================

def _process_day_stage2(day, topo, model, ct_threshold, cross_global_mean,
                        s1_model_path=""):
    cache_key = _make_cache_key(day, s1_model_path, ct_threshold, cross_global_mean)
    s2_cache, head_cache = _cache_paths(cache_key)

    if os.path.exists(s2_cache) and os.path.exists(head_cache):
        log.info(f"  Day {day}: cache HIT ({cache_key})")
        return pd.read_pickle(s2_cache), pd.read_pickle(head_cache)

    log.info(f"  Day {day}: cache MISS, computing ...")

    head, link, cross = load_day(day)
    unique_orders = link["order_id"].unique()

    s2_parts = []
    det_unc_cols = [c for c in STAGE2_ALL_COLS if c not in ("D11_night", "R1_max_p4", "R2_reject_ratio")]

    # 选择预测函数
    if STAGE1_MODEL in ("wrc", "wdr", "hierarchical", "hier", "ordinal", "dualhead", "triple"):
        from stage1_deep import predict_proba_wrc
        predict_fn = predict_proba_wrc
    else:
        from stage1_predict import predict_proba
        predict_fn = predict_proba

    for i in range(0, len(unique_orders), BATCH_SIZE_ORDERS):
        batch_orders = unique_orders[i:i + BATCH_SIZE_ORDERS]
        link_b  = link[link["order_id"].isin(batch_orders)]
        head_b  = head[head["order_id"].isin(batch_orders)]
        cross_b = cross[cross["order_id"].isin(batch_orders)]

        feat = build_stage1_features_batch(
            link_b, head_b, cross_b, topo,
            keep_extra_for_stage2=True,
        )
        pred = predict_fn(model, feat)

        det  = compute_deterministic(pred, cross_b, topo, ct_threshold, cross_global_mean)
        unc  = compute_uncertainty(pred)
        risk = compute_safety_risk(pred)

        batch_s2 = det.merge(unc, on=["order_id", "day"], how="outer")
        batch_s2 = batch_s2.merge(risk, on=["order_id", "day"], how="outer")

        for c in det_unc_cols:
            if c in batch_s2.columns:
                batch_s2[c] = batch_s2[c].fillna(0)

        s2_parts.append(batch_s2)

        del feat, pred, det, unc, risk, batch_s2, link_b, head_b, cross_b
        gc.collect()

    s2_all = pd.concat(s2_parts, ignore_index=True)
    del s2_parts, link, cross
    gc.collect()

    # D11
    night = compute_night(head)
    s2_all = s2_all.merge(night, on=["order_id", "day"], how="left")

    for c in STAGE2_ALL_COLS:
        if c in s2_all.columns:
            s2_all[c] = s2_all[c].fillna(0)

    head_slim = head[["order_id", "day", "ata", "simple_eta", "driver_id"]].copy()

    del head, night
    gc.collect()

    s2_all.to_pickle(s2_cache)
    head_slim.to_pickle(head_cache)
    log.info(f"  Day {day}: cached ({cache_key})")

    return s2_all, head_slim


# ============================================================
# Stage 1: 训练
# ============================================================

def _run_stage1_train(topo):
    log.info("=" * 70)
    log.info(f"STAGE 1: Training (model={STAGE1_MODEL})")
    log.info("=" * 70)

    if STAGE1_MODEL in ("wrc", "wdr", "hierarchical", "hier", "ordinal", "dualhead", "triple"):
        from stage1_deep import train_wrc_from_shards, save_wrc_model

        model = train_wrc_from_shards(
            hidden_dim=WRC_HIDDEN_DIM,
            num_layers=WRC_NUM_LAYERS,
            batch_size=WRC_BATCH_SIZE,
            epochs=WRC_EPOCHS,
            lr=WRC_LR,
            max_seq_len=WRC_MAX_SEQ_LEN,
        )
        model_path = save_wrc_model(model, tag="v1")
        return model, model_path

    elif STAGE1_MODEL == "lgbm":
        from stage1_predict import train_model, save_stage1_model

        X_parts, y_parts = [], []
        for day in TRAIN_DAYS:
            try:
                log.info(f"Extracting day {day} ...")
                head, link, cross = load_day(day)
                unique_orders = link["order_id"].unique()
                day_samples = 0

                for i in range(0, len(unique_orders), BATCH_SIZE_ORDERS):
                    batch_orders = unique_orders[i:i + BATCH_SIZE_ORDERS]
                    feat = build_stage1_features_batch(
                        link[link["order_id"].isin(batch_orders)],
                        head[head["order_id"].isin(batch_orders)],
                        cross[cross["order_id"].isin(batch_orders)],
                        topo,
                        keep_extra_for_stage2=False,
                    )
                    valid = feat[feat[STAGE1_TARGET].isin([1, 2, 3, 4])]
                    if len(valid) > 0:
                        X_parts.append(valid[STAGE1_FEATURE_COLS].values.astype("float32"))
                        y_parts.append((valid[STAGE1_TARGET].values - 1).astype("int32"))
                        day_samples += len(valid)
                    del feat, valid
                    gc.collect()

                del head, link, cross
                gc.collect()
                log.info(f"  Day {day}: {day_samples:,} samples")
            except FileNotFoundError:
                log.warning(f"Day {day} missing, skipped.")

        if not X_parts:
            raise RuntimeError("No training data for LightGBM")

        X_all = np.concatenate(X_parts)
        y_all = np.concatenate(y_parts)
        del X_parts, y_parts
        gc.collect()

        if TRAIN_SAMPLE_RATE < 1.0:
            n = int(len(y_all) * TRAIN_SAMPLE_RATE)
            idx = np.random.RandomState(42).choice(len(y_all), n, replace=False)
            X_all, y_all = X_all[idx], y_all[idx]

        model = train_model(X_all, y_all)
        model_path = save_stage1_model(model, tag="v3")
        del X_all, y_all
        gc.collect()
        return model, model_path

    else:
        raise ValueError(
            f"Unknown STAGE1_MODEL='{STAGE1_MODEL}'. "
            f"Supported: 'wdr', 'wrc', 'hier', 'hierarchical', "
            f"'ordinal', 'dualhead', 'triple', 'lgbm'."
        )


# ============================================================
# ★ 新增：Stage 1 单独评估
# ============================================================

def _build_stage1_eval_criterion(stage1_module, train_shards, device):
    """
    根据 stage1_deep.py 里实际存在的类，动态构建评估 criterion。
    兼容当前所有版本。
    """
    # 统一获取权重
    if hasattr(stage1_module, "estimate_weights"):
        result = stage1_module.estimate_weights(
            train_shards, min(5, len(train_shards))
        )
        # estimate_weights 可能返回 4 或 5 个值
        if len(result) == 5:
            pw_q1, pw_q2, pw_q3, cls_alpha, _ = result
        else:
            pw_q1, pw_q2, pw_q3, cls_alpha = result
    else:
        pw_q1, pw_q2, pw_q3, cls_alpha = 1.0, 2.0, 4.0, np.ones(4, dtype=np.float32)

    # 按优先级尝试不同的 loss 类
    # OrdinalPrimaryLoss（当前最新版）
    if hasattr(stage1_module, "OrdinalPrimaryLoss"):
        criterion = stage1_module.OrdinalPrimaryLoss(
            pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
            cls_alpha=cls_alpha,
        ).to(device)
        return criterion

    # TripleExpertLoss
    if hasattr(stage1_module, "TripleExpertLoss"):
        criterion = stage1_module.TripleExpertLoss(
            pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
            cls_alpha=cls_alpha,
            gamma_q1=0.0, gamma_q2=2.0, gamma_q3=2.0,
            gamma_cls=2.0, gamma_bnd=1.5,
            lambda_cls=1.0, lambda_bnd=0.8, lambda_cons=0.0,
        ).to(device)
        return criterion

    # HybridOrdinalClassificationLoss
    if hasattr(stage1_module, "HybridOrdinalClassificationLoss"):
        criterion = stage1_module.HybridOrdinalClassificationLoss(
            pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
            cls_alpha=cls_alpha,
        ).to(device)
        return criterion

    # FocalOrdinalLoss
    if hasattr(stage1_module, "FocalOrdinalLoss"):
        criterion = stage1_module.FocalOrdinalLoss(
            pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
        ).to(device)
        return criterion

    # CumulativeOrdinalLoss
    if hasattr(stage1_module, "CumulativeOrdinalLoss"):
        criterion = stage1_module.CumulativeOrdinalLoss(
            pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
        ).to(device)
        return criterion

    raise RuntimeError(
        "Cannot build Stage 1 evaluation criterion. "
        "No recognized loss class found in stage1_deep.py. "
        f"Available: {[x for x in dir(stage1_module) if 'Loss' in x]}"
    )


def _run_stage1_test_evaluation(s1_model):
    """
    在 pipeline 中单独评估 Stage 1，打印 confusion matrix / recall / DMR。
    """
    if STAGE1_MODEL not in ("wrc", "wdr", "hierarchical", "hier", "ordinal", "dualhead", "triple"):
        return

    log.info("=" * 70)
    log.info("STAGE 1: Test Evaluation")
    log.info("=" * 70)

    try:
        import stage1_deep as stage1_module

        train_shards, val_shards, test_shards = stage1_module.split_train_val_shards()

        eval_shards = test_shards if len(test_shards) > 0 else val_shards
        eval_name = "TEST" if len(test_shards) > 0 else "VAL(fallback)"

        criterion = _build_stage1_eval_criterion(stage1_module, train_shards, stage1_module.DEVICE)

        tau = float(s1_model.best_tau.item()) if hasattr(s1_model, "best_tau") else 0.55

        log.info(f"Evaluating Stage 1 on {eval_name} shards: {len(eval_shards)} | tau={tau:.2f}")

        stage1_eval = stage1_module.evaluate_on_shards(
            s1_model,
            eval_shards,
            device=stage1_module.DEVICE,
            criterion=criterion,
            batch_size=WRC_BATCH_SIZE * 2,
            max_seq_len=WRC_MAX_SEQ_LEN,
            verbose=True,
            tau=tau,
        )

        log.info(
            f"[Stage1-{eval_name}] "
            f"Integrated mF1={stage1_eval['macro_f1']:.4f} | "
            f"DMR={stage1_eval['dangerous_miss']:.4f} | "
            f"s4u={stage1_eval['s4_underestimate']:.4f}"
        )
        log.info(
            f"[Stage1-{eval_name}] Integrated Recall: "
            f"{[round(x, 4) for x in stage1_eval['recall_per_class']]}"
        )
        if "cls_recall" in stage1_eval:
            log.info(
                f"[Stage1-{eval_name}] Classifier Recall: "
                f"{[round(x, 4) for x in stage1_eval['cls_recall']]}"
            )
        log.info(f"[Stage1-{eval_name}] Reject rate: {stage1_eval['reject_rate']:.4f}")

    except Exception as e:
        log.warning(f"Stage 1 evaluation skipped due to error: {repr(e)}")


# ============================================================
# 全局统计量
# ============================================================

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

    log.info(f"  cross_time P{int(CROSS_TIME_QUANTILE * 100)} = {ct_threshold:.2f}s")
    log.info(f"  cross_time global_mean = {cross_global_mean:.2f}s")
    return ct_threshold, cross_global_mean


# ============================================================
# 主流程
# ============================================================

def run_pipeline(mode: str = "train"):
    topo = load_topology()

    # ---- Stage 1: train or load ----
    if mode == "train":
        s1_model, s1_model_path = _run_stage1_train(topo)
    else:
        if STAGE1_MODEL in ("wrc", "wdr", "hierarchical", "hier", "ordinal", "dualhead", "triple"):
            import stage1_deep as stage1_module

            # 优先级：triple > dualhead > ordinal > hier > wdr
            for prefix in ["stage1_ordpri", "stage1_triple", "stage1_dualhead", "stage1_ordinal"]:
                try:
                    s1_model_path = _find_latest_model(prefix, ".pt")
                    break
                except FileNotFoundError:
                    continue
            else:
                raise FileNotFoundError("No Stage 1 deep model found in saved_models/")

            s1_model = stage1_module.load_wrc_model(s1_model_path)
        else:
            from stage1_predict import load_stage1_model
            s1_model_path = _find_latest_model("stage1_lgbm", ".txt")
            s1_model = load_stage1_model(s1_model_path)

    # ---- ★ 新增：Stage 1 test evaluation ----
    _run_stage1_test_evaluation(s1_model)

    # ---- Stage 2/3 ----
    ct_threshold, cross_global_mean = _compute_global_cross_stats(TRAIN_DAYS)

    # Stage 2 train
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

    # Stage 2 test
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

    # Stage 3
    log.info("=" * 70)
    log.info("STAGE 3: Difficulty Anchoring")
    log.info("=" * 70)

    if mode == "train":
        train_signal = build_supervision_signal(train_head_slim)
        train_final = train_s2.merge(train_signal, on=["order_id", "day"], how="inner")
        del train_s2, train_head_slim
        gc.collect()

        log.info(f"Train samples: {len(train_final):,}")

        log.info("\nStage 2 feature statistics (train):")
        for col in STAGE2_ALL_COLS:
            if col in train_final.columns:
                vals = train_final[col]
                log.info(
                    f"  {col:25s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
                    f"min={vals.min():.4f}  max={vals.max():.4f}  "
                    f"zeros={int((vals == 0).sum()):,}/{len(vals):,}"
                )

        diff_model = DifficultyModel()
        diff_model.fit(train_final, train_final["y_tilde"])
        diff_model.save(tag="v5_R1R2")

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

    # ---- 分级输出 ----
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
        log.info(f"\n[{name}] Grade distribution:\n{df['grade'].value_counts().sort_index().to_string()}")

    log.info("\n[Test] Mean y_tilde by grade:")
    grade_stats = test_final.groupby("grade", observed=False)["y_tilde"].agg(["mean", "std", "count"])
    log.info(f"\n{grade_stats.to_string()}")

    test_final.to_csv("difficulty_test.csv", index=False)
    log.info("Saved: difficulty_test.csv")

    return s1_model, diff_model, train_final, test_final