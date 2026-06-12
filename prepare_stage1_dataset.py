"""
prepare_stage1_dataset.py
-------------------------
离线预处理：
把原始 head/link/cross 数据转成 WRC 训练用的 shard 文件（按 part 分片保存）。

输出结构：
prepared_data/
├── manifest.json
├── stats.json
├── train/
│   ├── shard_day04_part000.pt
│   ├── shard_day04_part001.pt
│   └── ...
└── test/
    ├── shard_day15_part000.pt
    └── ...
"""

import os
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from config import (
    TRAIN_DAYS, TEST_DAYS, BATCH_SIZE_ORDERS, STATUS_CLASSES
)
from loader import load_day, load_topology
from feature_eng import (
    build_stage1_features_batch,
    STAGE1_FEATURE_COLS, STAGE1_TARGET,
)
from logger import get_logger

log = get_logger()

# ============================================================
# 路径与配置
# ============================================================
PREPARED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepared_data")
TRAIN_DIR = os.path.join(PREPARED_DIR, "train")
TEST_DIR = os.path.join(PREPARED_DIR, "test")

SCHEMA_VERSION = "stage1_seq_v3"

# ★ 核心参数：每个 shard 存多少订单
ORDERS_PER_SHARD = 20000

# 保存类型：float16 可把磁盘占用减半
SAVE_DTYPE = np.float16


# ============================================================
# 标准化参数
# ============================================================

def fit_global_stats(topo, days, sample_orders_per_day=5000):
    """
    用训练集的一小部分订单拟合全局 mean/std。
    """
    log.info("Fitting global feature statistics ...")
    sampled_parts = []

    for day in days:
        try:
            head, link, cross = load_day(day)
            sample_orders = link["order_id"].unique()[:sample_orders_per_day]

            feat = build_stage1_features_batch(
                link[link["order_id"].isin(sample_orders)],
                head[head["order_id"].isin(sample_orders)],
                cross[cross["order_id"].isin(sample_orders)],
                topo,
                keep_extra_for_stage2=False,
            )
            sampled_parts.append(feat[STAGE1_FEATURE_COLS])

            del head, link, cross, feat
            gc.collect()

            log.info(f"  Day {day}: sampled {sample_orders_per_day} orders")

            # 一般两天样本就足够稳定
            if len(sampled_parts) >= 2:
                break

        except FileNotFoundError:
            log.warning(f"  Day {day} missing, skipped in stats fitting")

    if not sampled_parts:
        raise RuntimeError("No data available to fit feature stats")

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.replace([np.inf, -np.inf], np.nan)

    mean_dict = sampled.mean(skipna=True).fillna(0.0).to_dict()
    std_dict = sampled.std(skipna=True).replace(0, 1.0).fillna(1.0).to_dict()

    del sampled_parts, sampled
    gc.collect()

    log.info("Feature stats fitted successfully.")
    return mean_dict, std_dict


# prepare_stage1_dataset.py 中唯一需要替换的函数

# 全局常量：周期特征集合（与 stage1_deep_lstmframe.py 保持完全一致）
PERIODIC_FEATURES = frozenset({
    "sin_slice", "cos_slice",
    "sin_arr_slice", "cos_arr_slice",
})

def apply_stats(df, feature_cols, mean_dict, std_dict):
    """
    标准化规则（训练/验证/推理三路完全一致）:
      - 周期特征 (sin/cos): 仅 clip 到 [-1, 1]，不做 Z-Score
      - 其余特征: Z-Score 标准化
    """
    out = df.copy()
    mean_s = pd.Series(mean_dict)
    std_s = pd.Series(std_dict)

    x = out[feature_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(mean_s)

    for col in feature_cols:
        if col in PERIODIC_FEATURES:
            # ★ 周期特征：只 clip，保留单位圆几何性质
            x[col] = x[col].clip(-1.0, 1.0)
        else:
            denom = std_s[col] if std_s[col] > 1e-8 else 1.0
            x[col] = (x[col] - mean_s[col]) / denom

    x = x.astype("float32")
    x = x.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out[feature_cols] = x
    return out


# ============================================================
# shard 保存
# ============================================================

def save_one_shard(X_buf, y_buf, len_buf, oid_buf, day, output_dir, part_idx):
    """
    保存一个 shard part。
    """
    shard = {
        "X_list": X_buf,
        "y_list": y_buf,
        "lengths": len_buf,
        "order_ids": oid_buf,
        "day": day,
        "n_orders": len(X_buf),
        "n_links": int(sum(len_buf)),
        "schema": SCHEMA_VERSION,
    }

    os.makedirs(output_dir, exist_ok=True)
    shard_path = os.path.join(output_dir, f"shard_day{day}_part{part_idx:03d}.pt")
    torch.save(shard, shard_path)

    log.info(
        f"    saved {os.path.basename(shard_path)} | "
        f"orders={len(X_buf):,} links={sum(len_buf):,}"
    )

    # 返回 manifest 信息
    info = {
        "day": day,
        "part": part_idx,
        "path": shard_path,
        "n_orders": len(X_buf),
        "n_links": int(sum(len_buf)),
    }

    del shard
    gc.collect()
    return info


# ============================================================
# 单天处理 → 多个 shard
# ============================================================

def process_day_to_shards(day, topo, mean_dict, std_dict, output_dir):
    """
    单天数据处理成多个 shard part，避免单文件过大导致 torch.save OOM。
    """
    t0 = time.time()
    log.info(f"\nProcessing day {day} ...")

    head, link, cross = load_day(day)

    unique_orders = link["order_id"].unique()
    part_idx = 0
    manifest_parts = []

    # 当前 shard 缓冲区
    X_buf, y_buf, len_buf, oid_buf = [], [], [], []

    total_orders = 0
    total_links = 0

    for i in range(0, len(unique_orders), BATCH_SIZE_ORDERS):
        batch_orders = unique_orders[i:i + BATCH_SIZE_ORDERS]

        link_b = link[link["order_id"].isin(batch_orders)]
        head_b = head[head["order_id"].isin(batch_orders)]
        cross_b = cross[cross["order_id"].isin(batch_orders)]

        feat = build_stage1_features_batch(
            link_b, head_b, cross_b, topo,
            keep_extra_for_stage2=False,
        )

        keep_cols = STAGE1_FEATURE_COLS + [STAGE1_TARGET, "order_id", "day"]
        feat = feat[[c for c in keep_cols if c in feat.columns]]

        # 过滤有效标签
        valid = feat[feat[STAGE1_TARGET].isin(STATUS_CLASSES)].copy()
        if len(valid) == 0:
            del link_b, head_b, cross_b, feat, valid
            gc.collect()
            continue

        valid["_label"] = valid[STAGE1_TARGET] - 1

        # 标准化
        valid = apply_stats(valid, STAGE1_FEATURE_COLS, mean_dict, std_dict)

        # 按订单组织序列
        grouped = valid.groupby("order_id", sort=False)
        for oid, grp in grouped:
            feats = grp[STAGE1_FEATURE_COLS].values.astype(SAVE_DTYPE)
            labels = grp["_label"].values.astype(np.int8)

            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

            X_buf.append(feats)
            y_buf.append(labels)
            len_buf.append(len(feats))
            oid_buf.append(int(oid))

            total_orders += 1
            total_links += len(feats)

            # 满了就保存一个 shard
            if len(X_buf) >= ORDERS_PER_SHARD:
                info = save_one_shard(
                    X_buf, y_buf, len_buf, oid_buf,
                    day, output_dir, part_idx
                )
                manifest_parts.append(info)
                part_idx += 1

                # 释放缓冲
                X_buf, y_buf, len_buf, oid_buf = [], [], [], []
                gc.collect()

        del link_b, head_b, cross_b, feat, valid, grouped
        gc.collect()

    # 保存最后不足一 shard 的残留
    if len(X_buf) > 0:
        info = save_one_shard(
            X_buf, y_buf, len_buf, oid_buf,
            day, output_dir, part_idx
        )
        manifest_parts.append(info)
        X_buf, y_buf, len_buf, oid_buf = [], [], [], []
        gc.collect()

    del head, link, cross
    gc.collect()

    elapsed = time.time() - t0
    log.info(
        f"Day {day} complete: {total_orders:,} orders, {total_links:,} links, "
        f"{len(manifest_parts)} shard(s), elapsed={elapsed:.1f}s"
    )

    return manifest_parts


# ============================================================
# 主函数
# ============================================================

def main():
    log.info("=" * 70)
    log.info("PREPARE STAGE 1 DATASET")
    log.info("=" * 70)

    os.makedirs(PREPARED_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    topo = load_topology()

    # 1. 拟合标准化参数
    mean_dict, std_dict = fit_global_stats(topo, TRAIN_DAYS)

    # 保存 stats
    stats_path = os.path.join(PREPARED_DIR, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "feature_cols": STAGE1_FEATURE_COLS,
            "mean": mean_dict,
            "std": std_dict,
            "save_dtype": str(SAVE_DTYPE),
            "orders_per_shard": ORDERS_PER_SHARD,
        }, f, indent=2)
    log.info(f"Stats saved: {stats_path}")

    # 2. train shards
    train_manifest = []
    log.info("\n--- TRAIN DAYS ---")
    for day in TRAIN_DAYS:
        try:
            parts = process_day_to_shards(day, topo, mean_dict, std_dict, TRAIN_DIR)
            train_manifest.extend(parts)
        except FileNotFoundError:
            log.warning(f"Day {day} missing, skipped.")

    # 3. test shards
    test_manifest = []
    log.info("\n--- TEST DAYS ---")
    for day in TEST_DAYS:
        try:
            parts = process_day_to_shards(day, topo, mean_dict, std_dict, TEST_DIR)
            test_manifest.extend(parts)
        except FileNotFoundError:
            log.warning(f"Day {day} missing, skipped.")

    # 4. manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "feature_cols": STAGE1_FEATURE_COLS,
        "target_col": STAGE1_TARGET,
        "status_classes": STATUS_CLASSES,
        "train_shards": train_manifest,
        "test_shards": test_manifest,
        "created_at": datetime.now().isoformat(),
    }

    manifest_path = os.path.join(PREPARED_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    log.info(f"\nManifest saved: {manifest_path}")

    total_train_orders = sum(s["n_orders"] for s in train_manifest)
    total_train_links = sum(s["n_links"] for s in train_manifest)
    total_test_orders = sum(s["n_orders"] for s in test_manifest)
    total_test_links = sum(s["n_links"] for s in test_manifest)

    log.info("\n" + "=" * 70)
    log.info("PREPARE COMPLETE")
    log.info(f"Train shards: {len(train_manifest)} | orders={total_train_orders:,} | links={total_train_links:,}")
    log.info(f"Test  shards: {len(test_manifest)} | orders={total_test_orders:,} | links={total_test_links:,}")
    log.info(f"Prepared data dir: {PREPARED_DIR}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()