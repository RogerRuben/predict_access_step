"""
prepare_stage1_dataset.py — 完整版（增量生成 + 稳健诊断）

功能:
  1. 所有天的 shard 统一放 prepared_data/shards/（不区分 train/test）
  2. 已存在的 shard 自动跳过，只生成新的
  3. stats.json 如已存在则复用，不重新拟合
  4. manifest.json 记录所有已生成的 shard（train/test 划分延迟到训练时）
  5. 稳健的天数发现逻辑 + 详细错误日志
  6. 周期特征 (sin/cos) 不做 Z-Score，只 clip 到 [-1, 1]

用法:
  # 首次运行（生成所有可用天的 shard）
  python prepare_stage1_dataset.py

  # 新增数据后增量运行（只生成新天的 shard）
  python prepare_stage1_dataset.py

  # 改了 config.py 的 TRAIN_DAYS / TEST_DAYS 后，无需重新运行
  # 直接 python main.py train 即可

输出:
  prepared_data/
  ├── stats.json            ← 标准化参数（只生成一次）
  ├── manifest.json         ← 所有已生成 shard 的注册表
  └── shards/               ← 统一目录
      ├── shard_day01_part000.pt
      ├── shard_day01_part001.pt
      ├── shard_day04_part000.pt
      └── ...
"""

import os
import gc
import glob
import json
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from config import (
    BATCH_SIZE_ORDERS, STATUS_CLASSES,
    TRAIN_DAYS, TEST_DAYS,
    DATA_DIR, SPLIT_SUBDIRS,
)
from loader import load_day, load_topology
from feature_eng import (
    build_stage1_features_batch,
    STAGE1_FEATURE_COLS, STAGE1_TARGET,
)
from logger import get_logger
import re
log = get_logger()

from stage1_feature_schema import (
    RAW_ID_FEATURES,
    ORDINAL_RAW_FEATURES,
    PERIODIC_FEATURES,
    NON_STANDARDIZE_FEATURES,
    ID_CLIP_RANGES,
)
# ★ 从 stage1_deep 导入 transform，或在此处复制

# ============================================================
# 路径与常量
# ============================================================

PREPARED_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepared_data")
SHARD_DIR     = os.path.join(PREPARED_DIR, "shards")
STATS_PATH    = os.path.join(PREPARED_DIR, "stats.json")
MANIFEST_PATH = os.path.join(PREPARED_DIR, "manifest.json")

SCHEMA_VERSION   = "stage1_seq_v4"
ORDERS_PER_SHARD = 20000
SAVE_DTYPE       = np.float16

# 周期特征不做 Z-Score，只 clip
PERIODIC_FEATURES = frozenset({
    "sin_slice", "cos_slice",
    "sin_arr_slice", "cos_arr_slice",
})


# ============================================================
# 天数发现与校验
# ============================================================

def _extract_days_from_folder(ftype: str):
    """
    从文件名中提取 day。
    兼容：
      head01_1.csv
      head_01_1.csv
      head1_1.csv
      head_1_1.csv
      head01.csv
      head_01.csv
    返回统一两位字符串，如 '01', '15'
    """
    folder = os.path.join(DATA_DIR, SPLIT_SUBDIRS[ftype])
    if not os.path.exists(folder):
        log.warning(f"{ftype} folder not found: {folder}")
        return set()

    files = os.listdir(folder)
    days = set()

    # ★ 核心修复：prefix 后的下划线改为可选
    pat = re.compile(rf"^{ftype}_?(\d{{1,2}})(?:_|\.|$)")

    for fn in files:
        m = pat.match(fn)
        if m:
            day = m.group(1).zfill(2)
            days.add(day)

    sample = sorted(files)[:5]
    log.info(f"  {ftype:6s} sample files: {sample}")
    log.info(f"  {ftype:6s} discovered days: {sorted(days)}")
    return days


def _discover_available_days():
    """
    更稳健的日期发现逻辑：
    1. 直接从 head/link/cross 三个目录的真实文件名解析 day
    2. 取三者交集作为 truly available days
    3. 再与 config.py 中的 TRAIN_DAYS + TEST_DAYS 取交集
    """
    log.info("Discovering available days from real filenames ...")

    # 目录信息
    for ftype in ["head", "link", "cross"]:
        folder = os.path.join(DATA_DIR, SPLIT_SUBDIRS[ftype])
        exists = os.path.exists(folder)
        n_files = len(os.listdir(folder)) if exists else 0
        status = f"✓ ({n_files} files)" if exists else "✗ NOT FOUND"
        log.info(f"  {ftype:6s} dir: {folder} {status}")

    # 从真实文件名解析 day
    head_days  = _extract_days_from_folder("head")
    link_days  = _extract_days_from_folder("link")
    cross_days = _extract_days_from_folder("cross")

    # 三者交集 = 真正同时具备 head/link/cross 的天
    discovered = head_days & link_days & cross_days

    config_days = set(TRAIN_DAYS + TEST_DAYS)
    available = sorted(discovered & config_days)

    log.info(f"  config candidate days: {sorted(config_days)}")
    log.info(f"  head ∩ link ∩ cross : {sorted(discovered)}")
    log.info(f"  final available days: {available}")

    if not available:
        log.error(
            "No available days found after intersecting:\n"
            f"  head_days  = {sorted(head_days)}\n"
            f"  link_days  = {sorted(link_days)}\n"
            f"  cross_days = {sorted(cross_days)}\n"
            f"  config_days= {sorted(config_days)}\n"
            "Please inspect file naming convention."
        )

    return available



# ============================================================
# 标准化参数拟合
# ============================================================

def fit_global_stats(topo, available_days, sample_per_day=5000):
    """
    用训练集的一小部分订单拟合全局 mean/std。
    失败时输出详细的逐天诊断信息。
    """
    log.info("Fitting global feature statistics ...")

    if not available_days:
        raise RuntimeError(
            "No candidate days available for fitting stats.\n"
            f"  DATA_DIR = {os.path.abspath(DATA_DIR)}\n"
            f"  head dir = {os.path.join(DATA_DIR, SPLIT_SUBDIRS['head'])}\n"
            f"  link dir = {os.path.join(DATA_DIR, SPLIT_SUBDIRS['link'])}\n"
            f"  cross dir = {os.path.join(DATA_DIR, SPLIT_SUBDIRS['cross'])}\n"
            "Please check your data and config.py."
        )

    sampled_parts = []
    failed_days = []

    for day in available_days:
        try:
            log.info(f"  Trying day {day} ...")
            head, link, cross = load_day(day)

            unique_orders = link["order_id"].unique()
            if len(unique_orders) == 0:
                log.warning(f"  Day {day}: no orders in link table, skipped.")
                failed_days.append((day, "no orders"))
                del head, link, cross
                gc.collect()
                continue

            sample_orders = unique_orders[:sample_per_day]

            feat = build_stage1_features_batch(
                link[link["order_id"].isin(sample_orders)],
                head[head["order_id"].isin(sample_orders)],
                cross[cross["order_id"].isin(sample_orders)],
                topo,
                keep_extra_for_stage2=False,
            )

            if feat.empty or len(feat) == 0:
                log.warning(f"  Day {day}: feature table empty, skipped.")
                failed_days.append((day, "empty features"))
                del head, link, cross, feat
                gc.collect()
                continue

            sampled_parts.append(feat[STAGE1_FEATURE_COLS])
            log.info(f"  Day {day}: sampled {min(sample_per_day, len(sample_orders))} orders, "
                     f"{len(feat):,} link rows")

            del head, link, cross, feat
            gc.collect()

            # 两天样本通常足够稳定
            if len(sampled_parts) >= 2:
                break

        except FileNotFoundError as e:
            log.warning(f"  Day {day}: FileNotFoundError → {e}")
            failed_days.append((day, f"file not found: {e}"))
        except Exception as e:
            log.warning(f"  Day {day}: unexpected error → {repr(e)}")
            failed_days.append((day, repr(e)))

    if not sampled_parts:
        lines = ["No data available to fit stats."]
        lines.append(f"Candidate days: {available_days}")
        if failed_days:
            lines.append("Per-day failures:")
            for d, err in failed_days:
                lines.append(f"  day {d}: {err}")
        raise RuntimeError("\n".join(lines))

    combined = pd.concat(sampled_parts, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan)

    mean_dict = combined.mean(skipna=True).fillna(0.0).to_dict()
    std_dict  = combined.std(skipna=True).replace(0, 1.0).fillna(1.0).to_dict()

    del sampled_parts, combined
    gc.collect()

    log.info("Feature stats fitted successfully.")
    return mean_dict, std_dict


# ============================================================
# 标准化应用
# ============================================================

# 定义跳过标准化的特征（ID类）
SKIP_FEATURES = {"link_id", "slice_id", "arrival_slice_est"}

# 定义跳过标准化的特征（ID 类 + 状态类）
SKIP_FEATURES = {"link_id", "slice_id", "arrival_slice_est", "link_current_status"}

# ★ 导入轻量 transform
from stage1_feature_transform import transform_stage1_feature_frame

def apply_stats(df, feature_cols, mean_dict, std_dict):
    """
    返回 DataFrame，保留 order_id/day/_label 等非特征列。
    """
    return transform_stage1_feature_frame(
        df=df,
        feature_cols=feature_cols,
        mean_dict=mean_dict,
        std_dict=std_dict,
    )
#

# ============================================================
# manifest 工具
# ============================================================

def load_existing_manifest():
    """加载已有 manifest，不存在则返回空结构。"""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        n_shards = len(manifest.get("shards", []))
        existing_days = sorted(set(s["day"] for s in manifest.get("shards", [])))
        log.info(f"Existing manifest: {n_shards} shard(s), days={existing_days}")
        return manifest
    log.info("No existing manifest found, will create new one.")
    return {"schema_version": SCHEMA_VERSION, "shards": []}


def get_existing_days(manifest):
    """从 manifest 中提取已生成 shard 的天集合。"""
    return set(s["day"] for s in manifest.get("shards", []))


def save_manifest(manifest):
    """保存 manifest。"""
    manifest["updated_at"] = datetime.now().isoformat()
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["feature_cols"] = STAGE1_FEATURE_COLS
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info(f"Manifest saved: {MANIFEST_PATH}")


# ============================================================
# 单天 → 多个 shard
# ============================================================

def _save_one_shard(X_buf, y_buf, len_buf, oid_buf, day, part_idx):
    """保存一个 shard part 文件。"""
    shard = {
        "X_list":    X_buf,
        "y_list":    y_buf,
        "lengths":   len_buf,
        "order_ids": oid_buf,
        "day":       day,
        "n_orders":  len(X_buf),
        "n_links":   int(sum(len_buf)),
        "schema":    SCHEMA_VERSION,
    }

    os.makedirs(SHARD_DIR, exist_ok=True)
    shard_path = os.path.join(SHARD_DIR, f"shard_day{day}_part{part_idx:03d}.pt")
    torch.save(shard, shard_path)

    log.info(f"    saved {os.path.basename(shard_path)} | "
             f"orders={len(X_buf):,} links={sum(len_buf):,}")

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


def process_day_to_shards(day, topo, mean_dict, std_dict):
    """
    单天数据 → 多个 shard。
    按 ORDERS_PER_SHARD 切片保存，避免单文件过大导致 OOM。
    """
    t0 = time.time()
    log.info(f"\n  Processing day {day} ...")

    head, link, cross = load_day(day)

    unique_orders = link["order_id"].unique()
    log.info(f"    Day {day}: {len(unique_orders):,} orders, {len(link):,} link rows")

    part_idx = 0
    new_shards = []
    X_buf, y_buf, len_buf, oid_buf = [], [], [], []
    total_orders = 0
    total_links = 0

    for i in range(0, len(unique_orders), BATCH_SIZE_ORDERS):
        batch_orders = unique_orders[i:i + BATCH_SIZE_ORDERS]
        link_b  = link[link["order_id"].isin(batch_orders)]
        head_b  = head[head["order_id"].isin(batch_orders)]
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

            # 满一个 shard 就保存
            if len(X_buf) >= ORDERS_PER_SHARD:
                info = _save_one_shard(X_buf, y_buf, len_buf, oid_buf, day, part_idx)
                new_shards.append(info)
                part_idx += 1
                X_buf, y_buf, len_buf, oid_buf = [], [], [], []
                gc.collect()

        del link_b, head_b, cross_b, feat, valid, grouped
        gc.collect()

    # 保存残余
    if len(X_buf) > 0:
        info = _save_one_shard(X_buf, y_buf, len_buf, oid_buf, day, part_idx)
        new_shards.append(info)
        X_buf, y_buf, len_buf, oid_buf = [], [], [], []
        gc.collect()

    del head, link, cross
    gc.collect()

    elapsed = time.time() - t0
    log.info(
        f"  Day {day} complete: {total_orders:,} orders, {total_links:,} links, "
        f"{len(new_shards)} shard(s), {elapsed:.1f}s"
    )
    return new_shards


# ============================================================
# 旧 shard 清理（可选）
# ============================================================

def clear_day_shards(day: str):
    """
    清理某一天的旧 shard 文件。
    用于在需要重新生成某天数据时调用。
    """
    if not os.path.exists(SHARD_DIR):
        return
    removed = 0
    for fn in os.listdir(SHARD_DIR):
        if fn.startswith(f"shard_day{day}_") and fn.endswith(".pt"):
            fp = os.path.join(SHARD_DIR, fn)
            try:
                os.remove(fp)
                removed += 1
            except Exception as e:
                log.warning(f"Failed to delete {fp}: {e}")
    if removed > 0:
        log.info(f"  Cleared {removed} old shard(s) for day {day}")


# ============================================================
# 主函数
# ============================================================

def main():
    log.info("=" * 70)
    log.info("PREPARE STAGE 1 DATASET (Incremental)")
    log.info("=" * 70)
    log.info(f"  PREPARED_DIR: {os.path.abspath(PREPARED_DIR)}")
    log.info(f"  SHARD_DIR:    {os.path.abspath(SHARD_DIR)}")
    log.info(f"  DATA_DIR:     {os.path.abspath(DATA_DIR)}")
    log.info(f"  TRAIN_DAYS:   {TRAIN_DAYS}")
    log.info(f"  TEST_DAYS:    {TEST_DAYS}")

    os.makedirs(PREPARED_DIR, exist_ok=True)
    os.makedirs(SHARD_DIR, exist_ok=True)

    topo = load_topology()

    # ---- 1. 标准化参数 ----
    if os.path.exists(STATS_PATH):
        log.info(f"Stats already exist: {STATS_PATH}, skip fitting.")
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)
        mean_dict = stats["mean"]
        std_dict  = stats["std"]
        log.info(f"  Loaded {len(stats.get('feature_cols', []))} feature columns from stats")
    else:
        log.info("Stats not found, fitting from scratch ...")
        all_days = _discover_available_days()
        if not all_days:
            raise RuntimeError("Cannot fit stats: no available days found.")
        mean_dict, std_dict = fit_global_stats(topo, all_days)

        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": SCHEMA_VERSION,
                "feature_cols": STAGE1_FEATURE_COLS,
                "mean": mean_dict,
                "std": std_dict,
                "save_dtype": str(SAVE_DTYPE),
                "orders_per_shard": ORDERS_PER_SHARD,
                "created_at": datetime.now().isoformat(),
            }, f, indent=2)
        log.info(f"Stats saved: {STATS_PATH}")

    # ---- 2. 加载已有 manifest ----
    manifest = load_existing_manifest()
    existing_days = get_existing_days(manifest)

    # ---- 3. 发现可用天，增量生成 ----
    all_days = _discover_available_days()

    if not all_days:
        log.warning("No available days found. Nothing to generate.")
        save_manifest(manifest)
        return

    new_days = [d for d in all_days if d not in existing_days]

    if not new_days:
        log.info("All available days already prepared. Nothing to do.")
        log.info(f"  Prepared days: {sorted(existing_days)}")
        log.info(f"  Available days: {sorted(all_days)}")
    else:
        log.info(f"New days to prepare: {new_days}")
        log.info(f"Already prepared: {sorted(existing_days)}")

        for day in new_days:
            try:
                new_shards = process_day_to_shards(day, topo, mean_dict, std_dict)
                manifest["shards"].extend(new_shards)
            except FileNotFoundError as e:
                log.warning(f"Day {day} data not found: {e}, skipped.")
            except Exception as e:
                log.error(f"Day {day} failed: {repr(e)}, skipped.")

    # ---- 4. 保存 manifest ----
    save_manifest(manifest)

    # ---- 5. 统计汇总 ----
    total_shards = len(manifest["shards"])
    total_orders = sum(s["n_orders"] for s in manifest["shards"])
    total_links  = sum(s["n_links"]  for s in manifest["shards"])
    all_prepared_days = sorted(set(s["day"] for s in manifest["shards"]))

    log.info("\n" + "=" * 70)
    log.info("PREPARE COMPLETE")
    log.info(f"  Total shards: {total_shards}")
    log.info(f"  Total orders: {total_orders:,}")
    log.info(f"  Total links:  {total_links:,}")
    log.info(f"  Prepared days: {all_prepared_days}")
    log.info(f"  Shard dir: {os.path.abspath(SHARD_DIR)}")

    # 检查 config 里的天是否都已准备好
    missing_train = set(TRAIN_DAYS) - set(all_prepared_days)
    missing_test  = set(TEST_DAYS)  - set(all_prepared_days)
    if missing_train:
        log.warning(f"  ⚠ TRAIN_DAYS not yet prepared: {sorted(missing_train)}")
    if missing_test:
        log.warning(f"  ⚠ TEST_DAYS not yet prepared: {sorted(missing_test)}")
    if not missing_train and not missing_test:
        log.info("  ✓ All TRAIN_DAYS and TEST_DAYS are prepared. Ready to train.")

    log.info("=" * 70)


if __name__ == "__main__":
    main()