"""
stage1_dataset.py — 动态筛选版
改动:
  1. manifest 只记录"所有可用 shard"
  2. split_train_val_shards 根据 config.TRAIN_DAYS / TEST_DAYS 动态筛选
  3. shard 目录改为 prepared_data/shards/
"""

import os
import json
import random
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from concurrent.futures import ThreadPoolExecutor
from logger import get_logger

log = get_logger()

PREPARED_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepared_data")
MANIFEST_PATH = os.path.join(PREPARED_DIR, "manifest.json")
STATS_PATH    = os.path.join(PREPARED_DIR, "stats.json")

PERIODIC_FEATURES = frozenset({
    "sin_slice", "cos_slice",
    "sin_arr_slice", "cos_arr_slice",
})


# ============================================================
# 基础读取
# ============================================================

def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"manifest.json not found: {MANIFEST_PATH}\n"
            f"Please run: python prepare_stage1_dataset.py"
        )
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    log.info(f"Manifest loaded: {len(manifest.get('shards', []))} total shard(s)")
    return manifest


# 在文件顶部加一个模块级标记
_stats_loaded_logged = False

def load_stats():
    global _stats_loaded_logged
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"stats.json not found: {STATS_PATH}\n"
            f"Please run: python prepare_stage1_dataset.py"
        )
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # ★ 只在第一次打印
    if not _stats_loaded_logged:
        log.info(f"Stats loaded: {len(stats['feature_cols'])} features")
        _stats_loaded_logged = True

    return stats["mean"], stats["std"], stats["feature_cols"]


def _safe_load_torch_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shard not found: {path}")
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load shard: {path}\n{repr(e)}") from e


# ============================================================
# ★ 动态筛选 train / val / test
# ============================================================

def split_train_val_shards(val_ratio=0.15, seed=42):
    """
    根据 config.py 的 TRAIN_DAYS / TEST_DAYS 动态筛选 shard。
    不再依赖 manifest 里写死的 train_shards / test_shards。
    """
    from config import TRAIN_DAYS, TEST_DAYS

    manifest = load_manifest()
    all_shards = manifest.get("shards", [])

    if not all_shards:
        raise RuntimeError("No shards found in manifest.json")

    train_days_set = set(TRAIN_DAYS)
    test_days_set  = set(TEST_DAYS)

    # 按天筛选
    train_shards = [s for s in all_shards if s["day"] in train_days_set]
    test_shards  = [s for s in all_shards if s["day"] in test_days_set]

    # 检查是否有缺失
    available_days = set(s["day"] for s in all_shards)
    missing_train = train_days_set - available_days
    missing_test  = test_days_set - available_days

    if missing_train:
        log.warning(
            f"TRAIN_DAYS {sorted(missing_train)} not found in prepared shards. "
            f"Run: python prepare_stage1_dataset.py"
        )
    if missing_test:
        log.warning(
            f"TEST_DAYS {sorted(missing_test)} not found in prepared shards. "
            f"Run: python prepare_stage1_dataset.py"
        )

    if not train_shards:
        raise RuntimeError(
            f"No train shards found for TRAIN_DAYS={TRAIN_DAYS}. "
            f"Available days: {sorted(available_days)}"
        )

    # train 内部再拆 train / val（shard 级别）
    shard_indices = list(range(len(train_shards)))
    random.seed(seed)
    random.shuffle(shard_indices)

    n_val = max(1, int(len(train_shards) * val_ratio))
    val_idx = set(shard_indices[:n_val])

    train_split = [train_shards[i] for i in range(len(train_shards)) if i not in val_idx]
    val_split   = [train_shards[i] for i in range(len(train_shards)) if i in val_idx]

    log.info(f"Dynamic shard split:")
    log.info(f"  TRAIN_DAYS={sorted(train_days_set)} → {len(train_split)} train + {len(val_split)} val shard(s)")
    log.info(f"  TEST_DAYS={sorted(test_days_set)} → {len(test_shards)} test shard(s)")
    log.info(f"  Available days in manifest: {sorted(available_days)}")

    return train_split, val_split, test_shards


# ============================================================
# Dataset / collate_fn / prefetcher（不变）
# ============================================================

def _check_periodic_feature_range(X_sample, feature_cols, shard_path, n_check=500):
    for j, col in enumerate(feature_cols):
        if col in PERIODIC_FEATURES:
            vals = X_sample[:min(n_check, X_sample.shape[0]), j]
            if np.abs(vals).max() > 1.5:
                log.warning(
                    f"⚠ Periodic feature anomaly in {os.path.basename(shard_path)}: "
                    f"'{col}' max_abs={np.abs(vals).max():.4f}. Regenerate shards."
                )


class SingleShardSequenceDataset(Dataset):
    """
    加载单个 shard，含：
    1. 周期特征范围检查
    2. 数据驱动的 sequence-level sample_weights
    """

    def __init__(self, shard_obj: dict, max_seq_len: int = 100,
                 class_counts: np.ndarray = None):
        """
        Parameters
        ----------
        shard_obj    : torch.load 出来的 shard dict
        max_seq_len  : 截断长度
        class_counts : 全局类别频率 np.array([n_s1, n_s2, n_s3, n_s4])
                       用于数据驱动权重；若为 None，则用 shard 内频率估计
        """
        self.max_seq_len = max_seq_len
        self.X_list  = shard_obj["X_list"]
        self.y_list  = shard_obj["y_list"]
        self.lengths = [min(int(v), max_seq_len) for v in shard_obj["lengths"]]
        self.day     = shard_obj.get("day", "?")
        self.n_orders = shard_obj.get("n_orders", len(self.X_list))

        # 周期特征检查
        if len(self.X_list) > 0:
            try:
                _, _, fc = load_stats()
                _check_periodic_feature_range(
                    np.asarray(self.X_list[0], dtype=np.float32),
                    fc, shard_obj.get("path", "<unknown>"),
                )
            except Exception:
                pass

        # ★ 数据驱动的 sequence-level sample_weights
        self.sample_weights = self._compute_sample_weights(class_counts)

    def _compute_sample_weights(self, global_counts=None):
        """
        计算每个序列的采样权重。

        思路:
          1. 统计当前 shard 内每个序列的最高风险类别
          2. 用 inverse-frequency 方式赋权
          3. 含 s2/s3 边界 (同时含 s2 和 s3) 的序列额外加权

        Parameters
        ----------
        global_counts : np.array([n_s1,n_s2,n_s3,n_s4]) 来自 estimate_weights()
                        如果为 None，则用当前 shard 内频率估计
        """
        if global_counts is None:
            local_counts = np.zeros(4, dtype=np.int64)
            for y in self.y_list:
                arr = np.asarray(y)
                arr = arr[(arr >= 0) & (arr < 4)]
                if len(arr) > 0:
                    local_counts += np.bincount(arr, minlength=4)
            counts = np.maximum(local_counts, 1)
        else:
            counts = np.maximum(global_counts, 1)

        total = counts.sum()

        # inverse frequency，取 sqrt 平滑，避免极端权重
        inv_freq = np.sqrt(total / (4 * counts.astype(float)))
        inv_freq = inv_freq / inv_freq[0]           # 以 s1 为基准做归一化

        weights = []
        for y in self.y_list:
            arr = np.asarray(y)
            arr = arr[(arr >= 0) & (arr < 4)]

            if len(arr) == 0:
                weights.append(1.0)
                continue

            max_cls = int(arr.max())               # 序列最高风险类别
            w = inv_freq[max_cls]

            # ★ 额外奖励：同时含 s2 和 s3 的边界序列
            # 这是 boundary head 最需要学的
            unique_cls = set(arr.tolist())
            if 1 in unique_cls and 2 in unique_cls:
                w *= 1.5

            # clip，避免过于极端的权重
            w = float(np.clip(w, 1.0, 8.0))
            weights.append(w)

        return weights

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = np.asarray(self.X_list[idx][:self.max_seq_len], dtype=np.float32)
        y = np.asarray(self.y_list[idx][:self.max_seq_len], dtype=np.int64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.FloatTensor(x), torch.LongTensor(y), self.lengths[idx]


def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)
    so = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    seqs    = [seqs[i]    for i in so]
    labels  = [labels[i]  for i in so]
    lengths = [lengths[i] for i in so]
    return (
        pad_sequence(seqs, batch_first=True, padding_value=0.0),
        pad_sequence(labels, batch_first=True, padding_value=-1),
        torch.LongTensor(lengths),
    )


def load_single_shard(shard_info):
    shard = _safe_load_torch_file(shard_info["path"])
    shard["path"] = shard_info["path"]
    return shard


class ShardPrefetcher:
    def __init__(self, shard_infos):
        self.shard_infos = shard_infos
        self.idx = 0
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        if self.shard_infos:
            self.future = self.pool.submit(load_single_shard, self.shard_infos[0])

    def next(self):
        if self.future is None:
            return None, None
        shard_obj = self.future.result()
        shard_info = self.shard_infos[self.idx]
        self.idx += 1
        if self.idx < len(self.shard_infos):
            self.future = self.pool.submit(load_single_shard, self.shard_infos[self.idx])
        else:
            self.future = None
        return shard_info, shard_obj

    def close(self):
        self.pool.shutdown(wait=True)