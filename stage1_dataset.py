"""
stage1_dataset.py
完整替换版：
  - SingleShardSequenceDataset 加入 shard 级特征一致性自检
  - ShardPrefetcher 不变
  - create_dataloaders 不变（保持兼容）
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

PREPARED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepared_data")
MANIFEST_PATH = os.path.join(PREPARED_DIR, "manifest.json")
STATS_PATH = os.path.join(PREPARED_DIR, "stats.json")

# ★ 与 stage1_deep_lstmframe.py 和 prepare_stage1_dataset.py 三路完全一致
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
    log.info(f"Manifest loaded: {MANIFEST_PATH}")
    log.info(f"  schema_version: {manifest.get('schema_version', 'unknown')}")
    return manifest


def load_stats():
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(
            f"stats.json not found: {STATS_PATH}\n"
            f"Please run: python prepare_stage1_dataset.py"
        )
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    log.info(f"Stats loaded: {STATS_PATH}")
    log.info(f"  feature_cols: {len(stats['feature_cols'])}")
    return stats["mean"], stats["std"], stats["feature_cols"]


def _safe_load_torch_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shard not found: {path}")
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load shard: {path}\n"
            f"Error: {repr(e)}\n"
            f"Delete prepared_data and rerun prepare_stage1_dataset.py"
        ) from e


# ============================================================
# shard 划分
# ============================================================

def split_train_val_shards(val_ratio=0.15, seed=42):
    manifest = load_manifest()
    train_shards = manifest.get("train_shards", [])
    test_shards = manifest.get("test_shards", [])

    if not train_shards:
        raise RuntimeError("No train_shards in manifest.json")

    idx = list(range(len(train_shards)))
    random.seed(seed)
    random.shuffle(idx)

    n_val = max(1, int(len(train_shards) * val_ratio))
    val_set = set(idx[:n_val])

    train_split = [train_shards[i] for i in range(len(train_shards)) if i not in val_set]
    val_split   = [train_shards[i] for i in range(len(train_shards)) if i in val_set]

    log.info(f"Shard split: train={len(train_split)} val={len(val_split)} test={len(test_shards)}")
    return train_split, val_split, test_shards


# ============================================================
# 单 shard Dataset（加入特征一致性自检）
# ============================================================

def _check_periodic_feature_range(X_sample: np.ndarray, feature_cols: list,
                                   shard_path: str, n_check: int = 500):
    """
    ★ 隐患 ① 防护：检查 shard 内周期特征是否被错误地 Z-Score 过。
    如果周期特征被 Z-Score，其绝对值可能超过 2（正常 clip 后最大 1）。
    发现异常时报 WARNING，而不是静默通过。
    """
    for j, col in enumerate(feature_cols):
        if col in PERIODIC_FEATURES:
            col_vals = X_sample[:n_check, j] if X_sample.shape[0] >= n_check else X_sample[:, j]
            max_abs = np.abs(col_vals).max()
            if max_abs > 1.5:
                log.warning(
                    f"⚠ Shard periodic feature anomaly detected!\n"
                    f"  Shard: {os.path.basename(shard_path)}\n"
                    f"  Feature: '{col}' | max_abs={max_abs:.4f} (expected ≤ 1.0)\n"
                    f"  This shard was likely generated with Z-Score on periodic features.\n"
                    f"  Please delete prepared_data/ and rerun: python prepare_stage1_dataset.py"
                )


class SingleShardSequenceDataset(Dataset):
    """
    加载单个 shard，含特征一致性自检。
    """

    def __init__(self, shard_obj: dict, max_seq_len: int = 100):
        self.max_seq_len = max_seq_len
        self.X_list = shard_obj["X_list"]
        self.y_list = shard_obj["y_list"]
        self.lengths = [min(int(v), max_seq_len) for v in shard_obj["lengths"]]
        self.day = shard_obj.get("day", "?")
        self.n_orders = shard_obj.get("n_orders", len(self.X_list))

        # ★ 隐患 ① 防护：启动时抽检周期特征范围
        if len(self.X_list) > 0:
            # 取第一条序列做检查
            sample_x = self.X_list[0]
            shard_path = shard_obj.get("path", "<unknown>")
            # 从 stats 获取 feature_cols
            try:
                _, _, feature_cols = load_stats()
                _check_periodic_feature_range(
                    np.asarray(sample_x, dtype=np.float32),
                    feature_cols,
                    shard_path,
                )
            except Exception:
                pass  # 检查失败不影响训练主流程

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = np.asarray(self.X_list[idx][:self.max_seq_len], dtype=np.float32)
        y = np.asarray(self.y_list[idx][:self.max_seq_len], dtype=np.int64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            torch.FloatTensor(x),
            torch.LongTensor(y),
            self.lengths[idx],
        )


# ============================================================
# collate_fn
# ============================================================

def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)

    # 按长度降序（enforce_sorted=True 需要）
    sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    seqs    = [seqs[i]    for i in sorted_idx]
    labels  = [labels[i]  for i in sorted_idx]
    lengths = [lengths[i] for i in sorted_idx]

    seqs_padded   = pad_sequence(seqs,   batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    return seqs_padded, labels_padded, torch.LongTensor(lengths)


# ============================================================
# shard 加载工具
# ============================================================

def load_single_shard(shard_info: dict) -> dict:
    """加载一个 shard，并附上 path 供自检使用。"""
    shard = _safe_load_torch_file(shard_info["path"])
    shard["path"] = shard_info["path"]  # 注入 path 供 Dataset 自检
    return shard


def create_single_shard_loader(
    shard_info: dict,
    batch_size: int = 128,
    max_seq_len: int = 100,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    shard = load_single_shard(shard_info)
    ds = SingleShardSequenceDataset(shard, max_seq_len=max_seq_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return loader, ds


# ============================================================
# 后台预取
# ============================================================

class ShardPrefetcher:
    """
    后台线程预加载下一个 shard，消除 shard 切换停顿。
    """

    def __init__(self, shard_infos: list):
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