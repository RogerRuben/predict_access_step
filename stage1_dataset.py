"""
stage1_dataset.py
-----------------
支持：
1. manifest 驱动 shard 读取
2. 单 shard DataLoader
3. 后台预取下一个 shard
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
        raise FileNotFoundError(f"Shard file not found: {path}")
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load shard: {path}\n"
            f"Original error: {repr(e)}\n"
            f"Please delete prepared_data and rerun prepare_stage1_dataset.py"
        ) from e


# ============================================================
# shard 划分
# ============================================================

def split_train_val_shards(val_ratio=0.15, seed=42):
    manifest = load_manifest()
    train_shards = manifest.get("train_shards", [])
    test_shards = manifest.get("test_shards", [])

    if not train_shards:
        raise RuntimeError("No train_shards found in manifest.json")

    shard_indices = list(range(len(train_shards)))
    random.seed(seed)
    random.shuffle(shard_indices)

    n_val = max(1, int(len(train_shards) * val_ratio))
    val_idx = set(shard_indices[:n_val])

    train_split = [train_shards[i] for i in range(len(train_shards)) if i not in val_idx]
    val_split   = [train_shards[i] for i in range(len(train_shards)) if i in val_idx]

    log.info(f"Train/Val shard split: train={len(train_split)} val={len(val_split)} test={len(test_shards)}")
    return train_split, val_split, test_shards


# ============================================================
# 单 shard Dataset
# ============================================================

class SingleShardSequenceDataset(Dataset):
    """
    只持有一个 shard。
    """
    def __init__(self, shard_obj: dict, max_seq_len: int = 100):
        self.max_seq_len = max_seq_len
        self.X_list = shard_obj["X_list"]
        self.y_list = shard_obj["y_list"]
        self.lengths = [min(int(x), max_seq_len) for x in shard_obj["lengths"]]
        self.day = shard_obj.get("day", "?")
        self.n_orders = shard_obj.get("n_orders", len(self.X_list))
        self.n_links = shard_obj.get("n_links", sum(self.lengths))

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = self.X_list[idx][:self.max_seq_len]
        y = self.y_list[idx][:self.max_seq_len]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.FloatTensor(x),
            torch.LongTensor(y),
            self.lengths[idx],
        )


def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)

    sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    seqs = [seqs[i] for i in sorted_idx]
    labels = [labels[i] for i in sorted_idx]
    lengths = [lengths[i] for i in sorted_idx]

    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    return seqs_padded, labels_padded, torch.LongTensor(lengths)


def load_single_shard(shard_info: dict):
    shard = _safe_load_torch_file(shard_info["path"])
    return shard


def create_single_shard_loader(
    shard_info: dict,
    batch_size: int = 128,
    max_seq_len: int = 100,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    直接从 shard_info 创建 DataLoader
    """
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
# 后台预取器
# ============================================================

class ShardPrefetcher:
    """
    后台线程预取下一个 shard。
    用法：
        pf = ShardPrefetcher(shard_infos)
        cur = pf.next()
        while cur is not None:
            # train on cur
            cur = pf.next()
    """
    def __init__(self, shard_infos):
        self.shard_infos = shard_infos
        self.idx = 0
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None

        if len(self.shard_infos) > 0:
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