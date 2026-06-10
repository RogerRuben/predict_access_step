"""
stage1_dataset.py
-----------------
从预处理好的 shard 文件加载数据的 Dataset / DataLoader。
训练时不再碰原始 csv，直接读 .pt shard。
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from logger import get_logger

log = get_logger()

PREPARED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepared_data")


# ================================================================
# 加载 stats
# ================================================================

def load_stats():
    """加载离线计算的标准化参数。"""
    stats_path = os.path.join(PREPARED_DIR, "stats.json")
    with open(stats_path, "r") as f:
        stats = json.load(f)
    log.info(f"Stats loaded: {stats_path}")
    log.info(f"  feature_cols: {len(stats['feature_cols'])} features")
    return stats["mean"], stats["std"], stats["feature_cols"]


# ================================================================
# Shard Dataset
# ================================================================

class ShardSequenceDataset(Dataset):
    """
    从多个 shard 文件加载序列数据。

    两种模式:
    - preload=True:  一次性加载所有 shard 到内存（快，但吃内存）
    - preload=False: 每次 __getitem__ 时从对应 shard 读取（慢，但省内存）

    推荐: 内存够就 preload=True，不够就 preload=False
    """

    def __init__(
        self,
        shard_dir: str,
        max_seq_len: int = 100,
        preload: bool = True,
    ):
        self.max_seq_len = max_seq_len
        self.preload = preload

        # 扫描 shard 文件
        shard_files = sorted([
            os.path.join(shard_dir, f)
            for f in os.listdir(shard_dir)
            if f.endswith(".pt")
        ])

        if not shard_files:
            raise FileNotFoundError(f"No shard files in {shard_dir}")

        log.info(f"Found {len(shard_files)} shard(s) in {shard_dir}")

        # 加载所有 shard
        self.sequences = []
        self.labels    = []
        self.lengths   = []

        for sf in shard_files:
            shard = torch.load(sf, map_location="cpu")
            n = shard["n_orders"]

            for i in range(n):
                x = shard["X_list"][i]
                y = shard["y_list"][i]
                seq_len = min(len(x), max_seq_len)

                if self.preload:
                    self.sequences.append(x[:seq_len])
                    self.labels.append(y[:seq_len])
                else:
                    self.sequences.append((sf, i))
                    self.labels.append(None)

                self.lengths.append(seq_len)

            day = shard.get("day", "?")
            log.info(f"  Shard {os.path.basename(sf)}: day={day}, {n:,} orders")

            del shard
            import gc; gc.collect()

        total_links = sum(self.lengths)
        log.info(
            f"Dataset ready: {len(self.sequences):,} sequences, "
            f"{total_links:,} total links, "
            f"avg_len={np.mean(self.lengths):.1f}, "
            f"max_len={max(self.lengths)}"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.preload:
            x = self.sequences[idx]
            y = self.labels[idx]
        else:
            sf, i = self.sequences[idx]
            shard = torch.load(sf, map_location="cpu")
            x = shard["X_list"][i][:self.max_seq_len]
            y = shard["y_list"][i][:self.max_seq_len]
            del shard

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.FloatTensor(x),
            torch.LongTensor(y),
            self.lengths[idx],
        )


def collate_fn(batch):
    """变长序列 padding + 排序。"""
    seqs, labels, lengths = zip(*batch)

    sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    seqs    = [seqs[i]    for i in sorted_idx]
    labels  = [labels[i]  for i in sorted_idx]
    lengths = [lengths[i] for i in sorted_idx]

    seqs_padded   = pad_sequence(seqs,   batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    return seqs_padded, labels_padded, torch.LongTensor(lengths)


# ================================================================
# 构建 DataLoader 的便捷函数
# ================================================================

def create_dataloaders(
    max_seq_len: int = 100,
    batch_size: int = 128,
    val_ratio: float = 0.15,
    preload: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    从 prepared_data 构建 train / val / test DataLoader。

    训练集按 val_ratio 随机划分出验证集。
    """
    train_dir = os.path.join(PREPARED_DIR, "train")
    test_dir  = os.path.join(PREPARED_DIR, "test")

    # 加载完整训练集
    log.info("Building train dataset ...")
    full_ds = ShardSequenceDataset(train_dir, max_seq_len=max_seq_len, preload=preload)

    # 拆分 train / val
    n_total = len(full_ds)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val

    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    train_ds = torch.utils.data.Subset(full_ds, train_indices)
    val_ds   = torch.utils.data.Subset(full_ds, val_indices)

    log.info(f"  Train: {n_train:,} | Val: {n_val:,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 测试集
    test_loader = None
    if os.path.exists(test_dir) and os.listdir(test_dir):
        log.info("Building test dataset ...")
        test_ds = ShardSequenceDataset(test_dir, max_seq_len=max_seq_len, preload=preload)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers,
        )
    else:
        log.info("No test shards found, skipping.")

    return train_loader, val_loader, test_loader