"""
stage1_deep.py — WDR 创新架构完整版
适配 pipeline.py 的完整调用链路

创新点：
1. 时空衰减闸门残差传递结构 (Spatiotemporal Decay Gate)
2. 状态-不确定性风险多任务自对齐损失 (Multi-task Alignment Loss)
"""

import os
import gc
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from config import (
    MODEL_DIR, NUM_CLASSES, STATUS_CLASSES,
    WRC_HIDDEN_DIM, WRC_NUM_LAYERS, WRC_BATCH_SIZE,
    WRC_EPOCHS, WRC_LR, WRC_MAX_SEQ_LEN
)
from feature_eng import STAGE1_FEATURE_COLS
from stage1_dataset import (

    split_train_val_shards,
    load_stats,
    SingleShardSequenceDataset,
    collate_fn,
    ShardPrefetcher,
)
from logger import get_logger


log = get_logger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 时空特征在 STAGE1_FEATURE_COLS 中的索引（预计算，避免重复查找）
ST_FEATURE_NAMES = ["pos_ratio", "cum_travel_time", "sin_arr_slice", "cos_arr_slice"]
ST_INDICES = [STAGE1_FEATURE_COLS.index(n) for n in ST_FEATURE_NAMES]
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================================================================
# 1. 创新组件：动态时空衰减闸门
# ================================================================

class SpatiotemporalDecayGate(nn.Module):
    """
    创新点 1：动态时空衰减闸门。
    从序列特征中剥离时空维度，自适应计算历史记忆的过时衰减权重。

    物理含义：
    - 路径越靠后（pos_ratio 大）、累计行驶时间越长，出发时的路况信息越过时
    - 该门控机制动态决定"保留多少历史记忆 / 引入多少当前观测"
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(len(ST_FEATURE_NAMES), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        rnn_step: torch.Tensor,   # (B, H) 当前步 LSTM 输出
        st_step: torch.Tensor,    # (B, 4) 当前步时空特征
        h_prev: torch.Tensor,     # (B, H) 上一步调制后隐状态
    ) -> torch.Tensor:
        decay = self.gate(st_step)                              # (B, H) 0~1
        h_mod = (1.0 - decay) * h_prev + decay * rnn_step      # 残差调制
        return h_mod


# ================================================================
# 2. 核心 WDR 网络
# ================================================================

class CustomWDRNet(nn.Module):
    """
    Wide-Deep-Recurrent + 时空衰减闸门 + 多任务输出头

    输出：
      traffic_logits: (B, T, NUM_CLASSES)   —— 路况分类
      risk_preds:     (B, T, 1)              —— 拥堵风险回归
    """

    def __init__(self, dense_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dense_dim = dense_dim

        # Wide：直接线性映射，提供即时强特征
        self.wide = nn.Linear(dense_dim, hidden_dim)

        # Deep：MLP 挖掘非线性泛化特征
        self.deep = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Recurrent：LSTM 序列骨干
        self.lstm = nn.LSTM(
            input_size=dense_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # 时空衰减闸门
        self.decay_gate = SpatiotemporalDecayGate(hidden_dim)

        # 多任务输出头
        fusion_dim = hidden_dim * 3     # wide + deep + recurrent
        self.traffic_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),               # 输出 [0,1] 对应拥堵风险
        )

    def forward(
        self,
        x: torch.Tensor,           # (B, T, dense_dim)
        lengths: torch.Tensor,     # (B,)
        st_info: torch.Tensor,     # (B, T, 4)
    ):
        B, T, D = x.shape

        # LSTM（使用 pack_padded_sequence 处理变长序列）
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )                                                       # (B, T, H)

        # 时空衰减闸门逐步调制
        h_prev = torch.zeros(B, self.hidden_dim, device=x.device)
        steps = []
        for t in range(T):
            h_prev = self.decay_gate(lstm_out[:, t, :], st_info[:, t, :], h_prev)
            steps.append(h_prev.unsqueeze(1))
        decay_out = torch.cat(steps, dim=1)                     # (B, T, H)

        # 平铺 → Wide & Deep
        x_flat = x.reshape(B * T, D)
        wide_out = self.wide(x_flat).reshape(B, T, self.hidden_dim)
        deep_out = self.deep(x_flat).reshape(B, T, self.hidden_dim)

        # 融合
        fused = torch.cat(
            [wide_out, deep_out, lstm_out + decay_out], dim=-1
        )                                                       # (B, T, H*3)

        traffic_logits = self.traffic_head(fused)               # (B, T, C)
        risk_preds = self.risk_head(fused)                      # (B, T, 1)

        return traffic_logits, risk_preds


# ================================================================
# 3. 损失函数
# ================================================================

class MultiTaskAlignmentLoss(nn.Module):
    """
    创新点 2：多任务自对齐损失

    Loss = CE(traffic) + λ * MSE(risk_pred, stop_probability_proxy)

    stop_probability_proxy（无梯度）：
      = P(s3) + P(s4) from traffic_logits
      —— 强迫 risk_head 的表征空间对拥堵类别高度敏感
    """

    def __init__(self, risk_weight: float = 0.3, ignore_index: int = -1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")
        self.risk_weight = risk_weight
        self.ignore_index = ignore_index

    def forward(
        self,
        traffic_logits: torch.Tensor,   # (B, T, C)
        risk_preds: torch.Tensor,        # (B, T, 1)
        targets: torch.Tensor,           # (B, T)
    ):
        B, T, C = traffic_logits.shape

        # 交叉熵损失
        loss_ce = self.ce(traffic_logits.reshape(B * T, C), targets.reshape(B * T))

        # 自对齐 risk target（无梯度）
        with torch.no_grad():
            probs = F.softmax(traffic_logits, dim=-1)
            risk_target = (probs[:, :, 2] + probs[:, :, 3]).unsqueeze(-1)  # (B, T, 1)

        # 只在非 padding 位置计算
        mask = (targets != self.ignore_index).float().unsqueeze(-1)
        loss_risk = self.mse(risk_preds * mask, risk_target * mask)

        return loss_ce + self.risk_weight * loss_risk, loss_ce, loss_risk


# ================================================================
# 4. 评估
# ================================================================

def evaluate_wrc(model: CustomWDRNet, loader, device, verbose=False, use_tqdm=False):
    """
    验证集评估。只用 traffic_logits 计算分类指标。
    """
    model.eval()
    criterion = MultiTaskAlignmentLoss()

    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    iterator = tqdm(loader, desc="[valid]", leave=False, dynamic_ncols=True) if use_tqdm else loader

    with torch.no_grad():
        for X_b, y_b, lens_b in iterator:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            lens_b = lens_b.to(device)

            st_b = X_b[:, :, ST_INDICES]

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, risk = model(X_b, lens_b, st_b)
                loss, _, _ = criterion(logits, risk, y_b)

            total_loss += float(loss.item())
            n_batches += 1

            preds = logits.argmax(dim=-1)
            for b in range(len(lens_b)):
                L = lens_b[b].item()
                all_preds.extend(preds[b, :L].cpu().numpy())
                all_labels.extend(y_b[b, :L].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    mask = y_true >= 0
    y_true, y_pred = y_true[mask], y_pred[mask]

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    high_actual = y_true >= 2
    dmr = float((y_pred < 2)[high_actual].sum() / max(high_actual.sum(), 1))

    s4_actual = y_true == 3
    s4_under = float((y_pred < 3)[s4_actual].sum() / max(s4_actual.sum(), 1))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    recalls = [cm[i, i] / max(cm[i].sum(), 1) for i in range(4)]

    if verbose:
        labels_str = [f"s{k}" for k in STATUS_CLASSES]
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels_str],
            columns=[f"pred_{l}" for l in labels_str],
        )
        log.info(f"Confusion Matrix:\n{cm_df.to_string()}")
        log.info(f"\n{classification_report(y_true, y_pred, target_names=labels_str, digits=4)}")
        for i, l in enumerate(labels_str):
            log.info(f"  Recall {l}: {recalls[i]:.4f}")
        log.info(f"  Dangerous Miss Rate (≥3→≤2): {dmr:.4f}")
        log.info(f"  s4 Underestimate   (4→≤3):   {s4_under:.4f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "dangerous_miss": dmr,
        "s4_underestimate": s4_under,
        "recall_per_class": recalls,
    }


# ============================================================
# GPU 日志
# ============================================================

def log_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        p = torch.cuda.max_memory_allocated() / 1024**3
        log.info(f"{prefix}[GPU] alloc={a:.2f}GB reserved={r:.2f}GB peak={p:.2f}GB")


# ============================================================
# Checkpoint
# ============================================================

def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_dmr,
    best_macro_f1,
    ckpt_path,
):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_dmr": best_dmr,
        "best_macro_f1": best_macro_f1,
        "dense_dim": model.dense_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
    }, ckpt_path)
    log.info(f"Checkpoint saved: {ckpt_path}")


def load_training_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    log.info(f"Checkpoint loaded: {ckpt_path}")
    return ckpt


# ============================================================
# 验证：逐 shard 聚合
# ============================================================

def evaluate_wrc_on_shards(model, shard_infos, device, batch_size=256, max_seq_len=100, verbose=False):
    model.eval()
    criterion = MultiTaskAlignmentLoss(risk_weight=0.3, ignore_index=-1)

    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    pf = ShardPrefetcher(shard_infos)

    while True:
        shard_info, shard_obj = pf.next()
        if shard_obj is None:
            break

        ds = SingleShardSequenceDataset(shard_obj, max_seq_len=max_seq_len)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,                     # Windows 下这里建议 0，避免 worker 启动开销
            pin_memory=(device.type == "cuda"),
        )

        with torch.no_grad():
            for X_b, y_b, lens_b in loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                lens_b = lens_b.to(device)

                st_b = X_b[:, :, ST_INDICES]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits, risk = model(X_b, lens_b, st_b)
                    loss, _, _ = criterion(logits, risk, y_b)

                total_loss += float(loss.item())
                n_batches += 1

                preds = logits.argmax(dim=-1)
                for b in range(len(lens_b)):
                    L = lens_b[b].item()
                    all_preds.extend(preds[b, :L].cpu().numpy())
                    all_labels.extend(y_b[b, :L].cpu().numpy())

        del ds, loader, shard_obj
        gc.collect()

    pf.close()

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    mask = y_true >= 0
    y_true, y_pred = y_true[mask], y_pred[mask]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    high_actual = y_true >= 2
    dmr = float((y_pred < 2)[high_actual].sum() / max(high_actual.sum(), 1))

    s4_actual = y_true == 3
    s4_under = float((y_pred < 3)[s4_actual].sum() / max(s4_actual.sum(), 1))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    recalls = [cm[i, i] / max(cm[i].sum(), 1) for i in range(4)]

    if verbose:
        labels_str = [f"s{k}" for k in STATUS_CLASSES]
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels_str],
            columns=[f"pred_{l}" for l in labels_str],
        )
        log.info(f"Confusion Matrix:\n{cm_df.to_string()}")
        log.info(f"\n{classification_report(y_true, y_pred, target_names=labels_str, digits=4)}")
        for i, l in enumerate(labels_str):
            log.info(f"  Recall {l}: {recalls[i]:.4f}")
        log.info(f"  Dangerous Miss Rate (≥3→≤2): {dmr:.4f}")
        log.info(f"  s4 Underestimate   (4→≤3):   {s4_under:.4f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "dangerous_miss": dmr,
        "s4_underestimate": s4_under,
        "recall_per_class": recalls,
    }


# ============================================================
# 训练：逐 shard 流式 + checkpoint + OneCycleLR
# ============================================================

def train_wrc_from_shards(
    hidden_dim=WRC_HIDDEN_DIM,
    num_layers=WRC_NUM_LAYERS,
    batch_size=WRC_BATCH_SIZE,
    epochs=WRC_EPOCHS,
    lr=WRC_LR,
    max_seq_len=WRC_MAX_SEQ_LEN,
    device=None,
    resume=True,
):
    if device is None:
        device = DEVICE

    log.info("=" * 70)
    log.info("WDR Streaming Training from Shards")
    log.info("=" * 70)
    log.info(
        f"device={device} | hidden={hidden_dim} | layers={num_layers} | "
        f"batch={batch_size} | lr={lr} | max_seq_len={max_seq_len} | epochs={epochs}"
    )

    # metadata
    mean_dict, std_dict, feature_cols = load_stats()
    train_shards, val_shards, _ = split_train_val_shards(val_ratio=0.15, seed=42)

    log.info(f"Train shards: {len(train_shards)} | Val shards: {len(val_shards)}")

    # 估算总 step（供 OneCycleLR）
    total_train_orders = sum(s["n_orders"] for s in train_shards)
    steps_per_epoch = math.ceil(total_train_orders / batch_size)
    total_steps = epochs * steps_per_epoch
    log.info(f"Approx. steps_per_epoch={steps_per_epoch} | total_steps={total_steps}")

    # 模型
    model = CustomWDRNet(
        dense_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    log_gpu_memory("Init ")

    criterion = MultiTaskAlignmentLoss(risk_weight=0.3, ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ★ 动态学习率：OneCycleLR（按 batch 调整）
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,        # 初始 lr = max_lr / 10
        final_div_factor=100.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"wdr_stream_{ts}")
    writer = SummaryWriter(log_dir=tb_dir)
    log.info(f"TensorBoard: tensorboard --logdir {tb_dir}")
    log.info("             Then open http://localhost:6006")

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "wdr_latest.pt")
    best_ckpt   = os.path.join(CHECKPOINT_DIR, "wdr_best.pt")

    start_epoch = 0
    best_dmr = 1.0
    best_macro_f1 = 0.0
    best_state = None
    global_step = 0
    early_stop_patience = 2
    no_improve_epochs = 0

    # resume
    if resume and os.path.exists(latest_ckpt):
        ckpt = load_training_checkpoint(latest_ckpt, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt["scheduler_state"] is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if ckpt["scaler_state"] is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_dmr = ckpt["best_dmr"]
        best_macro_f1 = ckpt["best_macro_f1"]
        log.info(f"Resuming from epoch {start_epoch} | best_dmr={best_dmr:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_ce = 0.0
        train_risk = 0.0
        n_batches = 0

        epoch_shards = train_shards.copy()
        np.random.RandomState(epoch + 42).shuffle(epoch_shards)

        log.info(f"\nEpoch {epoch+1}/{epochs} | {len(epoch_shards)} shard(s)")

        pf = ShardPrefetcher(epoch_shards)
        shard_counter = 0

        while True:
            shard_info, shard_obj = pf.next()
            if shard_obj is None:
                break

            shard_counter += 1
            log.info(f"  Train shard {shard_counter}/{len(epoch_shards)}: {os.path.basename(shard_info['path'])}")

            ds = SingleShardSequenceDataset(shard_obj, max_seq_len=max_seq_len)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,                     # Windows 建议 0，减少 worker 初始化停顿
                pin_memory=(device.type == "cuda"),
            )

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch+1:>2} Shard {shard_counter:>3}",
                leave=False,
                dynamic_ncols=True,
                unit="batch",
            )

            for batch_idx, (X_b, y_b, lens_b) in enumerate(pbar):
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                lens_b = lens_b.to(device)

                st_b = X_b[:, :, ST_INDICES]

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits, risk = model(X_b, lens_b, st_b)
                    loss, loss_ce, loss_risk = criterion(logits, risk, y_b)

                if torch.isnan(loss) or torch.isinf(loss):
                    log.warning(f"NaN/Inf loss at epoch={epoch+1}, shard={shard_counter}, batch={batch_idx}, skipping")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()     # ★ 每 batch 动态调整 lr

                global_step += 1
                train_loss += float(loss.item())
                train_ce += float(loss_ce.item())
                train_risk += float(loss_risk.item())
                n_batches += 1

                if device.type == "cuda" and batch_idx % 50 == 0:
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix({
                        "loss": f"{train_loss/max(n_batches,1):.4f}",
                        "gpu": f"{alloc:.1f}G",
                        "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    })

            del ds, loader, shard_obj

            # ★ 不要每个 shard 都强制 gc / empty_cache，隔 10 个再做一次
            if shard_counter % 10 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        pf.close()

        avg_loss = train_loss / max(n_batches, 1)
        avg_ce   = train_ce / max(n_batches, 1)
        avg_risk = train_risk / max(n_batches, 1)

        val_m = evaluate_wrc_on_shards(
            model,
            val_shards,
            device=device,
            batch_size=batch_size * 2,
            max_seq_len=max_seq_len,
            verbose=False,
        )

        cur_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train_total", avg_loss, epoch)
        writer.add_scalar("Loss/train_ce", avg_ce, epoch)
        writer.add_scalar("Loss/train_risk", avg_risk, epoch)
        writer.add_scalar("Loss/valid", val_m["loss"], epoch)
        writer.add_scalar("Metrics/accuracy", val_m["accuracy"], epoch)
        writer.add_scalar("Metrics/macro_f1", val_m["macro_f1"], epoch)
        writer.add_scalar("Metrics/DMR", val_m["dangerous_miss"], epoch)
        writer.add_scalar("Metrics/s4_under", val_m["s4_underestimate"], epoch)
        writer.add_scalar("LR", cur_lr, epoch)
        for i, r in enumerate(val_m["recall_per_class"]):
            writer.add_scalar(f"Recall/s{i+1}", r, epoch)

        log.info(
            f"Epoch {epoch+1:>2}/{epochs} | "
            f"loss={avg_loss:.4f} (ce={avg_ce:.4f}, risk={avg_risk:.4f}) | "
            f"val_loss={val_m['loss']:.4f} | "
            f"acc={val_m['accuracy']:.4f} | macro_f1={val_m['macro_f1']:.4f} | "
            f"DMR={val_m['dangerous_miss']:.4f} | s4={val_m['s4_underestimate']:.4f} | "
            f"recall=[{', '.join(f'{r:.3f}' for r in val_m['recall_per_class'])}] | "
            f"lr={cur_lr:.1e}"
        )
        log_gpu_memory(f"Epoch {epoch+1:>2} ")

        # 保存 latest checkpoint
        save_training_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, best_dmr, best_macro_f1,
            latest_ckpt,
        )

        # best checkpoint
        improved = (
            val_m["dangerous_miss"] < best_dmr
            or (abs(val_m["dangerous_miss"] - best_dmr) < 1e-8 and val_m["macro_f1"] > best_macro_f1)
        )
        if improved:
            best_dmr = val_m["dangerous_miss"]
            best_macro_f1 = val_m["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_training_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, best_dmr, best_macro_f1,
                best_ckpt,
            )
            no_improve_epochs = 0
            log.info(f"  ★ Best checkpoint updated: DMR={best_dmr:.4f}, macro_f1={best_macro_f1:.4f}")
        else:
            no_improve_epochs += 1
            log.info(f"  No improvement for {no_improve_epochs} epoch(s)")

        # early stopping
        if no_improve_epochs >= early_stop_patience:
            log.info(f"Early stopping triggered (patience={early_stop_patience})")
            break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info("Restored best checkpoint")

    log.info("\n=== Final Validation ===")
    evaluate_wrc_on_shards(
        model, val_shards, device=device,
        batch_size=batch_size * 2,
        max_seq_len=max_seq_len,
        verbose=True,
    )

    writer.close()
    return model


# ================================================================
# 7. 保存 / 加载（修复：存完整结构参数）
# ================================================================

def save_wrc_model(model: CustomWDRNet, tag: str = "wdr_v1") -> str:
    """
    保存模型结构参数 + 权重，确保 load_wrc_model 能完整重建。
    """
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"stage1_wdr_{tag}_{ts}"
    path = os.path.join(MODEL_DIR, name + ".pt")

    torch.save({
        "model_state": model.state_dict(),
        "dense_dim":   model.dense_dim,
        "hidden_dim":  model.hidden_dim,
        "num_layers":  model.num_layers,
        "feature_cols": STAGE1_FEATURE_COLS,
        "st_indices":   ST_INDICES,
    }, path)

    log.info(f"WDR model saved: {path}")
    return path


def load_wrc_model(path: str, device: torch.device = None) -> CustomWDRNet:
    """
    完整重建 WDR 模型并加载权重。
    """
    if device is None:
        device = DEVICE

    data = torch.load(path, map_location=device)
    model = CustomWDRNet(
        dense_dim=data["dense_dim"],
        hidden_dim=data["hidden_dim"],
        num_layers=data["num_layers"],
    ).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()

    log.info(f"WDR model loaded: {path}")
    log.info(f"  dense_dim={data['dense_dim']} hidden={data['hidden_dim']} layers={data['num_layers']}")
    return model


# ================================================================
# 8. 推理接口（pipeline.py 调用入口）
# ================================================================

def predict_proba_wrc(
    model: CustomWDRNet,
    df: pd.DataFrame,
    max_seq_len: int = WRC_MAX_SEQ_LEN,
    batch_size: int = 512,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    批推理入口，由 pipeline.py 的 _process_day_stage2 按 batch 调用。

    输入：  build_stage1_features_batch 返回的 link 级 DataFrame（含 order_id / day 列）
    输出：  追加 pred_p1..p4 / pred_status / pred_entropy / pred_cong_prob / pred_omega
    """
    if device is None:
        device = DEVICE

    model.eval()
    model.to(device)

    # 加载标准化参数（训练时固定的）
    mean_dict, std_dict, feature_cols = load_stats()
    mean_s = pd.Series(mean_dict)
    std_s  = pd.Series(std_dict)

    df = df.copy().reset_index(drop=True)
    df["_row_idx"] = np.arange(len(df))

    # 标准化
    x = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(mean_s)
    scaled = ((x - mean_s) / std_s).astype("float32")
    scaled = scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0).values  # (N, D)

    pred_proba = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)

    grouped    = df.groupby(["order_id", "day"], sort=False)
    order_keys = list(grouped.groups.keys())

    for start in tqdm(
        range(0, len(order_keys), batch_size),
        desc="[WDR predict]",
        dynamic_ncols=True,
        unit="batch",
    ):
        batch_keys = order_keys[start:start + batch_size]

        seqs, row_indices_list, lengths = [], [], []

        for key in batch_keys:
            grp  = grouped.get_group(key)
            idxs = grp["_row_idx"].values
            feats = scaled[idxs]

            if len(feats) > max_seq_len:
                feats = feats[:max_seq_len]
                idxs  = idxs[:max_seq_len]

            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            seqs.append(torch.FloatTensor(feats))
            row_indices_list.append(idxs)
            lengths.append(len(feats))

        # padding（按长度降序）
        sort_order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        seqs             = [seqs[i]             for i in sort_order]
        row_indices_list = [row_indices_list[i] for i in sort_order]
        lengths          = [lengths[i]          for i in sort_order]

        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0.0).to(device)
        lens_t      = torch.LongTensor(lengths).to(device)
        st_t        = seqs_padded[:, :, ST_INDICES]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = model(seqs_padded, lens_t, st_t)
                proba = F.softmax(logits, dim=-1).cpu().numpy()  # (B, T, 4)

        for b, (L, idxs) in enumerate(zip(lengths, row_indices_list)):
            pred_proba[idxs] = proba[b, :L]

    # 写回 DataFrame
    for k in range(NUM_CLASSES):
        df[f"pred_p{k+1}"] = pred_proba[:, k]

    df["pred_status"] = (
        pred_proba * np.array([[1, 2, 3, 4]])
    ).sum(axis=1).astype("float32")

    eps = 1e-10
    df["pred_entropy"] = -(
        pred_proba * np.log(pred_proba + eps)
    ).sum(axis=1).astype("float32")

    df["pred_cong_prob"] = (pred_proba[:, 2] + pred_proba[:, 3]).astype("float32")
    df["pred_omega"]     = (pred_proba[:, 2] * 1 + pred_proba[:, 3] * 3).astype("float32")

    df.drop(columns=["_row_idx"], inplace=True)
    return df