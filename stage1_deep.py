"""
stage1_deep.py — 完整版
Safety-Aware WDR (Wide-Deep-Recurrent) with:
  1. 时空衰减闸门残差传递结构 (Spatiotemporal Decay Gate)
  2. 安全感知多任务损失 (Safety-Aware Multi-Task Loss)
  3. 安全阈值决策规则 (Safety Decision Rule)
  4. 逐 shard 流式训练 + checkpoint + OneCycleLR
  5. AMP 混合精度
  6. 后台 shard 预取
"""

import os
import gc
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
)

from config import (
    MODEL_DIR, NUM_CLASSES, STATUS_CLASSES,
    WRC_HIDDEN_DIM, WRC_NUM_LAYERS, WRC_BATCH_SIZE,
    WRC_EPOCHS, WRC_LR, WRC_MAX_SEQ_LEN,
    FOCAL_GAMMA,
    LOSS_LAMBDA_RISK, LOSS_LAMBDA_FN, LOSS_LAMBDA_S4,
    SAFE_TAU_CONG, SAFE_TAU_RISK,
    SAFE_TAU_S4, SAFE_TAU_RISK_HIGH,
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

ST_FEATURE_NAMES = ["pos_ratio", "cum_travel_time", "sin_arr_slice", "cos_arr_slice"]
ST_INDICES = [STAGE1_FEATURE_COLS.index(n) for n in ST_FEATURE_NAMES]

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ================================================================
# 1. 创新组件：动态时空衰减闸门
# ================================================================

class SpatiotemporalDecayGate(nn.Module):
    """
    从序列特征中剥离时空维度，自适应计算历史记忆的过时衰减权重。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(len(ST_FEATURE_NAMES), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, rnn_step, st_step, h_prev):
        decay = self.gate(st_step)
        return (1.0 - decay) * h_prev + decay * rnn_step


# ================================================================
# 2. WDR 网络
# ================================================================

class CustomWDRNet(nn.Module):
    def __init__(self, dense_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dense_dim = dense_dim

        self.wide = nn.Linear(dense_dim, hidden_dim)

        self.deep = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=dense_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        self.decay_gate = SpatiotemporalDecayGate(hidden_dim)

        fusion_dim = hidden_dim * 3

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
        )

    def forward(self, x, lengths, st_info):
        B, T, D = x.shape

        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )

        h_prev = torch.zeros(B, self.hidden_dim, device=x.device)
        steps = []
        for t in range(T):
            h_prev = self.decay_gate(lstm_out[:, t, :], st_info[:, t, :], h_prev)
            steps.append(h_prev.unsqueeze(1))
        decay_out = torch.cat(steps, dim=1)

        x_flat = x.reshape(B * T, D)
        wide_out = self.wide(x_flat).reshape(B, T, self.hidden_dim)
        deep_out = self.deep(x_flat).reshape(B, T, self.hidden_dim)

        fused = torch.cat(
            [wide_out, deep_out, lstm_out + decay_out], dim=-1
        )

        traffic_logits = self.traffic_head(fused)
        risk_logits = self.risk_head(fused)

        return traffic_logits, risk_logits


# ================================================================
# 3. Safety-Aware Multi-Task Loss
# ================================================================

class SafetyAwareMultiTaskLoss(nn.Module):
    """
    安全感知多任务损失:
      1) focal classification loss          (四分类主任务)
      2) binary risk BCE loss               (拥堵风险辅助: 用真实标签监督)
      3) false-negative penalty             (抑制 s3/s4 → s1/s2 漏判)
      4) extreme-congestion preservation    (抑制 s4 → 非s4 降级)
    """

    def __init__(
        self,
        alpha_cls,
        risk_pos_weight=4.0,
        gamma=2.0,
        lambda_risk=0.5,
        lambda_fn=0.8,
        lambda_s4=0.6,
        ignore_index=-1,
    ):
        super().__init__()
        self.alpha_cls = torch.tensor(alpha_cls, dtype=torch.float32)
        self.gamma = gamma
        self.lambda_risk = lambda_risk
        self.lambda_fn = lambda_fn
        self.lambda_s4 = lambda_s4
        self.ignore_index = ignore_index
        self.risk_pos_weight = torch.tensor([risk_pos_weight], dtype=torch.float32)

    def focal_ce(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        alpha = self.alpha_cls.to(logits.device)[targets]
        loss = -alpha * ((1 - pt) ** self.gamma) * torch.log(pt)
        return loss.mean()

    def forward(self, traffic_logits, risk_logits, targets):
        B, T, C = traffic_logits.shape

        logits_flat = traffic_logits.reshape(B * T, C)
        risk_flat = risk_logits.reshape(B * T)
        y_flat = targets.reshape(B * T)

        valid = y_flat != self.ignore_index
        logits_flat = logits_flat[valid]
        risk_flat = risk_flat[valid]
        y_flat = y_flat[valid]

        if len(y_flat) == 0:
            zero = traffic_logits.sum() * 0.0
            return zero, zero, zero, zero, zero

        # (1) 四分类 focal loss
        loss_cls = self.focal_ce(logits_flat, y_flat)

        # (2) 风险头 BCE（用真实标签: s3/s4=1, s1/s2=0）
        risk_target = (y_flat >= 2).float()
        pos_weight = self.risk_pos_weight.to(risk_flat.device)
        loss_risk = F.binary_cross_entropy_with_logits(
            risk_flat, risk_target, pos_weight=pos_weight, reduction="mean",
        )

        # (3) 漏判惩罚：真实 s3/s4 时, 要求 p_cong 高
        probs = F.softmax(logits_flat, dim=-1)
        p_cong = (probs[:, 2] + probs[:, 3]).clamp_min(1e-8)
        risky_mask = (y_flat >= 2).float()
        if risky_mask.sum() > 0:
            loss_fn = (-(torch.log(p_cong)) * risky_mask).sum() / risky_mask.sum()
        else:
            loss_fn = loss_cls.new_tensor(0.0)

        # (4) s4 保留惩罚：真实 s4 时, 要求 p4 高
        p_s4 = probs[:, 3].clamp_min(1e-8)
        severe_mask = (y_flat == 3).float()
        if severe_mask.sum() > 0:
            loss_s4 = (-(torch.log(p_s4)) * severe_mask).sum() / severe_mask.sum()
        else:
            loss_s4 = loss_cls.new_tensor(0.0)

        total = (
            loss_cls
            + self.lambda_risk * loss_risk
            + self.lambda_fn * loss_fn
            + self.lambda_s4 * loss_s4
        )

        return total, loss_cls, loss_risk, loss_fn, loss_s4


# ================================================================
# 4. 安全决策规则
# ================================================================

def safety_decision_rule(traffic_logits, risk_logits):
    """
    安全阈值决策:
      1. P(s4) >= τ_s4 or risk >= τ_risk_high → 直接 s4
      2. P(s3)+P(s4) >= τ_cong or risk >= τ_risk → s3/s4 择大
      3. 否则 s1/s2 择大
    """
    probs = F.softmax(traffic_logits, dim=-1)
    risk_prob = torch.sigmoid(risk_logits).squeeze(-1)

    p1, p2, p3, p4 = probs[:, :, 0], probs[:, :, 1], probs[:, :, 2], probs[:, :, 3]
    p_cong = p3 + p4

    pred_low = torch.where(p2 > p1, torch.ones_like(p1, dtype=torch.long),
                           torch.zeros_like(p1, dtype=torch.long))
    pred_high = torch.where(p4 > p3, torch.full_like(pred_low, 3),
                            torch.full_like(pred_low, 2))

    pred = pred_low.clone()

    cong_mask = (p_cong >= SAFE_TAU_CONG) | (risk_prob >= SAFE_TAU_RISK)
    pred[cong_mask] = pred_high[cong_mask]

    severe_mask = (p4 >= SAFE_TAU_S4) | (risk_prob >= SAFE_TAU_RISK_HIGH)
    pred[severe_mask] = 3

    return pred, probs, risk_prob


# ================================================================
# 5. GPU / Checkpoint 工具
# ================================================================

def log_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        p = torch.cuda.max_memory_allocated() / 1024**3
        log.info(f"{prefix}[GPU] alloc={a:.2f}GB reserved={r:.2f}GB peak={p:.2f}GB")


def save_training_checkpoint(model, optimizer, scheduler, scaler, epoch,
                             best_dmr, best_macro_f1, ckpt_path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
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


# ================================================================
# 6. 类别分布估计
# ================================================================

def estimate_label_distribution_from_shards(train_shards, max_shards=5):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for shard_info in train_shards[:max_shards]:
        shard = torch.load(shard_info["path"], map_location="cpu")
        for y in shard["y_list"]:
            y = np.asarray(y)
            y = y[(y >= 0) & (y < NUM_CLASSES)]
            counts += np.bincount(y, minlength=NUM_CLASSES)
        del shard
        gc.collect()

    counts = np.maximum(counts, 1)
    total = counts.sum()

    alpha_cls = total / (NUM_CLASSES * counts.astype(float))
    alpha_cls[0] = np.sqrt(alpha_cls[0])
    alpha_cls[1:] = np.clip(alpha_cls[1:] * 1.5, 1.0, 20.0)
    alpha_cls = np.clip(alpha_cls, 0.5, 20.0).astype(np.float32)

    pos = counts[2] + counts[3]
    neg = counts[0] + counts[1]
    risk_pos_weight = float(neg / max(pos, 1))

    log.info(f"Class counts: {counts.tolist()}")
    log.info(f"alpha_cls: {alpha_cls.round(3).tolist()}")
    log.info(f"risk_pos_weight: {risk_pos_weight:.3f}")

    return alpha_cls, risk_pos_weight


# ================================================================
# 7. 验证：逐 shard 聚合 + 安全决策
# ================================================================

def evaluate_wrc_on_shards(model, shard_infos, device,
                           batch_size=256, max_seq_len=100,
                           verbose=False):
    model.eval()

    alpha_cls, risk_pw = estimate_label_distribution_from_shards(
        shard_infos, max_shards=min(3, len(shard_infos))
    )
    criterion = SafetyAwareMultiTaskLoss(
        alpha_cls=alpha_cls,
        risk_pos_weight=risk_pw,
        gamma=FOCAL_GAMMA,
        lambda_risk=LOSS_LAMBDA_RISK,
        lambda_fn=LOSS_LAMBDA_FN,
        lambda_s4=LOSS_LAMBDA_S4,
    )

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
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        with torch.no_grad():
            for X_b, y_b, lens_b in loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                lens_b = lens_b.to(device)
                st_b = X_b[:, :, ST_INDICES]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits, risk_logits = model(X_b, lens_b, st_b)
                    loss, _, _, _, _ = criterion(logits, risk_logits, y_b)

                total_loss += float(loss.item())
                n_batches += 1

                pred_cls, _, _ = safety_decision_rule(logits, risk_logits)

                for b in range(len(lens_b)):
                    L = lens_b[b].item()
                    all_preds.extend(pred_cls[b, :L].cpu().numpy())
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


# ================================================================
# 8. 训练主函数
# ================================================================

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
    log.info("WDR Safety-Aware Streaming Training")
    log.info("=" * 70)
    log.info(
        f"device={device} | hidden={hidden_dim} | layers={num_layers} | "
        f"batch={batch_size} | lr={lr} | max_seq_len={max_seq_len} | epochs={epochs}"
    )
    log.info(
        f"Loss weights: risk={LOSS_LAMBDA_RISK} fn={LOSS_LAMBDA_FN} s4={LOSS_LAMBDA_S4} | "
        f"Thresholds: cong={SAFE_TAU_CONG} risk={SAFE_TAU_RISK} s4={SAFE_TAU_S4} risk_high={SAFE_TAU_RISK_HIGH}"
    )

    mean_dict, std_dict, feature_cols = load_stats()
    train_shards, val_shards, _ = split_train_val_shards(val_ratio=0.15, seed=42)

    log.info(f"Train shards: {len(train_shards)} | Val shards: {len(val_shards)}")

    # 估算 step 数
    # 精确计算每个 epoch 的真实 batch 数（按 shard 分别 ceil）
    steps_per_epoch = sum(
        math.ceil(s["n_orders"] / batch_size) for s in train_shards
    )
    total_steps = epochs * steps_per_epoch

    log.info(
        f"Exact steps_per_epoch={steps_per_epoch} | "
        f"total_steps={total_steps} | "
        f"train_shards={len(train_shards)} | batch_size={batch_size}"
    )

    # 类别分布
    alpha_cls, risk_pos_weight = estimate_label_distribution_from_shards(
        train_shards, max_shards=min(5, len(train_shards))
    )

    # 模型
    model = CustomWDRNet(
        dense_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    log_gpu_memory("Init ")

    criterion = SafetyAwareMultiTaskLoss(
        alpha_cls=alpha_cls,
        risk_pos_weight=risk_pos_weight,
        gamma=FOCAL_GAMMA,
        lambda_risk=LOSS_LAMBDA_RISK,
        lambda_fn=LOSS_LAMBDA_FN,
        lambda_s4=LOSS_LAMBDA_S4,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"wdr_safe_{ts}")
    writer = SummaryWriter(log_dir=tb_dir)
    log.info(f"TensorBoard: tensorboard --logdir {tb_dir}")

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "wdr_latest.pt")
    best_ckpt = os.path.join(CHECKPOINT_DIR, "wdr_best.pt")

    start_epoch = 0
    best_dmr = 1.0
    best_macro_f1 = 0.0
    best_state = None
    global_step = 0
    early_stop_patience = 2
    no_improve_epochs = 0

    if resume and os.path.exists(latest_ckpt):
        ckpt = load_training_checkpoint(latest_ckpt, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt["scheduler_state"]:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception:
                log.warning("Scheduler state incompatible, re-initializing")
        if ckpt["scaler_state"]:
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
        train_fn = 0.0
        train_s4 = 0.0
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

            ds = SingleShardSequenceDataset(shard_obj, max_seq_len=max_seq_len)
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

            pbar = tqdm(
                loader,
                desc=f"E{epoch+1} S{shard_counter:>3}/{len(epoch_shards)}",
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
                    logits, risk_logits = model(X_b, lens_b, st_b)
                    loss, loss_cls, loss_risk, loss_fn, loss_s4_val = criterion(
                        logits, risk_logits, y_b
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    log.warning(
                        f"NaN/Inf loss at E{epoch+1} S{shard_counter} B{batch_idx}, skipping"
                    )
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                # ★ 防止 OneCycleLR 超出 total_steps
                if global_step < total_steps:
                    scheduler.step()
                else:
                    log.warning(
                        f"Scheduler step skipped: global_step={global_step}, total_steps={total_steps}"
                    )

                global_step += 1
                train_loss += float(loss.item())
                train_ce += float(loss_cls.item())
                train_risk += float(loss_risk.item())
                train_fn += float(loss_fn.item())
                train_s4 += float(loss_s4_val.item())
                n_batches += 1

                if device.type == "cuda" and batch_idx % 30 == 0:
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix({
                        "loss": f"{train_loss/max(n_batches,1):.4f}",
                        "cls": f"{train_ce/max(n_batches,1):.3f}",
                        "risk": f"{train_risk/max(n_batches,1):.3f}",
                        "fn": f"{train_fn/max(n_batches,1):.3f}",
                        "gpu": f"{alloc:.1f}G",
                        "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    })

            del ds, loader, shard_obj

            if shard_counter % 15 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # 每 50 个 shard 打印一次小结
            if shard_counter % 50 == 0:
                log.info(
                    f"  [E{epoch+1} S{shard_counter}/{len(epoch_shards)}] "
                    f"loss={train_loss/max(n_batches,1):.4f} "
                    f"(cls={train_ce/max(n_batches,1):.3f} "
                    f"risk={train_risk/max(n_batches,1):.3f} "
                    f"fn={train_fn/max(n_batches,1):.3f} "
                    f"s4={train_s4/max(n_batches,1):.3f}) | "
                    f"lr={optimizer.param_groups[0]['lr']:.1e} | "
                    f"step={global_step}"
                )

        pf.close()

        avg_loss = train_loss / max(n_batches, 1)
        avg_ce = train_ce / max(n_batches, 1)
        avg_risk = train_risk / max(n_batches, 1)
        avg_fn = train_fn / max(n_batches, 1)
        avg_s4 = train_s4 / max(n_batches, 1)

        val_m = evaluate_wrc_on_shards(
            model, val_shards, device=device,
            batch_size=batch_size * 2, max_seq_len=max_seq_len,
            verbose=False,
        )

        cur_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train_total", avg_loss, epoch)
        writer.add_scalar("Loss/train_cls", avg_ce, epoch)
        writer.add_scalar("Loss/train_risk", avg_risk, epoch)
        writer.add_scalar("Loss/train_fn", avg_fn, epoch)
        writer.add_scalar("Loss/train_s4", avg_s4, epoch)
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
            f"loss={avg_loss:.4f} "
            f"(cls={avg_ce:.3f} risk={avg_risk:.3f} fn={avg_fn:.3f} s4={avg_s4:.3f}) | "
            f"val_loss={val_m['loss']:.4f} | "
            f"acc={val_m['accuracy']:.4f} | macro_f1={val_m['macro_f1']:.4f} | "
            f"DMR={val_m['dangerous_miss']:.4f} | s4={val_m['s4_underestimate']:.4f} | "
            f"recall=[{', '.join(f'{r:.3f}' for r in val_m['recall_per_class'])}] | "
            f"lr={cur_lr:.1e}"
        )
        log_gpu_memory(f"Epoch {epoch+1:>2} ")

        save_training_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch, best_dmr, best_macro_f1,
            latest_ckpt,
        )

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
            log.info(f"  ★ Best: DMR={best_dmr:.4f} macro_f1={best_macro_f1:.4f}")
        else:
            no_improve_epochs += 1
            log.info(f"  No improvement for {no_improve_epochs} epoch(s)")

        if no_improve_epochs >= early_stop_patience:
            log.info(f"Early stopping (patience={early_stop_patience})")
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
        batch_size=batch_size * 2, max_seq_len=max_seq_len,
        verbose=True,
    )

    writer.close()
    return model


# ================================================================
# 9. 保存 / 加载
# ================================================================

def save_wrc_model(model: CustomWDRNet, tag: str = "wdr_safe") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"stage1_wdr_{tag}_{ts}"
    path = os.path.join(MODEL_DIR, name + ".pt")

    torch.save({
        "model_state": model.state_dict(),
        "dense_dim": model.dense_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "feature_cols": STAGE1_FEATURE_COLS,
        "st_indices": ST_INDICES,
    }, path)

    log.info(f"WDR model saved: {path}")
    return path


def load_wrc_model(path: str, device: torch.device = None) -> CustomWDRNet:
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
    log.info(f"  dense={data['dense_dim']} hidden={data['hidden_dim']} layers={data['num_layers']}")
    return model


# ================================================================
# 10. 推理接口（pipeline.py 调用入口）
# ================================================================

def predict_proba_wrc(
    model: CustomWDRNet,
    df: pd.DataFrame,
    max_seq_len: int = WRC_MAX_SEQ_LEN,
    batch_size: int = 512,
    device: torch.device = None,
) -> pd.DataFrame:
    """
    推理接口:
      - 输出原始概率 pred_p1..p4
      - 输出 risk 概率 pred_risk_prob
      - pred_cong_prob = max(p_cong, risk_prob)
      - pred_omega, pred_status, pred_entropy
    """
    if device is None:
        device = DEVICE

    model.eval()
    model.to(device)

    mean_dict, std_dict, feature_cols = load_stats()
    mean_s = pd.Series(mean_dict)
    std_s = pd.Series(std_dict)

    df = df.copy().reset_index(drop=True)
    df["_row_idx"] = np.arange(len(df))

    x = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(mean_s)
    scaled = ((x - mean_s) / std_s).astype("float32")
    scaled = scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0).values

    pred_proba = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    pred_risk = np.zeros(len(df), dtype=np.float32)

    grouped = df.groupby(["order_id", "day"], sort=False)
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
            grp = grouped.get_group(key)
            idxs = grp["_row_idx"].values
            feats = scaled[idxs]

            if len(feats) > max_seq_len:
                feats = feats[:max_seq_len]
                idxs = idxs[:max_seq_len]

            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            seqs.append(torch.FloatTensor(feats))
            row_indices_list.append(idxs)
            lengths.append(len(feats))

        sort_order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        seqs = [seqs[i] for i in sort_order]
        row_indices_list = [row_indices_list[i] for i in sort_order]
        lengths = [lengths[i] for i in sort_order]

        seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0.0).to(device)
        lens_t = torch.LongTensor(lengths).to(device)
        st_t = seqs_padded[:, :, ST_INDICES]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, risk_logits = model(seqs_padded, lens_t, st_t)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                risk_prob = torch.sigmoid(risk_logits).squeeze(-1).cpu().numpy()

        for b, (L, idxs) in enumerate(zip(lengths, row_indices_list)):
            pred_proba[idxs] = probs[b, :L]
            pred_risk[idxs] = risk_prob[b, :L]

    for k in range(NUM_CLASSES):
        df[f"pred_p{k+1}"] = pred_proba[:, k]

    df["pred_status"] = (
        pred_proba * np.array([[1, 2, 3, 4]])
    ).sum(axis=1).astype("float32")

    eps = 1e-10
    df["pred_entropy"] = -(
        pred_proba * np.log(pred_proba + eps)
    ).sum(axis=1).astype("float32")

    raw_cong = pred_proba[:, 2] + pred_proba[:, 3]
    df["pred_risk_prob"] = pred_risk.astype("float32")
    df["pred_cong_prob"] = np.maximum(raw_cong, pred_risk).astype("float32")
    df["pred_omega"] = (pred_proba[:, 2] * 1 + pred_proba[:, 3] * 3).astype("float32")

    df.drop(columns=["_row_idx"], inplace=True)
    return df