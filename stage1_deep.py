"""
stage1_deep.py — 完整版 v3
修复:
  Bug ①: 双向 GRU 衰减门解耦（前向/反向分别衰减）
  Bug ②: sin/cos 周期特征不做 Z-Score
  Bug ③: 门控网络改为沙漏型 Bottleneck
  Bug ④: loss_s4 平滑因子
  Bug ⑤: fusion 用 decay_out 替代 gru_out+decay_out
  Bug ⑥: enforce_sorted=True

架构:
  - Wide-Deep-Recurrent + 双向解耦衰减门 + Safety Loss
  - 逐 shard 流式训练 + checkpoint + OneCycleLR + AMP
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

# 门控特征索引（5维）
GATE_FEATURE_NAMES = [
    "pos_ratio", "cum_travel_time",
    "sin_arr_slice", "cos_arr_slice",
    "downstream_cross_time",
]
GATE_INDICES = [STAGE1_FEATURE_COLS.index(n) for n in GATE_FEATURE_NAMES]

# ★ 修复 Bug ②：标记不应被 Z-Score 的周期特征
PERIODIC_FEATURES = {"sin_slice", "cos_slice", "sin_arr_slice", "cos_arr_slice"}

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ================================================================
# 1. 双向解耦衰减门（修复 Bug ① + ③）
# ================================================================

class BidirectionalDecoupledDecayGate(nn.Module):
    def __init__(self, rnn_hidden_dim: int):
        super().__init__()
        gate_input_dim = len(GATE_FEATURE_NAMES)
        single_hidden = rnn_hidden_dim
        bottleneck = max(single_hidden // 2, 16)

        def make_gate(out_dim):
            return nn.Sequential(
                nn.Linear(gate_input_dim, bottleneck),
                nn.Tanh(),
                nn.Linear(bottleneck, out_dim),
                nn.Sigmoid(),
            ).float()  # ★ __init__ 里固定为 float32

        self.forward_gate  = make_gate(single_hidden)
        self.backward_gate = make_gate(single_hidden)

    def forward(self, gru_out: torch.Tensor,
                gate_info: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        B, T, H2 = gru_out.shape
        H = H2 // 2
        dev = gru_out.device
        orig_dtype = gru_out.dtype

        fwd_gru = gru_out[:, :, :H]
        bwd_gru = gru_out[:, :, H:]

        gate_info_f32 = gate_info.float()
        lengths_cpu = lengths.cpu()  # 避免在循环里反复同步

        with torch.cuda.amp.autocast(enabled=False):
            fwd_gru_f32 = fwd_gru.float()
            bwd_gru_f32 = bwd_gru.float()

            # ---- 前向衰减 t = 0 → T ----
            h_fwd = torch.zeros(B, H, device=dev, dtype=torch.float32)
            fwd_steps = []

            for t in range(T):
                # 全量计算（不切片），用 mask 抑制 padding 样本的状态更新
                decay_f = self.forward_gate(gate_info_f32[:, t, :])
                h_new = (1.0 - decay_f) * h_fwd + decay_f * fwd_gru_f32[:, t, :]

                # ★ 正确做法：用 mask 决定哪些样本更新状态
                # lengths > t 的样本才在这个时间步有真实数据
                active_mask = (lengths_cpu > t).float().unsqueeze(1).to(dev)
                h_fwd = active_mask * h_new + (1.0 - active_mask) * h_fwd

                fwd_steps.append(h_fwd.unsqueeze(1))

            fwd_out = torch.cat(fwd_steps, dim=1).to(orig_dtype)

            # ---- 反向衰减 t = T-1 → 0 ----
            h_bwd = torch.zeros(B, H, device=dev, dtype=torch.float32)
            bwd_steps = [None] * T

            for t in range(T - 1, -1, -1):
                decay_b = self.backward_gate(gate_info_f32[:, t, :])
                h_new = (1.0 - decay_b) * h_bwd + decay_b * bwd_gru_f32[:, t, :]

                # 反向同理：lengths > t 的样本才在时间步 t 有真实数据
                active_mask = (lengths_cpu > t).float().unsqueeze(1).to(dev)
                h_bwd = active_mask * h_new + (1.0 - active_mask) * h_bwd

                bwd_steps[t] = h_bwd.unsqueeze(1)

            bwd_out = torch.cat(bwd_steps, dim=1).to(orig_dtype)

        return torch.cat([fwd_out, bwd_out], dim=-1)

# ================================================================
# 2. WDR 网络 v3
# ================================================================

class CustomWDRNet(nn.Module):
    """
    Wide-Deep-Recurrent v3:
      Wide:  Linear(D, H)
      Deep:  MLP(D → H → H)
      Recurrent: Bi-GRU(D, H, bidir=True) → 双向解耦衰减门
      Fusion: [Wide(H), Deep(H), DecayOut(H*2)] = H*4
    """

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

        self.gru = nn.GRU(
            input_size=dense_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        self.decay_gate = BidirectionalDecoupledDecayGate(rnn_hidden_dim=hidden_dim)

        fusion_dim = hidden_dim * 4  # wide(H) + deep(H) + decay(H*2)

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

    def forward(self, x, lengths, gate_info):
        B, T, D = x.shape

        # ★ Bug ⑥ 修复：collate_fn 已排序，用 enforce_sorted=True
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )

        # ★ Bug ① 修复：双向解耦衰减
        decay_out = self.decay_gate(gru_out, gate_info, lengths)

        x_flat = x.reshape(B * T, D)
        wide_out = self.wide(x_flat).reshape(B, T, self.hidden_dim)
        deep_out = self.deep(x_flat).reshape(B, T, self.hidden_dim)

        # ★ Bug ⑤ 修复：只用 decay_out，不再加 gru_out
        fused = torch.cat([wide_out, deep_out, decay_out], dim=-1)

        traffic_logits = self.traffic_head(fused)
        risk_logits = self.risk_head(fused)

        return traffic_logits, risk_logits


# ================================================================
# 3. Safety-Aware Multi-Task Loss（修复 Bug ④）
# ================================================================

class SafetyAwareMultiTaskLoss(nn.Module):
    def __init__(self, alpha_cls, risk_pos_weight=4.0, gamma=2.0,
                 lambda_risk=0.5, lambda_fn=0.8, lambda_s4=0.6,
                 ignore_index=-1, smooth_eps=1e-4):
        super().__init__()
        self.alpha_cls = torch.tensor(alpha_cls, dtype=torch.float32)
        self.gamma = gamma
        self.lambda_risk = lambda_risk
        self.lambda_fn = lambda_fn
        self.lambda_s4 = lambda_s4
        self.ignore_index = ignore_index
        self.risk_pos_weight = torch.tensor([risk_pos_weight], dtype=torch.float32)
        self.smooth_eps = smooth_eps  # ★ Bug ④ 修复

    def focal_ce(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        alpha = self.alpha_cls.to(logits.device)[targets]
        return (-alpha * ((1 - pt) ** self.gamma) * torch.log(pt)).mean()

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

        loss_cls = self.focal_ce(logits_flat, y_flat)

        risk_target = (y_flat >= 2).float()
        pos_weight = self.risk_pos_weight.to(risk_flat.device)
        loss_risk = F.binary_cross_entropy_with_logits(
            risk_flat, risk_target, pos_weight=pos_weight, reduction="mean"
        )

        probs = F.softmax(logits_flat, dim=-1)

        # 漏判惩罚
        p_cong = (probs[:, 2] + probs[:, 3]).clamp_min(1e-8)
        risky_mask = (y_flat >= 2).float()
        risky_count = risky_mask.sum() + self.smooth_eps  # ★ Bug ④
        loss_fn = (-(torch.log(p_cong)) * risky_mask).sum() / risky_count

        # s4 保留惩罚
        p_s4 = probs[:, 3].clamp_min(1e-8)
        severe_mask = (y_flat == 3).float()
        severe_count = severe_mask.sum() + self.smooth_eps  # ★ Bug ④
        loss_s4 = (-(torch.log(p_s4)) * severe_mask).sum() / severe_count

        total = (loss_cls
                 + self.lambda_risk * loss_risk
                 + self.lambda_fn * loss_fn
                 + self.lambda_s4 * loss_s4)

        return total, loss_cls, loss_risk, loss_fn, loss_s4


# ================================================================
# 4. 安全决策规则
# ================================================================

def safety_decision_rule(traffic_logits, risk_logits):
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
# 5. 工具
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


def load_training_checkpoint(ckpt_path, device):
    return torch.load(ckpt_path, map_location=device)


def estimate_label_distribution_from_shards(train_shards, max_shards=5):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for info in train_shards[:max_shards]:
        shard = torch.load(info["path"], map_location="cpu")
        for y in shard["y_list"]:
            y = np.asarray(y)
            y = y[(y >= 0) & (y < NUM_CLASSES)]
            counts += np.bincount(y, minlength=NUM_CLASSES)
        del shard; gc.collect()

    counts = np.maximum(counts, 1)
    total = counts.sum()
    alpha = total / (NUM_CLASSES * counts.astype(float))
    alpha[0] = np.sqrt(alpha[0])
    alpha[1:] = np.clip(alpha[1:] * 1.5, 1.0, 20.0)
    alpha = np.clip(alpha, 0.5, 20.0).astype(np.float32)

    pos = counts[2] + counts[3]
    neg = counts[0] + counts[1]
    rpw = float(neg / max(pos, 1))

    log.info(f"Class counts: {counts.tolist()}")
    log.info(f"alpha_cls: {alpha.round(3).tolist()}")
    log.info(f"risk_pos_weight: {rpw:.3f}")
    return alpha, rpw


# ================================================================
# 6. 验证
# ================================================================

def evaluate_wrc_on_shards(model, shard_infos, device,
                           batch_size=256, max_seq_len=100, verbose=False):
    model.eval()
    alpha, rpw = estimate_label_distribution_from_shards(
        shard_infos, max_shards=min(3, len(shard_infos))
    )
    criterion = SafetyAwareMultiTaskLoss(
        alpha_cls=alpha, risk_pos_weight=rpw,
        gamma=FOCAL_GAMMA, lambda_risk=LOSS_LAMBDA_RISK,
        lambda_fn=LOSS_LAMBDA_FN, lambda_s4=LOSS_LAMBDA_S4,
    )

    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    pf = ShardPrefetcher(shard_infos)
    while True:
        _, shard_obj = pf.next()
        if shard_obj is None:
            break

        ds = SingleShardSequenceDataset(shard_obj, max_seq_len=max_seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0,
                            pin_memory=(device.type == "cuda"))

        with torch.no_grad():
            for X_b, y_b, lens_b in loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                lens_b = lens_b.to(device)
                gate_b = X_b[:, :, GATE_INDICES]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits, risk = model(X_b, lens_b, gate_b)
                    loss, _, _, _, _ = criterion(logits, risk, y_b)

                total_loss += float(loss.item())
                n_batches += 1

                pred_cls, _, _ = safety_decision_rule(logits, risk)
                for b in range(len(lens_b)):
                    L = lens_b[b].item()
                    all_preds.extend(pred_cls[b, :L].cpu().numpy())
                    all_labels.extend(y_b[b, :L].cpu().numpy())

        del ds, loader, shard_obj; gc.collect()

    pf.close()

    y_t = np.array(all_labels)
    y_p = np.array(all_preds)
    m = y_t >= 0
    y_t, y_p = y_t[m], y_p[m]

    acc = accuracy_score(y_t, y_p)
    mf1 = f1_score(y_t, y_p, average="macro")
    ha = y_t >= 2
    dmr = float((y_p < 2)[ha].sum() / max(ha.sum(), 1))
    s4a = y_t == 3
    s4u = float((y_p < 3)[s4a].sum() / max(s4a.sum(), 1))
    cm = confusion_matrix(y_t, y_p, labels=[0,1,2,3])
    rec = [cm[i,i] / max(cm[i].sum(), 1) for i in range(4)]

    if verbose:
        ls = [f"s{k}" for k in STATUS_CLASSES]
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in ls], columns=[f"pred_{l}" for l in ls])
        log.info(f"Confusion Matrix:\n{cm_df.to_string()}")
        log.info(f"\n{classification_report(y_t, y_p, target_names=ls, digits=4)}")
        for i, l in enumerate(ls):
            log.info(f"  Recall {l}: {rec[i]:.4f}")
        log.info(f"  DMR: {dmr:.4f} | s4_under: {s4u:.4f}")

    return {"loss": total_loss / max(n_batches,1), "accuracy": acc,
            "macro_f1": mf1, "dangerous_miss": dmr,
            "s4_underestimate": s4u, "recall_per_class": rec}


# ================================================================
# 7. 训练
# ================================================================

def train_wrc_from_shards(
    hidden_dim=WRC_HIDDEN_DIM, num_layers=WRC_NUM_LAYERS,
    batch_size=WRC_BATCH_SIZE, epochs=WRC_EPOCHS,
    lr=WRC_LR, max_seq_len=WRC_MAX_SEQ_LEN,
    device=None, resume=True,
):
    if device is None:
        device = DEVICE

    log.info("=" * 70)
    log.info("WDR v3: Bi-GRU Decoupled Decay + Topo Gate + Safety Loss")
    log.info("=" * 70)
    log.info(f"device={device} | hidden={hidden_dim} | layers={num_layers} | "
             f"batch={batch_size} | lr={lr} | seq={max_seq_len} | epochs={epochs}")

    _, _, feature_cols = load_stats()
    train_shards, val_shards, _ = split_train_val_shards(val_ratio=0.15, seed=42)
    log.info(f"Train: {len(train_shards)} shards | Val: {len(val_shards)} shards")

    steps_per_epoch = sum(math.ceil(s["n_orders"] / batch_size) for s in train_shards)
    total_steps = epochs * steps_per_epoch
    log.info(f"steps/epoch={steps_per_epoch} | total={total_steps}")

    alpha, rpw = estimate_label_distribution_from_shards(train_shards, min(5, len(train_shards)))

    model = CustomWDRNet(len(feature_cols), hidden_dim, num_layers).to(device)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    log_gpu_memory("Init ")

    criterion = SafetyAwareMultiTaskLoss(alpha, rpw, FOCAL_GAMMA,
                                         LOSS_LAMBDA_RISK, LOSS_LAMBDA_FN, LOSS_LAMBDA_S4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"wdr_v3_{ts}")
    writer = SummaryWriter(log_dir=tb_dir)
    log.info(f"TB: tensorboard --logdir {tb_dir}")

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "wdr_latest.pt")
    best_ckpt = os.path.join(CHECKPOINT_DIR, "wdr_best.pt")

    start_epoch, best_dmr, best_mf1, best_state = 0, 1.0, 0.0, None
    global_step, no_improve = 0, 0
    patience = 2

    if resume and os.path.exists(latest_ckpt):
        ckpt = load_training_checkpoint(latest_ckpt, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            try: scheduler.load_state_dict(ckpt["scheduler_state"])
            except: log.warning("Scheduler incompatible, re-init")
        if ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_dmr = ckpt["best_dmr"]
        best_mf1 = ckpt["best_macro_f1"]
        log.info(f"Resume epoch {start_epoch} | best_dmr={best_dmr:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t_loss = t_ce = t_risk = t_fn = t_s4 = 0.0
        nb = 0

        shards = train_shards.copy()
        np.random.RandomState(epoch + 42).shuffle(shards)
        log.info(f"\nEpoch {epoch+1}/{epochs} | {len(shards)} shard(s)")

        pf = ShardPrefetcher(shards)
        sc = 0

        while True:
            _, shard_obj = pf.next()
            if shard_obj is None:
                break
            sc += 1

            ds = SingleShardSequenceDataset(shard_obj, max_seq_len)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_fn, num_workers=0,
                                pin_memory=(device.type == "cuda"))

            pbar = tqdm(loader, desc=f"E{epoch+1} S{sc:>3}/{len(shards)}",
                        leave=False, dynamic_ncols=True, unit="b")

            for bi, (X, y, lens) in enumerate(pbar):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                lens = lens.to(device)
                gate = X[:, :, GATE_INDICES]

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits, risk = model(X, lens, gate)
                    loss, lc, lr_, lf, ls = criterion(logits, risk, y)

                if torch.isnan(loss) or torch.isinf(loss):
                    log.warning(f"NaN E{epoch+1} S{sc} B{bi}, skip")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if global_step < total_steps:
                    scheduler.step()
                global_step += 1

                t_loss += float(loss.item())
                t_ce += float(lc.item())
                t_risk += float(lr_.item())
                t_fn += float(lf.item())
                t_s4 += float(ls.item())
                nb += 1

                if device.type == "cuda" and bi % 30 == 0:
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix(loss=f"{t_loss/nb:.4f}", gpu=f"{alloc:.1f}G",
                                     lr=f"{optimizer.param_groups[0]['lr']:.1e}")

            del ds, loader, shard_obj
            if sc % 15 == 0:
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()

            if sc % 50 == 0:
                log.info(f"  [E{epoch+1} S{sc}] loss={t_loss/nb:.4f} "
                         f"(cls={t_ce/nb:.3f} risk={t_risk/nb:.3f} fn={t_fn/nb:.3f} s4={t_s4/nb:.3f})")

        pf.close()
        a = lambda x: x / max(nb, 1)

        val = evaluate_wrc_on_shards(model, val_shards, device, batch_size*2, max_seq_len, False)

        clr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/train", a(t_loss), epoch)
        writer.add_scalar("Loss/cls", a(t_ce), epoch)
        writer.add_scalar("Loss/risk", a(t_risk), epoch)
        writer.add_scalar("Loss/fn", a(t_fn), epoch)
        writer.add_scalar("Loss/s4p", a(t_s4), epoch)
        writer.add_scalar("Loss/val", val["loss"], epoch)
        writer.add_scalar("M/acc", val["accuracy"], epoch)
        writer.add_scalar("M/mf1", val["macro_f1"], epoch)
        writer.add_scalar("M/DMR", val["dangerous_miss"], epoch)
        writer.add_scalar("M/s4u", val["s4_underestimate"], epoch)
        writer.add_scalar("LR", clr, epoch)
        for i, r in enumerate(val["recall_per_class"]):
            writer.add_scalar(f"R/s{i+1}", r, epoch)

        log.info(
            f"Epoch {epoch+1:>2}/{epochs} | "
            f"loss={a(t_loss):.4f} (cls={a(t_ce):.3f} risk={a(t_risk):.3f} fn={a(t_fn):.3f} s4={a(t_s4):.3f}) | "
            f"val={val['loss']:.4f} | acc={val['accuracy']:.4f} | mf1={val['macro_f1']:.4f} | "
            f"DMR={val['dangerous_miss']:.4f} | s4u={val['s4_underestimate']:.4f} | "
            f"rec=[{','.join(f'{r:.3f}' for r in val['recall_per_class'])}] | lr={clr:.1e}"
        )
        log_gpu_memory(f"E{epoch+1} ")

        save_training_checkpoint(model, optimizer, scheduler, scaler,
                                 epoch, best_dmr, best_mf1, latest_ckpt)

        imp = (val["dangerous_miss"] < best_dmr or
               (abs(val["dangerous_miss"] - best_dmr) < 1e-8 and val["macro_f1"] > best_mf1))
        if imp:
            best_dmr = val["dangerous_miss"]
            best_mf1 = val["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_training_checkpoint(model, optimizer, scheduler, scaler,
                                     epoch, best_dmr, best_mf1, best_ckpt)
            no_improve = 0
            log.info(f"  ★ Best: DMR={best_dmr:.4f} mf1={best_mf1:.4f}")
        else:
            no_improve += 1
            log.info(f"  No improve x{no_improve}")

        if no_improve >= patience:
            log.info(f"Early stop (patience={patience})")
            break

        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    if best_state:
        model.load_state_dict(best_state)
        log.info("Restored best")

    log.info("\n=== Final Validation ===")
    evaluate_wrc_on_shards(model, val_shards, device, batch_size*2, max_seq_len, True)
    writer.close()
    return model


# ================================================================
# 8. 保存 / 加载
# ================================================================

def save_wrc_model(model, tag="wdr_v3"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_DIR, f"stage1_wdr_{tag}_{ts}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "dense_dim": model.dense_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "feature_cols": STAGE1_FEATURE_COLS,
        "gate_indices": GATE_INDICES,
    }, path)
    log.info(f"WDR v3 saved: {path}")
    return path


def load_wrc_model(path, device=None):
    if device is None: device = DEVICE
    d = torch.load(path, map_location=device)
    model = CustomWDRNet(d["dense_dim"], d["hidden_dim"], d["num_layers"]).to(device)
    model.load_state_dict(d["model_state"])
    model.eval()
    log.info(f"WDR v3 loaded: {path}")
    return model


# ================================================================
# 9. 推理（★ 修复 Bug ②：sin/cos 不做 Z-Score）
# ================================================================

def predict_proba_wrc(model, df, max_seq_len=WRC_MAX_SEQ_LEN,
                      batch_size=512, device=None):
    if device is None: device = DEVICE
    model.eval(); model.to(device)

    mean_dict, std_dict, feature_cols = load_stats()
    mean_s = pd.Series(mean_dict)
    std_s = pd.Series(std_dict)

    df = df.copy().reset_index(drop=True)
    df["_row_idx"] = np.arange(len(df))

    x = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(mean_s)

    # ★ Bug ② 修复：周期特征不做 Z-Score
    for col in feature_cols:
        if col in PERIODIC_FEATURES:
            x[col] = x[col].clip(-1.0, 1.0)  # 只做 clip，不归一化
        else:
            x[col] = (x[col] - mean_s[col]) / std_s[col]

    scaled = x.astype("float32").replace([np.inf, -np.inf], 0.0).fillna(0.0).values

    pred_proba = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    pred_risk = np.zeros(len(df), dtype=np.float32)

    grouped = df.groupby(["order_id", "day"], sort=False)
    keys = list(grouped.groups.keys())

    for start in tqdm(range(0, len(keys), batch_size),
                      desc="[WDR predict]", dynamic_ncols=True, unit="b"):
        bk = keys[start:start+batch_size]
        seqs, ri_list, lens = [], [], []

        for key in bk:
            grp = grouped.get_group(key)
            idxs = grp["_row_idx"].values
            feats = scaled[idxs]
            if len(feats) > max_seq_len:
                feats, idxs = feats[:max_seq_len], idxs[:max_seq_len]
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            seqs.append(torch.FloatTensor(feats))
            ri_list.append(idxs)
            lens.append(len(feats))

        so = sorted(range(len(lens)), key=lambda i: lens[i], reverse=True)
        seqs = [seqs[i] for i in so]
        ri_list = [ri_list[i] for i in so]
        lens = [lens[i] for i in so]

        sp = pad_sequence(seqs, batch_first=True, padding_value=0.0).to(device)
        lt = torch.LongTensor(lens).to(device)
        gt = sp[:, :, GATE_INDICES]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, rlog = model(sp, lt, gt)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                rp = torch.sigmoid(rlog).squeeze(-1).cpu().numpy()

        for b, (L, idxs) in enumerate(zip(lens, ri_list)):
            pred_proba[idxs] = probs[b, :L]
            pred_risk[idxs] = rp[b, :L]

    for k in range(NUM_CLASSES):
        df[f"pred_p{k+1}"] = pred_proba[:, k]

    df["pred_status"] = (pred_proba * np.array([[1,2,3,4]])).sum(1).astype("float32")
    eps = 1e-10
    df["pred_entropy"] = -(pred_proba * np.log(pred_proba + eps)).sum(1).astype("float32")

    raw_cong = pred_proba[:, 2] + pred_proba[:, 3]
    df["pred_risk_prob"] = pred_risk.astype("float32")
    df["pred_cong_prob"] = np.maximum(raw_cong, pred_risk).astype("float32")
    df["pred_omega"] = (pred_proba[:, 2] + pred_proba[:, 3] * 3).astype("float32")

    df.drop(columns=["_row_idx"], inplace=True)
    return df