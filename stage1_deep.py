"""
stage1_deep.py — 三头专家版 Final
架构升级:
  1. Task-specific adapter：ordinal/classifier 各自独立特征空间
  2. s2/s3 boundary expert head：专门拉开中间边界
  3. 三头决策整合器：classifier + ordinal veto + boundary expert
  4. 和现有 Stage 2/3 完全兼容

输出给 Stage 2/3:
  pred_cong_prob = q2  (ordinal, 稳定风险信号)
  pred_risk_prob = q3  (ordinal, 极端风险)
  pred_entropy        (fused, 不确定性)
  pred_omega          (fused)
"""

import os
import gc
import math
import time
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
)
from feature_eng import STAGE1_FEATURE_COLS
from stage1_dataset import (
    split_train_val_shards, load_stats,
    SingleShardSequenceDataset, collate_fn, ShardPrefetcher,
)
from logger import get_logger

log = get_logger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GATE_FEATURE_NAMES = [
    "pos_ratio", "cum_travel_time",
    "sin_arr_slice", "cos_arr_slice",
    "downstream_cross_time",
]
GATE_INDICES = [STAGE1_FEATURE_COLS.index(n) for n in GATE_FEATURE_NAMES]
PERIODIC_FEATURES = frozenset({"sin_slice", "cos_slice", "sin_arr_slice", "cos_arr_slice"})

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 安全 veto 阈值（可在 config 里覆盖）

TAU_S4_VETO  = 0.50    # 0.25 → 0.50：只有非常高置信才强制 s4
TAU_CONG_MIN = 0.45    # 0.35 → 0.45：更温和的安全兜底


# ================================================================
# 1. 双向解耦衰减门
# ================================================================

class BidirectionalDecoupledDecayGate(nn.Module):
    def __init__(self, rnn_hidden_dim):
        super().__init__()
        gate_input_dim = len(GATE_FEATURE_NAMES)
        bottleneck = max(rnn_hidden_dim // 2, 16)

        def make_gate(out_dim):
            return nn.Sequential(
                nn.Linear(gate_input_dim, bottleneck), nn.Tanh(),
                nn.Linear(bottleneck, out_dim), nn.Sigmoid(),
            ).float()

        self.forward_gate = make_gate(rnn_hidden_dim)
        self.backward_gate = make_gate(rnn_hidden_dim)

    def forward(self, gru_out, gate_info, lengths):
        B, T, H2 = gru_out.shape
        H = H2 // 2
        dev = gru_out.device
        orig_dtype = gru_out.dtype
        fwd_gru, bwd_gru = gru_out[:, :, :H], gru_out[:, :, H:]

        t_idx = torch.arange(T, device=dev, dtype=torch.long).unsqueeze(0)
        mask_2d = (lengths.unsqueeze(1) > t_idx).float().unsqueeze(-1)

        with torch.cuda.amp.autocast(enabled=False):
            assert self.forward_gate[0].weight.dtype == torch.float32
            gate_f32 = gate_info.float()
            fwd_f32, bwd_f32 = fwd_gru.float(), bwd_gru.float()

            gate_flat = gate_f32.reshape(B * T, -1)
            decay_f = self.forward_gate(gate_flat).reshape(B, T, H)
            decay_b = self.backward_gate(gate_flat).reshape(B, T, H)

            h = torch.zeros(B, H, device=dev, dtype=torch.float32)
            fwd_steps = [None] * T
            for t in range(T):
                h_new = (1.0 - decay_f[:, t]) * h + decay_f[:, t] * fwd_f32[:, t]
                a = mask_2d[:, t]
                h = a * h_new + (1.0 - a) * h
                fwd_steps[t] = h
            fwd_out = torch.stack(fwd_steps, dim=1)

            h = torch.zeros(B, H, device=dev, dtype=torch.float32)
            bwd_steps = [None] * T
            for t in range(T - 1, -1, -1):
                h_new = (1.0 - decay_b[:, t]) * h + decay_b[:, t] * bwd_f32[:, t]
                a = mask_2d[:, t]
                h = a * h_new + (1.0 - a) * h
                bwd_steps[t] = h
            bwd_out = torch.stack(bwd_steps, dim=1)

        return torch.cat([fwd_out, bwd_out], dim=-1).to(orig_dtype)


# ================================================================
# 2. 单调序数头（带 margin bias）
# ================================================================

class MonotonicOrdinalHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, min_margin=0.3):
        super().__init__()
        self.min_margin = min_margin
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        )
        self.head_base   = nn.Linear(hidden_dim, 1)
        self.head_delta1 = nn.Linear(hidden_dim, 1)
        self.head_delta2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        base   = self.head_base(h).squeeze(-1)
        delta1 = F.softplus(self.head_delta1(h).squeeze(-1)) + self.min_margin
        delta2 = F.softplus(self.head_delta2(h).squeeze(-1)) + self.min_margin
        return base, base - delta1, base - delta1 - delta2


# ================================================================
# 3. 三头专家 WDR 网络
# ================================================================

class TripleExpertWDRNet(nn.Module):
    """
    Encoder → shared fused
    ├── ordinal_adapter   → ordinal_head   (q1/q2/q3)
    ├── cls_adapter       → classifier_head (4-class softmax)
    └── boundary_adapter  → boundary_head   (s2 vs s3)
    """

    def __init__(self, dense_dim, hidden_dim=64, num_layers=2, min_margin=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dense_dim  = dense_dim

        # Encoder
        self.wide = nn.Linear(dense_dim, hidden_dim)
        self.deep = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=dense_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.decay_gate = BidirectionalDecoupledDecayGate(rnn_hidden_dim=hidden_dim)

        fusion_dim = hidden_dim * 4   # wide(H) + deep(H) + decay(2H)

        # Task-specific adapters
        adapter_dim = hidden_dim
        self.ordinal_adapter = nn.Sequential(
            nn.Linear(fusion_dim, adapter_dim), nn.ReLU(), nn.Dropout(0.1),
        )
        self.cls_adapter = nn.Sequential(
            nn.Linear(fusion_dim, adapter_dim), nn.ReLU(), nn.Dropout(0.1),
        )
        self.boundary_adapter = nn.Sequential(
            nn.Linear(fusion_dim, adapter_dim // 2), nn.ReLU(), nn.Dropout(0.1),
        )

        # Heads
        self.ordinal_head   = MonotonicOrdinalHead(adapter_dim, adapter_dim // 2, min_margin)
        self.cls_head       = nn.Linear(adapter_dim, NUM_CLASSES)
        self.boundary_head  = nn.Linear(adapter_dim // 2, 1)   # logit: P(s3 | y in {s2,s3})

        self.register_buffer("best_tau", torch.tensor(0.55))

    def forward(self, x, lengths, gate_info):
        B, T, D = x.shape

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        decay_out = self.decay_gate(gru_out, gate_info, lengths)

        x_flat   = x.reshape(B * T, D)
        wide_out = self.wide(x_flat).reshape(B, T, self.hidden_dim)
        deep_out = self.deep(x_flat).reshape(B, T, self.hidden_dim)
        fused    = torch.cat([wide_out, deep_out, decay_out], dim=-1)   # (B, T, 4H)

        # Ordinal branch
        h_ord  = self.ordinal_adapter(fused)
        lq1, lq2, lq3 = self.ordinal_head(h_ord)

        # Classifier branch
        h_cls       = self.cls_adapter(fused)
        cls_logits  = self.cls_head(h_cls)                  # (B, T, 4)

        # Boundary branch
        h_bnd        = self.boundary_adapter(fused)
        bnd_logit    = self.boundary_head(h_bnd).squeeze(-1) # (B, T)

        return lq1, lq2, lq3, cls_logits, bnd_logit

    def get_all_probs(self, lq1, lq2, lq3, cls_logits, eta=0.4):
        """
        恢复三套概率分布。
        ★ 对 p_ord 的各分量做 relu 保护，防止 AMP 浮点漂移产生负数概率，
          然后重新归一化，保证概率空间合法。
        """
        q1 = torch.sigmoid(lq1)
        q2 = torch.sigmoid(lq2)
        q3 = torch.sigmoid(lq3)

        # ★ Patch 3: relu 拦截微小负值，防止 mse_loss 平方放大
        p4 = F.relu(q3)
        p3 = F.relu(q2 - q3)
        p2 = F.relu(q1 - q2)
        p1 = F.relu(1.0 - q1)

        p_ord = torch.stack([p1, p2, p3, p4], dim=-1)
        # 归一化，保证和为 1
        p_ord = p_ord / p_ord.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        p_cls = F.softmax(cls_logits, dim=-1)
        p_fuse = eta * p_ord + (1.0 - eta) * p_cls

        return p_ord, p_cls, p_fuse, q1, q2, q3

    def integrated_decision(self, lq1, lq2, lq3, cls_logits, bnd_logit,
                            lengths, reject_entropy_threshold=1.2):
        """
        三头整合决策器（修正版）:
          Step 1  classifier 给出基础类别（s2/s3 最强）
          Step 2  boundary expert 精修 s2 vs s3
          Step 3  ordinal safety veto（最后执行，只做极端覆盖）
        """
        B, T = lq1.shape
        dev = lq1.device

        q1 = torch.sigmoid(lq1).float()
        q2 = torch.sigmoid(lq2).float()
        q3 = torch.sigmoid(lq3).float()

        # ---- Step 1: classifier 基础类别 ----
        p_cls = F.softmax(cls_logits, dim=-1)
        pred = p_cls.argmax(dim=-1)  # (B, T)

        # ---- Step 2: boundary expert 精修 s2 vs s3 ----
        in_middle = (pred == 1) | (pred == 2)  # cls 判为 s2 或 s3
        p_bnd = torch.sigmoid(bnd_logit)  # P(s3 | middle)
        is_s3 = p_bnd >= 0.5
        pred = torch.where(in_middle & is_s3, torch.full_like(pred, 2), pred)
        pred = torch.where(in_middle & ~is_s3, torch.ones_like(pred), pred)

        # ---- Step 3: ordinal safety veto（最后执行）----
        # 只在 q3 非常高时才强制 s4（不再用 0.25 这么低的阈值）
        force_s4 = q3 >= TAU_S4_VETO
        pred = torch.where(force_s4, torch.full_like(pred, 3), pred)

        # 如果 cls 判 s1，但 q2 明显偏高，至少提升到 s2
        force_ge_s2 = (q2 >= TAU_CONG_MIN) & (pred == 0)
        pred = torch.where(force_ge_s2, torch.ones_like(pred), pred)

        # ---- Padding mask ----
        t_idx = torch.arange(T, device=dev).unsqueeze(0)
        mask = lengths.unsqueeze(1) > t_idx
        pred = torch.where(mask, pred, torch.zeros_like(pred))

        # ---- Entropy / Reject ----
        eps = 1e-10
        safe_p = p_cls.clamp_min(eps)
        entropy = -(safe_p * torch.log(safe_p)).sum(dim=-1)
        entropy = torch.where(mask, entropy, torch.zeros_like(entropy))
        should_reject = (entropy > reject_entropy_threshold) & mask

        return pred, should_reject, entropy


# ================================================================
# 4. 三头损失函数
# ================================================================

class TripleExpertLoss(nn.Module):
    """
    L = L_ord + λ_cls * L_cls + λ_bnd * L_bnd + λ_cons * L_cons

    L_bnd: 只在 y ∈ {s2, s3} 的样本上监督边界
    """

    def __init__(self, pw_q1, pw_q2, pw_q3, cls_alpha,
                 gamma_q1=0.0, gamma_q2=2.0, gamma_q3=2.0,
                 gamma_cls=2.0, gamma_bnd=1.5,
                 lambda_cls=1.0, lambda_bnd=0.8, lambda_cons=0.0,
                 ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma_q1 = gamma_q1
        self.gamma_q2 = gamma_q2
        self.gamma_q3 = gamma_q3
        self.gamma_cls = gamma_cls
        self.gamma_bnd = gamma_bnd
        self.lambda_cls  = lambda_cls
        self.lambda_bnd  = lambda_bnd
        self.current_lambda_cons = lambda_cons

        self.register_buffer("pw_q1",     torch.tensor(pw_q1,  dtype=torch.float32))
        self.register_buffer("pw_q2",     torch.tensor(pw_q2,  dtype=torch.float32))
        self.register_buffer("pw_q3",     torch.tensor(pw_q3,  dtype=torch.float32))
        self.register_buffer("cls_alpha", torch.tensor(cls_alpha, dtype=torch.float32))

    def set_consistency_weight(self, v):
        self.current_lambda_cons = float(v)

    def _focal_bce(self, logits, targets, pw, gamma):
        pw = pw.to(device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        if gamma <= 0:
            return bce.mean()
        p  = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        return ((1 - pt).pow(gamma) * bce).mean()

    def _focal_ce(self, logits, targets, alpha, gamma):
        alpha = alpha.to(device=logits.device, dtype=logits.dtype)
        ce = F.cross_entropy(logits, targets, reduction="none")
        if gamma <= 0:
            return (alpha[targets] * ce).mean()
        probs = F.softmax(logits, dim=-1)
        pt    = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        return (alpha[targets] * (1 - pt).pow(gamma) * ce).mean()

    def forward(self, lq1, lq2, lq3, cls_logits, bnd_logit, targets):
        y_flat = targets.reshape(-1)
        valid  = y_flat != self.ignore_index

        q1f  = lq1.reshape(-1)[valid]
        q2f  = lq2.reshape(-1)[valid]
        q3f  = lq3.reshape(-1)[valid]
        clf  = cls_logits.reshape(-1, NUM_CLASSES)[valid]
        bndf = bnd_logit.reshape(-1)[valid]
        y    = y_flat[valid]

        dev, dt = q1f.device, q1f.dtype

        if len(y) == 0:
            z = torch.tensor(0.0, device=dev, dtype=dt)
            return z, z, z, z, z

        # ---- Ordinal loss ----
        l_q1  = self._focal_bce(q1f, (y >= 1).float(), self.pw_q1, self.gamma_q1)
        l_q2  = self._focal_bce(q2f, (y >= 2).float(), self.pw_q2, self.gamma_q2)
        l_q3  = self._focal_bce(q3f, (y >= 3).float(), self.pw_q3, self.gamma_q3)
        l_ord = l_q1 + l_q2 + l_q3

        # ---- Classification loss ----
        l_cls = self._focal_ce(clf, y, self.cls_alpha, self.gamma_cls)

        # ---- Boundary loss (只在 s2/s3 样本上) ----
        mid_mask = (y == 1) | (y == 2)
        if mid_mask.sum() > 0:
            bnd_target = (y[mid_mask] == 2).float()   # s2=0, s3=1
            bce_bnd = F.binary_cross_entropy_with_logits(
                bndf[mid_mask], bnd_target, reduction="none"
            )
            if self.gamma_bnd > 0:
                p_b  = torch.sigmoid(bndf[mid_mask])
                pt_b = p_b * bnd_target + (1 - p_b) * (1 - bnd_target)
                bce_bnd = (1 - pt_b).pow(self.gamma_bnd) * bce_bnd
            l_bnd = bce_bnd.mean()
        else:
            l_bnd = torch.tensor(0.0, device=dev, dtype=dt)

        # ---- Consistency loss (optional warm-up) ----
        if self.current_lambda_cons > 0:
            with torch.no_grad():
                q1 = torch.sigmoid(q1f)
                q2 = torch.sigmoid(q2f)
                q3 = torch.sigmoid(q3f)
                p_ord = torch.stack([1.0 - q1, q1 - q2, q2 - q3, q3], dim=-1)
            p_cls_soft = F.softmax(clf, dim=-1)
            l_cons = F.mse_loss(p_cls_soft, p_ord.detach())

        else:
            l_cons = torch.tensor(0.0, device=dev, dtype=dt)

        total = (l_ord
                 + self.lambda_cls * l_cls
                 + self.lambda_bnd * l_bnd
                 + self.current_lambda_cons * l_cons)

        return total, l_ord, l_cls, l_bnd, l_cons

# ================================================================
# 5. 工具
# ================================================================

def log_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024 ** 3
        r = torch.cuda.memory_reserved() / 1024 ** 3
        p = torch.cuda.max_memory_allocated() / 1024 ** 3
        log.info(f"{prefix}[GPU] alloc={a:.2f}GB reserved={r:.2f}GB peak={p:.2f}GB")

def save_training_checkpoint(model, optimizer, scheduler, scaler, epoch,
                             best_score, ckpt_path):
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_score": best_score,
        "dense_dim": model.dense_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
    }, ckpt_path)

def load_training_checkpoint(ckpt_path, device):
    return torch.load(ckpt_path, map_location=device)

def estimate_weights(train_shards, max_shards=5):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for info in train_shards[:max_shards]:
        shard = torch.load(info["path"], map_location="cpu")
        for y in shard["y_list"]:
            y_arr = np.asarray(y)
            y_arr = y_arr[(y_arr >= 0) & (y_arr < NUM_CLASSES)]
            counts += np.bincount(y_arr, minlength=NUM_CLASSES)
        del shard;
        gc.collect()
    counts = np.maximum(counts, 1)

    def eff(n, beta):
        return (1.0 - beta ** n) / (1.0 - beta)

    n1p = counts[1] + counts[2] + counts[3];
    n1n = counts[0]
    n2p = counts[2] + counts[3];
    n2n = counts[0] + counts[1]
    n3p = counts[3];
    n3n = counts[0] + counts[1] + counts[2]

    pw_q1 = float(np.clip(np.sqrt(eff(n1n, 0.999) / max(eff(n1p, 0.999), 1e-8)), 1.0, 4.0))
    pw_q2 = float(np.clip(np.sqrt(eff(n2n, 0.9995) / max(eff(n2p, 0.9995), 1e-8)) * 1.2, 2.0, 8.0))
    pw_q3 = float(np.clip(np.sqrt(eff(n3n, 0.9999) / max(eff(n3p, 0.9999), 1e-8)) * 1.5, 4.0, 15.0))
    pw_q2 = max(pw_q2, pw_q1 + 0.5)
    pw_q3 = max(pw_q3, pw_q2 + 1.0)

    total = counts.sum()
    cls_alpha = np.sqrt(total / (NUM_CLASSES * counts.astype(float)))
    cls_alpha = np.clip(cls_alpha, 0.5, 5.0).astype(np.float32)

    log.info(f"Counts: {counts.tolist()}")
    log.info(f"Ordinal pw: q1={pw_q1:.2f} q2={pw_q2:.2f} q3={pw_q3:.2f}")
    log.info(f"Cls alpha:  {cls_alpha.round(3).tolist()}")
    return pw_q1, pw_q2, pw_q3, cls_alpha

# ================================================================
# 6. 验证
# ================================================================

def evaluate_on_shards(model, shard_infos, device, criterion,
                       batch_size=256, max_seq_len=100, verbose=False, tau=None):
    model.eval()
    if tau is None:
        tau = float(model.best_tau.item())

    all_int_preds, all_ord_preds, all_cls_preds = [], [], []
    all_labels, all_rejects = [], []
    total_loss = total_ord = total_cls = total_bnd = total_cons = 0.0
    n_batches = 0

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
                    lq1, lq2, lq3, cls_logits, bnd_logit = model(X_b, lens_b, gate_b)
                    loss, l_ord, l_cls, l_bnd, l_cons = criterion(
                        lq1, lq2, lq3, cls_logits, bnd_logit, y_b
                    )

                total_loss += float(loss.item())
                total_ord += float(l_ord.item())
                total_cls += float(l_cls.item())
                total_bnd += float(l_bnd.item())
                total_cons += float(l_cons.item())
                n_batches += 1

                # ★ Patch 1: 统一从 get_all_probs 解包，不再有废弃残留代码
                p_ord, p_cls, p_fuse, q1, q2, q3 = model.get_all_probs(
                    lq1, lq2, lq3, cls_logits
                )

                # Integrated decision（三头整合）
                int_pred, should_reject, _ = model.integrated_decision(
                    lq1.float(), lq2.float(), lq3.float(),
                    cls_logits, bnd_logit, lens_b,
                )

                # Ordinal quantile decision（对比用）
                cdf_tmp = p_ord.cumsum(dim=-1)
                ord_pred = (cdf_tmp >= tau).float().argmax(dim=-1)

                # ★ Patch 2: cls_pred 用 mask 保护 padding 位置
                B, T = cls_logits.shape[:2]
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                pad_mask = (lens_b.unsqueeze(1) > t_idx)  # (B, T) True = valid

                cls_pred_raw = cls_logits.argmax(dim=-1)  # (B, T)
                cls_pred = torch.where(pad_mask, cls_pred_raw,
                                       torch.zeros_like(cls_pred_raw))

                for b in range(len(lens_b)):
                    L = lens_b[b].item()
                    all_int_preds.extend(int_pred[b, :L].cpu().numpy())
                    all_ord_preds.extend(ord_pred[b, :L].cpu().numpy())
                    all_cls_preds.extend(cls_pred[b, :L].cpu().numpy())
                    all_labels.extend(y_b[b, :L].cpu().numpy())
                    all_rejects.extend(should_reject[b, :L].cpu().numpy())

        del ds, loader, shard_obj
        gc.collect()
    pf.close()

    y_t = np.array(all_labels);
    m = y_t >= 0;
    y_t = y_t[m]
    int_p = np.array(all_int_preds)[m]
    ord_p = np.array(all_ord_preds)[m]
    cls_p = np.array(all_cls_preds)[m]
    rej = np.array(all_rejects)[m]
    high = y_t >= 2;
    s4a = y_t == 3

    def metrics(pred, name):
        acc = accuracy_score(y_t, pred)
        mf1 = f1_score(y_t, pred, average="macro")
        dmr = float((pred < 2)[high].sum() / max(high.sum(), 1))
        s4u = float((pred < 3)[s4a].sum() / max(s4a.sum(), 1))
        cm = confusion_matrix(y_t, pred, labels=[0, 1, 2, 3])
        rec = [cm[i, i] / max(cm[i].sum(), 1) for i in range(4)]
        return {"name": name, "acc": acc, "mf1": mf1,
                "dmr": dmr, "s4u": s4u, "rec": rec, "cm": cm}

    r_int = metrics(int_p, "Integrated")
    r_ord = metrics(ord_p, f"Ordinal(τ={tau:.2f})")
    r_cls = metrics(cls_p, "Classifier")

    if verbose:
        for r in [r_int, r_ord, r_cls]:
            ls_n = [f"s{k}" for k in STATUS_CLASSES]
            cm_df = pd.DataFrame(r["cm"],
                                 index=[f"true_{l}" for l in ls_n],
                                 columns=[f"pred_{l}" for l in ls_n])
            log.info(f"\n--- {r['name']} ---")
            log.info(f"Confusion Matrix:\n{cm_df.to_string()}")
            log.info(f"Acc={r['acc']:.4f} | mF1={r['mf1']:.4f} | "
                     f"DMR={r['dmr']:.4f} | s4u={r['s4u']:.4f}")
            log.info(f"Recall: [{', '.join(f's{i + 1}={rv:.4f}' for i, rv in enumerate(r['rec']))}]")

    log.info(f"\n  Int  mF1={r_int['mf1']:.4f} | "
             f"Ord  mF1={r_ord['mf1']:.4f} | "
             f"Cls  mF1={r_cls['mf1']:.4f} | "
             f"DMR(Int)={r_int['dmr']:.4f}")
    log.info(f"  Int  Recall: [{', '.join(f's{i + 1}={rv:.3f}' for i, rv in enumerate(r_int['rec']))}]")
    log.info(f"  Cls  Recall: [{', '.join(f's{i + 1}={rv:.3f}' for i, rv in enumerate(r_cls['rec']))}]")

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_ord": total_ord / max(n_batches, 1),
        "loss_cls": total_cls / max(n_batches, 1),
        "loss_bnd": total_bnd / max(n_batches, 1),
        "loss_cons": total_cons / max(n_batches, 1),
        # ★ Patch 2: score 基于 r_int（整合决策结果），而非 r_cls
        "macro_f1": r_int["mf1"],
        "dangerous_miss": r_int["dmr"],
        "s4_underestimate": r_int["s4u"],
        "recall_per_class": r_int["rec"],
        "reject_rate": float(rej.mean()),
        "cls_macro_f1": r_cls["mf1"],
        "cls_recall": r_cls["rec"],
        "ord_macro_f1": r_ord["mf1"],
        "ord_recall": r_ord["rec"],
    }

# ================================================================
# 7. τ 搜索
# ================================================================

def search_best_tau(model, val_shards, device, criterion,
                    batch_size=512, max_seq_len=200, tau_candidates=None):
    if tau_candidates is None:
        tau_candidates = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    log.info("\n" + "=" * 60)
    log.info("TAU SEARCH (ordinal branch, s2/s3 recall constraints)")
    log.info("=" * 60)

    model.eval()
    all_q1, all_q2, all_q3, all_y = [], [], [], []

    pf = ShardPrefetcher(val_shards)
    while True:
        _, shard_obj = pf.next()
        if shard_obj is None: break
        ds = SingleShardSequenceDataset(shard_obj, max_seq_len=max_seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0,
                            pin_memory=(device.type == "cuda"))
        with torch.no_grad():
            for X_b, y_b, lens_b in loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device);
                lens_b = lens_b.to(device)
                gate_b = X_b[:, :, GATE_INDICES]
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lq1, lq2, lq3, _, _ = model(X_b, lens_b, gate_b)
                q1, q2, q3 = torch.sigmoid(lq1), torch.sigmoid(lq2), torch.sigmoid(lq3)
                B, T = q1.shape
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                lm = (lens_b.unsqueeze(1) > t_idx) & (y_b >= 0)
                all_q1.append(q1[lm].cpu());
                all_q2.append(q2[lm].cpu())
                all_q3.append(q3[lm].cpu());
                all_y.append(y_b[lm].cpu())
        del ds, loader, shard_obj;
        gc.collect()
    pf.close()

    q1n = torch.cat(all_q1).numpy();
    q2n = torch.cat(all_q2).numpy()
    q3n = torch.cat(all_q3).numpy();
    yn = torch.cat(all_y).numpy()
    log.info(f"Collected {len(yn):,} samples")

    probs = np.stack([1 - q1n, q1n - q2n, q2n - q3n, q3n], axis=-1)
    cdf = probs.cumsum(axis=-1)
    high = yn >= 2;
    s4a = yn == 3

    results = []
    log.info(f"\n{'tau':>6s} {'Acc':>8s} {'mF1':>8s} {'DMR':>8s} {'s4u':>8s} "
             f"{'R_s1':>8s} {'R_s2':>8s} {'R_s3':>8s} {'R_s4':>8s}")
    log.info("-" * 78)

    for tau in tau_candidates:
        pred = (cdf >= tau).argmax(axis=1)
        acc = accuracy_score(yn, pred)
        mf1 = f1_score(yn, pred, average="macro")
        dmr = float((pred < 2)[high].sum() / max(high.sum(), 1))
        s4u = float((pred < 3)[s4a].sum() / max(s4a.sum(), 1))
        cm = confusion_matrix(yn, pred, labels=[0, 1, 2, 3])
        rec = [cm[i, i] / max(cm[i].sum(), 1) for i in range(4)]
        results.append({"tau": tau, "accuracy": acc, "macro_f1": mf1,
                        "DMR": dmr, "s4_under": s4u,
                        "recall_s1": rec[0], "recall_s2": rec[1],
                        "recall_s3": rec[2], "recall_s4": rec[3]})
        log.info(f"{tau:>6.2f} {acc:>8.4f} {mf1:>8.4f} {dmr:>8.4f} {s4u:>8.4f} "
                 f"{rec[0]:>8.4f} {rec[1]:>8.4f} {rec[2]:>8.4f} {rec[3]:>8.4f}")

    df = pd.DataFrame(results)
    safe = df[(df["DMR"] < 0.20) & (df["recall_s2"] >= 0.15) & (df["recall_s3"] >= 0.08)]
    if len(safe) > 0:
        best_row = safe.loc[safe["macro_f1"].idxmax()]
    else:
        safe2 = df[df["DMR"] < 0.25]
        best_row = safe2.loc[safe2["macro_f1"].idxmax()] if len(safe2) > 0 else df.loc[df["DMR"].idxmin()]

    best_tau = float(best_row["tau"])
    log.info(f"\n★ Best τ = {best_tau:.2f}")
    log.info(f"  DMR={best_row['DMR']:.4f} | mF1={best_row['macro_f1']:.4f} | "
             f"s2={best_row['recall_s2']:.4f} | s3={best_row['recall_s3']:.4f} | "
             f"s4={best_row['recall_s4']:.4f}")
    model.best_tau.fill_(best_tau)
    return best_tau, df

# ================================================================
# 8. 训练
# ================================================================

def train_wrc_from_shards(
        hidden_dim=WRC_HIDDEN_DIM, num_layers=WRC_NUM_LAYERS,
        batch_size=WRC_BATCH_SIZE, epochs=WRC_EPOCHS,
        lr=WRC_LR, max_seq_len=WRC_MAX_SEQ_LEN,
        device=None, resume=True,
):
    if device is None: device = DEVICE

    log.info("=" * 70)
    log.info("Triple-Expert WDR: Ordinal + Classifier + Boundary Head")
    log.info("=" * 70)

    _, _, feature_cols = load_stats()
    train_shards, val_shards, _ = split_train_val_shards(val_ratio=0.15, seed=42)
    log.info(f"Train: {len(train_shards)} | Val: {len(val_shards)}")

    steps_per_epoch = sum(math.ceil(s["n_orders"] / batch_size) for s in train_shards)
    total_steps = epochs * steps_per_epoch

    pw_q1, pw_q2, pw_q3, cls_alpha = estimate_weights(train_shards, min(5, len(train_shards)))

    model = TripleExpertWDRNet(len(feature_cols), hidden_dim, num_layers).to(device)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    log_gpu_memory("Init ")

    criterion = TripleExpertLoss(
        pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3, cls_alpha=cls_alpha,
        gamma_q1=0.0, gamma_q2=2.0, gamma_q3=2.0,
        gamma_cls=2.0, gamma_bnd=1.5,
        lambda_cls=1.0, lambda_bnd=0.8, lambda_cons=0.0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"triple_{ts}")
    writer = SummaryWriter(log_dir=tb_dir)
    log.info(f"TensorBoard: tensorboard --logdir {tb_dir}")

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "triple_latest.pt")
    best_ckpt = os.path.join(CHECKPOINT_DIR, "triple_best.pt")

    start_epoch, best_score, best_state = 0, -1.0, None
    global_step, no_improve, patience = 0, 0, 3

    if resume and os.path.exists(latest_ckpt):
        ckpt = load_training_checkpoint(latest_ckpt, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except:
                log.warning("Scheduler re-init")
        if ckpt.get("scaler_state"): scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", -1.0)
        log.info(f"Resume epoch {start_epoch} | best_score={best_score:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t_loss = t_ord = t_cls = t_bnd = t_cons = 0.0
        nb = 0;
        epoch_start = time.time()

        # Consistency warm-up
        if epoch < 2:
            criterion.set_consistency_weight(0.0)
        else:
            criterion.set_consistency_weight(0.03)

        shards = train_shards.copy()
        np.random.RandomState(epoch + 42).shuffle(shards)

        log.info(f"\n{'=' * 60}")
        log.info(f"Epoch {epoch + 1}/{epochs} | {len(shards)} shard(s) | "
                 f"cons_λ={criterion.current_lambda_cons:.3f}")
        log.info(f"{'=' * 60}")

        pf = ShardPrefetcher(shards);
        sc = 0

        while True:
            _, shard_obj = pf.next()
            if shard_obj is None: break
            sc += 1

            ds = SingleShardSequenceDataset(shard_obj, max_seq_len)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_fn, num_workers=0,
                                pin_memory=(device.type == "cuda"))
            pbar = tqdm(loader, desc=f"E{epoch + 1} S{sc:>3}/{len(shards)} ({ds.n_orders:,}ord)",
                        leave=False, dynamic_ncols=True, unit="b")

            for bi, (X, y, lens) in enumerate(pbar):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                lens = lens.to(device)
                gate = X[:, :, GATE_INDICES]

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lq1, lq2, lq3, cls_logits, bnd_logit = model(X, lens, gate)
                    loss, l_ord, l_cls, l_bnd, l_cons = criterion(
                        lq1, lq2, lq3, cls_logits, bnd_logit, y
                    )

                if torch.isnan(loss) or torch.isinf(loss): continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if global_step < total_steps: scheduler.step()
                global_step += 1

                t_loss += float(loss.item())
                t_ord += float(l_ord.item())
                t_cls += float(l_cls.item())
                t_bnd += float(l_bnd.item())
                t_cons += float(l_cons.item())
                nb += 1

                if device.type == "cuda" and bi % 20 == 0:
                    alloc = torch.cuda.memory_allocated() / 1024 ** 3
                    pbar.set_postfix(
                        loss=f"{t_loss / nb:.4f}",
                        ord=f"{t_ord / nb:.3f}",
                        cls=f"{t_cls / nb:.3f}",
                        bnd=f"{t_bnd / nb:.3f}",
                        gpu=f"{alloc:.1f}G",
                    )

            del ds, loader, shard_obj
            if sc % 15 == 0:
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
            if sc % 30 == 0:
                elapsed = time.time() - epoch_start
                log.info(f"  [E{epoch + 1} S{sc}/{len(shards)}] loss={t_loss / nb:.4f} "
                         f"(ord={t_ord / nb:.3f} cls={t_cls / nb:.3f} bnd={t_bnd / nb:.3f}) | "
                         f"speed={nb / elapsed:.1f}b/s")

        pf.close()
        epoch_time = time.time() - epoch_start
        a = lambda x: x / max(nb, 1)

        log.info(f"\n--- Epoch {epoch + 1} Training ---")
        log.info(f"  Batches={nb:,} | Time={epoch_time:.0f}s | Speed={nb / max(epoch_time, 1):.1f}b/s")
        log.info(f"  Loss={a(t_loss):.4f} (ord={a(t_ord):.3f} cls={a(t_cls):.3f} "
                 f"bnd={a(t_bnd):.3f} cons={a(t_cons):.3f})")

        log.info(f"\n--- Epoch {epoch + 1} Validation ---")
        val = evaluate_on_shards(model, val_shards, device, criterion,
                                 batch_size * 2, max_seq_len, verbose=True, tau=0.55)

        # 综合评分：兼顾安全（DMR↓）和分类（cls_mF1↑）
        score = 0.4 * val["cls_macro_f1"] + 0.4 * (1.0 - val["dangerous_miss"]) + 0.2 * val["macro_f1"]
        log.info(f"  Score={score:.4f} (cls_mF1={val['cls_macro_f1']:.4f} "
                 f"DMR={val['dangerous_miss']:.4f} int_mF1={val['macro_f1']:.4f})")

        writer.add_scalar("Loss/train", a(t_loss), epoch)
        writer.add_scalar("Loss/ord", a(t_ord), epoch)
        writer.add_scalar("Loss/cls", a(t_cls), epoch)
        writer.add_scalar("Loss/bnd", a(t_bnd), epoch)
        writer.add_scalar("M/int_mf1", val["macro_f1"], epoch)
        writer.add_scalar("M/cls_mf1", val["cls_macro_f1"], epoch)
        writer.add_scalar("M/DMR", val["dangerous_miss"], epoch)
        writer.add_scalar("M/score", score, epoch)
        for i, r in enumerate(val["cls_recall"]):
            writer.add_scalar(f"Cls_R/s{i + 1}", r, epoch)

        save_training_checkpoint(model, optimizer, scheduler, scaler, epoch, score, latest_ckpt)

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_training_checkpoint(model, optimizer, scheduler, scaler, epoch, score, best_ckpt)
            no_improve = 0
            log.info(f"  ★ BEST score={best_score:.4f}")
        else:
            no_improve += 1
            log.info(f"  No improve ({no_improve}/{patience})")

        if no_improve >= patience:
            log.info("  Early stopping.")
            break

        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()

    if best_state:
        model.load_state_dict(best_state)
        log.info("Restored best checkpoint.")

    best_tau, tau_df = search_best_tau(model, val_shards, device, criterion,
                                       batch_size * 2, max_seq_len)
    tau_df.to_csv("tau_search_results.csv", index=False)

    log.info(f"\n=== Final Validation (τ={best_tau:.2f}) ===")
    evaluate_on_shards(model, val_shards, device, criterion,
                       batch_size * 2, max_seq_len, verbose=True, tau=best_tau)

    writer.close()
    return model

# ================================================================
# 9. 保存 / 加载
# ================================================================

def save_wrc_model(model, tag="triple_v1"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_DIR, f"stage1_triple_{tag}_{ts}.pt")
    torch.save({
        "model_state": model.state_dict(),  # ★ best_tau 已经在 state_dict 里，不需要单独存
        "dense_dim": model.dense_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "model_type": "triple_expert_ordinal_cls_boundary",
    }, path)
    log.info(f"Saved: {path} (τ={model.best_tau.item():.2f})")
    return path

def load_wrc_model(path, device=None):
    if device is None:
        device = DEVICE

    d = torch.load(path, map_location=device)

    model = TripleExpertWDRNet(
        d["dense_dim"], d["hidden_dim"], d["num_layers"]
    ).to(device)

    # ★ state_dict 里已经包含 best_tau（register_buffer 自动管理）
    # 不需要单独 fill_，直接 load_state_dict 即可
    model.load_state_dict(d["model_state"])
    model.eval()

    log.info(f"Loaded: {path} (τ={model.best_tau.item():.2f})")
    return model

# ================================================================
# 10. 推理（Stage 2 兼容）
# ================================================================

def predict_proba_wrc(model, df, max_seq_len=WRC_MAX_SEQ_LEN,
                      batch_size=512, device=None):
    if device is None: device = DEVICE
    model.eval();
    model.to(device)

    mean_dict, std_dict, feature_cols = load_stats()
    mean_s = pd.Series(mean_dict);
    std_s = pd.Series(std_dict)

    df = df.copy().reset_index(drop=True)
    df["_row_idx"] = np.arange(len(df))

    x = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(mean_s)
    for col in feature_cols:
        if col in PERIODIC_FEATURES:
            x[col] = x[col].clip(-1.0, 1.0)
        else:
            d = std_s[col] if std_s[col] > 1e-8 else 1.0
            x[col] = (x[col] - mean_s[col]) / d
    scaled = x.astype("float32").replace([np.inf, -np.inf], 0.0).fillna(0.0).values

    n = len(df)
    pred_proba = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    pred_q2 = np.zeros(n, dtype=np.float32)
    pred_q3 = np.zeros(n, dtype=np.float32)
    pred_reject = np.zeros(n, dtype=bool)
    pred_status = np.ones(n, dtype=np.float32)

    grouped = df.groupby(["order_id", "day"], sort=False)
    keys = list(grouped.groups.keys())

    for start in tqdm(range(0, len(keys), batch_size),
                      desc="[Triple predict]", dynamic_ncols=True, unit="b"):
        bk = keys[start:start + batch_size]
        seqs, ri, lens = [], [], []

        for key in bk:
            grp = grouped.get_group(key)
            idxs = grp["_row_idx"].values
            feats = scaled[idxs]
            if len(feats) > max_seq_len:
                feats, idxs = feats[:max_seq_len], idxs[:max_seq_len]
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            seqs.append(torch.FloatTensor(feats))
            ri.append(idxs)
            lens.append(len(feats))

        so = sorted(range(len(lens)), key=lambda i: lens[i], reverse=True)
        seqs = [seqs[i] for i in so]
        ri = [ri[i] for i in so]
        lens = [lens[i] for i in so]

        sp = pad_sequence(seqs, batch_first=True, padding_value=0.0).to(device)
        lt = torch.LongTensor(lens).to(device)
        gt = sp[:, :, GATE_INDICES]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                lq1, lq2, lq3, cls_logits, bnd_logit = model(sp, lt, gt)

                # ★ Patch 1: 统一从 get_all_probs 解包，q2/q3 来源唯一
                p_ord, p_cls, p_fuse, q1_t, q2_t, q3_t = model.get_all_probs(
                    lq1, lq2, lq3, cls_logits, eta=0.4
                )

                # ★ Patch 1: 进决策器前统一 .float()，消除 AMP dtype 隐患
                int_pred, should_reject, _ = model.integrated_decision(
                    q1_t.float(), q2_t.float(), q3_t.float(),
                    cls_logits, bnd_logit, lt,
                )

                probs_np = p_fuse.cpu().numpy()
                q2_np = q2_t.cpu().numpy()
                q3_np = q3_t.cpu().numpy()
                reject_np = should_reject.cpu().numpy()
                cls_np = int_pred.cpu().numpy()

        for b, (L, idxs) in enumerate(zip(lens, ri)):
            pred_proba[idxs] = probs_np[b, :L]
            pred_q2[idxs] = q2_np[b, :L]
            pred_q3[idxs] = q3_np[b, :L]
            pred_reject[idxs] = reject_np[b, :L]
            pred_status[idxs] = cls_np[b, :L].astype(np.float32) + 1.0

    for k in range(NUM_CLASSES):
        df[f"pred_p{k + 1}"] = pred_proba[:, k]

    df["pred_status"] = pred_status.astype("float32")
    eps = 1e-10
    safe_p = pred_proba.clip(min=eps)
    df["pred_entropy"] = -(safe_p * np.log(safe_p)).sum(1).astype("float32")
    df["pred_omega"] = (pred_proba[:, 2] + pred_proba[:, 3] * 3).astype("float32")

    df["pred_cong_prob"] = pred_q2.astype("float32")
    df["pred_risk_prob"] = pred_q3.astype("float32")
    df["pred_reject"] = pred_reject

    df.drop(columns=["_row_idx"], inplace=True)
    return df