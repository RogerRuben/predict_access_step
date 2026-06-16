"""
stage1_deep.py — Triple-Head Ordinal-Primary + Adaptive Aux Weighting

设计哲学:
  1. Ordinal head = 主头，唯一产出 Stage 2/3 的概率分布
  2. Classifier head = 辅助监督，提升 nominal 分类能力
  3. Boundary head = 辅助监督，专门救 s2/s3 边界
  4. L_align = classifier CDF 向 ordinal CDF 几何对齐
  5. L_sharp = boundary head 锐化 ordinal q2 在 s2/s3 边界的精度

损失函数:
  L = L_ord (固定主任务)
    + adaptive_weight(L_cls, L_bnd, L_align, L_sharp)
  辅助任务用 uncertainty weighting 自适应，不再手调 lambda

训练策略:
  - warm-up: 前 N epoch 关闭 L_align / L_sharp，让辅助头先学边界
  - 数据驱动 sequence-level weighted sampling
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
from torch.utils.data import DataLoader, WeightedRandomSampler
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

# Warm-up 配置
WARMUP_EPOCHS = 2


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
# 3. Triple-Head 网络（Ordinal 主导）
# ================================================================

class TripleExpertWDRNet(nn.Module):
    def __init__(self, dense_dim, hidden_dim=64, num_layers=2, min_margin=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dense_dim  = dense_dim

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

        fusion_dim = hidden_dim * 4

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
        self.ordinal_head  = MonotonicOrdinalHead(adapter_dim, adapter_dim // 2, min_margin)
        self.cls_head      = nn.Linear(adapter_dim, NUM_CLASSES)
        self.boundary_head = nn.Linear(adapter_dim // 2, 1)

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
        fused    = torch.cat([wide_out, deep_out, decay_out], dim=-1)

        h_ord = self.ordinal_adapter(fused)
        lq1, lq2, lq3 = self.ordinal_head(h_ord)

        h_cls      = self.cls_adapter(fused)
        cls_logits = self.cls_head(h_cls)

        h_bnd     = self.boundary_adapter(fused)
        bnd_logit = self.boundary_head(h_bnd).squeeze(-1)

        return lq1, lq2, lq3, cls_logits, bnd_logit

    def get_ordinal_probs(self, lq1, lq2, lq3):
        """主输出：只从 ordinal head 恢复概率分布"""
        q1 = torch.sigmoid(lq1)
        q2 = torch.sigmoid(lq2)
        q3 = torch.sigmoid(lq3)

        p4 = F.relu(q3)
        p3 = F.relu(q2 - q3)
        p2 = F.relu(q1 - q2)
        p1 = F.relu(1.0 - q1)

        probs = torch.stack([p1, p2, p3, p4], dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        return probs, q1, q2, q3


# ================================================================
# 4. 自适应辅助权重（Uncertainty Weighting）
# ================================================================

class AdaptiveAuxWeight(nn.Module):
    """
    主任务 L_main 固定为 1.0
    辅助任务权重在 [min_weight, max_weight] 范围内自适应
    """

    def __init__(self, task_names, max_weights=None, min_weights=None, init_weights=None):
        super().__init__()

        if max_weights is None:
            max_weights = {
                "cls":   0.40,
                "bnd":   0.55,
                "align": 0.05,
                "sharp": 0.15,
            }

        if min_weights is None:
            min_weights = {
                "cls":   0.08,
                "bnd":   0.10,
                "align": 0.00,
                "sharp": 0.03,
            }

        if init_weights is None:
            init_weights = {
                "cls":   0.20,
                "bnd":   0.30,
                "align": 0.01,
                "sharp": 0.05,
            }

        self.max_weights = max_weights
        self.min_weights = min_weights
        self.raw = nn.ParameterDict()

        for name in task_names:
            w0   = init_weights[name]
            wmax = max_weights[name]
            wmin = min_weights[name]

            # w = wmin + (wmax - wmin) * sigmoid(raw)
            # 反解 raw 使 w ≈ w0
            span = max(wmax - wmin, 1e-8)
            ratio = min(max((w0 - wmin) / span, 1e-4), 1 - 1e-4)
            init_raw = np.log(ratio / (1 - ratio))

            self.raw[name] = nn.Parameter(
                torch.tensor(init_raw, dtype=torch.float32)
            )

    def get_weight(self, name: str):
        wmax = self.max_weights[name]
        wmin = self.min_weights[name]
        return wmin + (wmax - wmin) * torch.sigmoid(self.raw[name])

    def get_weight_dict(self):
        return {name: self.get_weight(name).item() for name in self.raw.keys()}

    def get_raw_dict(self):
        return {name: self.raw[name].item() for name in self.raw.keys()}

    def forward(self, main_loss, aux_losses: dict, disabled_keys=None):
        if disabled_keys is None:
            disabled_keys = set()

        total = main_loss
        weight_info = {}
        contrib_info = {}

        main_val = max(float(main_loss.detach().item()), 1e-8)

        for name, loss in aux_losses.items():
            if name in disabled_keys:
                weight_info[name] = 0.0
                contrib_info[name] = 0.0
                continue

            w = self.get_weight(name)
            total = total + w * loss

            w_val = float(w.detach().item())
            loss_val = float(loss.detach().item())
            contrib = (w_val * loss_val) / main_val

            weight_info[name] = w_val
            contrib_info[name] = contrib

        return total, weight_info, contrib_info


# ================================================================
# 5. 联合损失（主 ordinal + 辅助 cls/bnd/align/sharp）
# ================================================================

class OrdinalPrimaryLoss(nn.Module):
    """
    主损失: L_ord (cumulative ordinal focal BCE)
    辅助: L_cls, L_bnd, L_align, L_sharp
    辅助权重: 由 AdaptiveAuxWeight 自适应
    """

    def __init__(self, pw_q1, pw_q2, pw_q3, cls_alpha,
                 gamma_q1=0.0, gamma_q2=2.0, gamma_q3=2.0,
                 gamma_cls=2.0, gamma_bnd=1.5,
                 ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma_q1 = gamma_q1
        self.gamma_q2 = gamma_q2
        self.gamma_q3 = gamma_q3
        self.gamma_cls = gamma_cls
        self.gamma_bnd = gamma_bnd

        self.register_buffer("pw_q1", torch.tensor(pw_q1, dtype=torch.float32))
        self.register_buffer("pw_q2", torch.tensor(pw_q2, dtype=torch.float32))
        self.register_buffer("pw_q3", torch.tensor(pw_q3, dtype=torch.float32))
        self.register_buffer("cls_alpha", torch.tensor(cls_alpha, dtype=torch.float32))

    def _focal_bce(self, logits, targets, pw, gamma):
        pw = pw.to(device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
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
            return z, z, z, z, z, z

        # ---- 主损失: L_ord ----
        l_q1  = self._focal_bce(q1f, (y >= 1).float(), self.pw_q1, self.gamma_q1)
        l_q2  = self._focal_bce(q2f, (y >= 2).float(), self.pw_q2, self.gamma_q2)
        l_q3  = self._focal_bce(q3f, (y >= 3).float(), self.pw_q3, self.gamma_q3)
        l_ord = l_q1 + l_q2 + l_q3

        # ---- 辅助: L_cls ----
        l_cls = self._focal_ce(clf, y, self.cls_alpha, self.gamma_cls)

        # ---- 辅助: L_bnd (只在 s2/s3 样本上) ----
        mid_mask = (y == 1) | (y == 2)
        if mid_mask.sum() > 0:
            bnd_target = (y[mid_mask] == 2).float()
            bce_bnd = F.binary_cross_entropy_with_logits(bndf[mid_mask], bnd_target, reduction="none")
            if self.gamma_bnd > 0:
                p_b  = torch.sigmoid(bndf[mid_mask])
                pt_b = p_b * bnd_target + (1 - p_b) * (1 - bnd_target)
                bce_bnd = (1 - pt_b).pow(self.gamma_bnd) * bce_bnd
            l_bnd = bce_bnd.mean()
        else:
            l_bnd = torch.tensor(0.0, device=dev, dtype=dt)

        # ---- 辅助: L_align (classifier CDF → ordinal CDF) ----
        p_cls = F.softmax(clf, dim=-1)
        f_cls_1 = p_cls[:, 1] + p_cls[:, 2] + p_cls[:, 3]
        f_cls_2 = p_cls[:, 2] + p_cls[:, 3]
        f_cls_3 = p_cls[:, 3]

        with torch.no_grad():
            q1_detach = torch.sigmoid(q1f)
            q2_detach = torch.sigmoid(q2f)
            q3_detach = torch.sigmoid(q3f)

        l_align = (
            F.mse_loss(f_cls_1, q1_detach) +
            F.mse_loss(f_cls_2, q2_detach) +
            F.mse_loss(f_cls_3, q3_detach)
        ) / 3.0

        # ---- 辅助: L_sharp (boundary 锐化 q2 在 s2/s3 边界) ----
        if mid_mask.sum() > 0:
            sharp_target = (y[mid_mask] == 2).float()
            l_sharp = F.binary_cross_entropy_with_logits(
                q2f[mid_mask], sharp_target, reduction="mean"
            )
        else:
            l_sharp = torch.tensor(0.0, device=dev, dtype=dt)

        return l_ord, l_cls, l_bnd, l_align, l_sharp, l_q3
# ================================================================
# 6. 工具
# ================================================================

def log_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        p = torch.cuda.max_memory_allocated() / 1024**3
        log.info(f"{prefix}[GPU] alloc={a:.2f}GB reserved={r:.2f}GB peak={p:.2f}GB")


def save_training_checkpoint(model, optimizer, scheduler, scaler,
                             aux_weight_module, epoch, best_score, ckpt_path):
    torch.save({
        "epoch": epoch, "best_score": best_score,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state":    scaler.state_dict()     if scaler     else None,
        "aux_weight_state": aux_weight_module.state_dict(),
        "dense_dim": model.dense_dim, "hidden_dim": model.hidden_dim,
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
        del shard; gc.collect()
    counts = np.maximum(counts, 1)

    def eff(n, beta):
        return (1.0 - beta ** n) / (1.0 - beta)

    n1p = counts[1]+counts[2]+counts[3]; n1n = counts[0]
    n2p = counts[2]+counts[3];           n2n = counts[0]+counts[1]
    n3p = counts[3];                     n3n = counts[0]+counts[1]+counts[2]

    pw_q1 = float(np.clip(np.sqrt(eff(n1n,0.999)/max(eff(n1p,0.999),1e-8)),   1.0, 4.0))
    pw_q2 = float(np.clip(np.sqrt(eff(n2n,0.9995)/max(eff(n2p,0.9995),1e-8))*1.2, 2.0, 8.0))
    pw_q3 = float(np.clip(np.sqrt(eff(n3n,0.9999)/max(eff(n3p,0.9999),1e-8))*1.5, 4.0, 15.0))
    pw_q2 = max(pw_q2, pw_q1 + 0.5)
    pw_q3 = max(pw_q3, pw_q2 + 1.0)

    total     = counts.sum()
    cls_alpha = np.sqrt(total / (NUM_CLASSES * counts.astype(float)))
    cls_alpha = np.clip(cls_alpha, 0.5, 5.0).astype(np.float32)

    log.info(f"Counts: {counts.tolist()}")
    log.info(f"Ordinal pw: q1={pw_q1:.2f} q2={pw_q2:.2f} q3={pw_q3:.2f}")
    log.info(f"Cls alpha:  {cls_alpha.round(3).tolist()}")
    return pw_q1, pw_q2, pw_q3, cls_alpha , counts


# ================================================================
# 7. 验证
# ================================================================

def evaluate_on_shards(model, shard_infos, device, criterion,
                       batch_size=256, max_seq_len=100, verbose=False, tau=None):
    model.eval()
    if tau is None:
        tau = float(model.best_tau.item())

    all_ord_preds, all_cls_preds, all_labels, all_rejects = [], [], [], []
    total_loss = total_ord = total_cls = total_bnd = 0.0
    n_batches = 0

    pf = ShardPrefetcher(shard_infos)
    while True:
        _, shard_obj = pf.next()
        if shard_obj is None: break

        ds = SingleShardSequenceDataset(shard_obj, max_seq_len=max_seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0,
                            pin_memory=(device.type == "cuda"))

        with torch.no_grad():
            for X_b, y_b, lens_b in loader:
                X_b    = X_b.to(device, non_blocking=True)
                y_b    = y_b.to(device, non_blocking=True)
                lens_b = lens_b.to(device)
                gate_b = X_b[:, :, GATE_INDICES]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lq1, lq2, lq3, cls_logits, bnd_logit = model(X_b, lens_b, gate_b)
                    l_ord, l_cls, l_bnd, _, _, _ = criterion(
                        lq1, lq2, lq3, cls_logits, bnd_logit, y_b
                    )

                total_ord += float(l_ord.item())
                total_cls += float(l_cls.item())
                total_bnd += float(l_bnd.item())
                n_batches += 1

                # 主输出：ordinal probability
                probs, q1, q2, q3 = model.get_ordinal_probs(lq1, lq2, lq3)

                # ordinal quantile decision
                cdf = probs.cumsum(dim=-1)
                ord_pred = (cdf >= tau).float().argmax(dim=-1)

                # classifier argmax
                cls_pred = cls_logits.argmax(dim=-1)

                # padding mask
                B, T = lq1.shape
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                mask  = lens_b.unsqueeze(1) > t_idx

                # entropy + reject
                eps = 1e-10
                safe_p  = probs.clamp_min(eps)
                entropy = -(safe_p * torch.log(safe_p)).sum(dim=-1)
                entropy = torch.where(mask, entropy, torch.zeros_like(entropy))
                should_reject = (entropy > 1.2) & mask

                for b in range(len(lens_b)):
                    L = lens_b[b].item()
                    all_ord_preds.extend(ord_pred[b, :L].cpu().numpy())
                    all_cls_preds.extend(cls_pred[b, :L].cpu().numpy())
                    all_labels.extend(y_b[b, :L].cpu().numpy())
                    all_rejects.extend(should_reject[b, :L].cpu().numpy())

        del ds, loader, shard_obj; gc.collect()
    pf.close()

    y_t   = np.array(all_labels);   m = y_t >= 0; y_t = y_t[m]
    ord_p = np.array(all_ord_preds)[m]
    cls_p = np.array(all_cls_preds)[m]
    rej   = np.array(all_rejects)[m]
    high  = y_t >= 2; s4a = y_t == 3

    def metrics(pred, name):
        acc = accuracy_score(y_t, pred)
        mf1 = f1_score(y_t, pred, average="macro")
        dmr = float((pred < 2)[high].sum() / max(high.sum(), 1))
        s4u = float((pred < 3)[s4a].sum() / max(s4a.sum(), 1))
        cm  = confusion_matrix(y_t, pred, labels=[0,1,2,3])
        rec = [cm[i,i] / max(cm[i].sum(), 1) for i in range(4)]
        return {"name": name, "acc": acc, "mf1": mf1, "dmr": dmr, "s4u": s4u, "rec": rec, "cm": cm}

    r_ord = metrics(ord_p, f"Ordinal(τ={tau:.2f})")
    r_cls = metrics(cls_p, "Classifier")

    if verbose:
        for r in [r_ord, r_cls]:
            ls_n = [f"s{k}" for k in STATUS_CLASSES]
            cm_df = pd.DataFrame(r["cm"], index=[f"true_{l}" for l in ls_n],
                                 columns=[f"pred_{l}" for l in ls_n])
            log.info(f"\n--- {r['name']} ---")
            log.info(f"CM:\n{cm_df.to_string()}")
            log.info(f"Acc={r['acc']:.4f} | mF1={r['mf1']:.4f} | DMR={r['dmr']:.4f} | s4u={r['s4u']:.4f}")
            log.info(f"Recall: [{', '.join(f's{i+1}={rv:.4f}' for i, rv in enumerate(r['rec']))}]")

    return {
        "loss_ord": total_ord / max(n_batches, 1),
        "loss_cls": total_cls / max(n_batches, 1),
        "loss_bnd": total_bnd / max(n_batches, 1),
        "ord_macro_f1": r_ord["mf1"],
        "ord_dmr":      r_ord["dmr"],
        "ord_s4u":      r_ord["s4u"],
        "ord_recall":   r_ord["rec"],
        "cls_macro_f1": r_cls["mf1"],
        "cls_recall":   r_cls["rec"],
        "reject_rate":  float(rej.mean()),
    }


# ================================================================
# 8. τ 搜索
# ================================================================

def search_best_tau(model, val_shards, device, criterion,
                    batch_size=512, max_seq_len=200, tau_candidates=None):
    if tau_candidates is None:
        tau_candidates = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    log.info("\n" + "=" * 60)
    log.info("TAU SEARCH")
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
                y_b = y_b.to(device); lens_b = lens_b.to(device)
                gate_b = X_b[:, :, GATE_INDICES]
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lq1, lq2, lq3, _, _ = model(X_b, lens_b, gate_b)
                q1, q2, q3 = torch.sigmoid(lq1), torch.sigmoid(lq2), torch.sigmoid(lq3)
                B, T = q1.shape
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                lm = (lens_b.unsqueeze(1) > t_idx) & (y_b >= 0)
                all_q1.append(q1[lm].cpu()); all_q2.append(q2[lm].cpu())
                all_q3.append(q3[lm].cpu()); all_y.append(y_b[lm].cpu())
        del ds, loader, shard_obj; gc.collect()
    pf.close()

    q1n = torch.cat(all_q1).numpy(); q2n = torch.cat(all_q2).numpy()
    q3n = torch.cat(all_q3).numpy(); yn  = torch.cat(all_y).numpy()
    log.info(f"Collected {len(yn):,} samples")

    probs = np.stack([1-q1n, q1n-q2n, q2n-q3n, q3n], axis=-1).clip(min=0)
    probs = probs / probs.sum(axis=-1, keepdims=True).clip(min=1e-8)
    cdf = probs.cumsum(axis=-1)
    high = yn >= 2; s4a = yn == 3

    results = []
    log.info(f"\n{'tau':>6s} {'Acc':>8s} {'mF1':>8s} {'DMR':>8s} {'s4u':>8s} "
             f"{'R_s1':>8s} {'R_s2':>8s} {'R_s3':>8s} {'R_s4':>8s}")
    log.info("-" * 78)

    for tau in tau_candidates:
        pred = (cdf >= tau).argmax(axis=1)
        acc  = accuracy_score(yn, pred)
        mf1  = f1_score(yn, pred, average="macro")
        dmr  = float((pred < 2)[high].sum() / max(high.sum(), 1))
        s4u  = float((pred < 3)[s4a].sum() / max(s4a.sum(), 1))
        cm   = confusion_matrix(yn, pred, labels=[0,1,2,3])
        rec  = [cm[i,i] / max(cm[i].sum(), 1) for i in range(4)]
        results.append({"tau": tau, "accuracy": acc, "macro_f1": mf1,
                        "DMR": dmr, "s4_under": s4u,
                        "recall_s1": rec[0], "recall_s2": rec[1],
                        "recall_s3": rec[2], "recall_s4": rec[3]})
        log.info(f"{tau:>6.2f} {acc:>8.4f} {mf1:>8.4f} {dmr:>8.4f} {s4u:>8.4f} "
                 f"{rec[0]:>8.4f} {rec[1]:>8.4f} {rec[2]:>8.4f} {rec[3]:>8.4f}")

    df   = pd.DataFrame(results)
    safe = df[(df["DMR"] < 0.20) & (df["recall_s2"] >= 0.15) & (df["recall_s3"] >= 0.08)]
    if len(safe) > 0:
        best_row = safe.loc[safe["macro_f1"].idxmax()]
    else:
        safe2 = df[df["DMR"] < 0.30]
        best_row = safe2.loc[safe2["macro_f1"].idxmax()] if len(safe2) > 0 else df.loc[df["DMR"].idxmin()]

    best_tau = float(best_row["tau"])
    log.info(f"\n★ Best τ = {best_tau:.2f}")
    model.best_tau.fill_(best_tau)
    return best_tau, df
# ================================================================
# 9. 训练
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
    """
    Triple-Head / Ordinal-Primary 训练主循环
    包含：
      1. 主任务固定 + AdaptiveAuxWeight 辅助任务自适应权重
      2. warm-up: 前 WARMUP_EPOCHS 关闭 align / sharp
      3. WeightedRandomSampler 过采样
      4. shard 级进度日志 + aux weight / contribution 监控
      5. checkpoint / resume
    """
    if device is None:
        device = DEVICE

    log.info("=" * 70)
    log.info("Triple-Head Ordinal-Primary + Adaptive Aux Weighting")
    log.info("=" * 70)

    _, _, feature_cols = load_stats()
    train_shards, val_shards, _ = split_train_val_shards(val_ratio=0.15, seed=42)
    log.info(f"Train: {len(train_shards)} | Val: {len(val_shards)}")

    steps_per_epoch = sum(math.ceil(s["n_orders"] / batch_size) for s in train_shards)
    total_steps = epochs * steps_per_epoch
    log.info(f"steps/epoch={steps_per_epoch} | total={total_steps}")

    # 权重估计
    pw_q1, pw_q2, pw_q3, cls_alpha, global_counts = estimate_weights(
        train_shards, min(5, len(train_shards))
    )
    # 模型
    model = TripleExpertWDRNet(
        len(feature_cols), hidden_dim, num_layers
    ).to(device)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    log_gpu_memory("Init ")

    # 主损失
    criterion = OrdinalPrimaryLoss(
        pw_q1=pw_q1,
        pw_q2=pw_q2,
        pw_q3=pw_q3,
        cls_alpha=cls_alpha,
    ).to(device)

    # ★ 自适应辅助权重
    aux_weight = AdaptiveAuxWeight(
        task_names=["cls", "bnd", "align", "sharp"],
        max_weights={
            "cls":   0.40,
            "bnd":   0.55,
            "align": 0.05,
            "sharp": 0.15,
        },
        init_weights={
            "cls":   0.20,
            "bnd":   0.30,
            "align": 0.01,
            "sharp": 0.05,
        },
    ).to(device)

    # 自检
    aux_params = [n for n, _ in aux_weight.named_parameters()]
    log.info(f"AdaptiveAuxWeight registered params: {aux_params}")

    w_dict = aux_weight.get_weight_dict()
    raw_dict = aux_weight.get_raw_dict()
    log.info("AdaptiveAuxWeight initial weights:")
    for name in ["cls", "bnd", "align", "sharp"]:
        log.info(
            f"  {name:>8s}: raw={raw_dict[name]:+.4f} → weight={w_dict[name]:.4f}"
        )

    # 优化器
    all_params = list(model.parameters()) + list(aux_weight.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
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
    tb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"ordpri_{ts}")
    writer = SummaryWriter(log_dir=tb_dir)
    log.info(f"TensorBoard: tensorboard --logdir {tb_dir}")

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "ordpri_latest.pt")
    best_ckpt   = os.path.join(CHECKPOINT_DIR, "ordpri_best.pt")

    start_epoch, best_score, best_state = 0, -1.0, None
    global_step, no_improve, patience = 0, 0, 3

    # resume
    if resume and os.path.exists(latest_ckpt):
        ckpt = load_training_checkpoint(latest_ckpt, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

        if ckpt.get("scheduler_state"):
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception:
                log.warning("Scheduler state incompatible, re-init scheduler")
        if ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        if ckpt.get("aux_weight_state"):
            try:
                aux_weight.load_state_dict(ckpt["aux_weight_state"])
            except Exception:
                log.warning("AdaptiveAuxWeight state incompatible, using fresh weights")

        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", -1.0)
        log.info(f"Resume epoch {start_epoch} | best_score={best_score:.4f}")

    # ============================================================
    # epoch loop
    # ============================================================
    for epoch in range(start_epoch, epochs):
        model.train()
        t_loss = t_ord = t_cls = t_bnd = t_cons = 0.0
        nb = 0
        epoch_start = time.time()

        # ★ warm-up: 前 WARMUP_EPOCHS 关闭 align / sharp
        # disabled = {"align", "sharp"} if epoch < WARMUP_EPOCHS else set()
        # ★ sharp 从一开始就开（它直接帮助 q2 边界），只 warm-up 关 align
        disabled = {"align"} if epoch < WARMUP_EPOCHS else set()
        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch+1}/{epochs} | {len(train_shards)} shard(s)")
        log.info(f"Warm-up disabled aux tasks: {sorted(disabled) if disabled else 'None'}")
        log.info(f"{'='*60}")

        shards = train_shards.copy()
        np.random.RandomState(epoch + 42).shuffle(shards)

        pf = ShardPrefetcher(shards)
        sc = 0

        while True:
            _, shard_obj = pf.next()
            if shard_obj is None:
                break
            sc += 1

            ds = SingleShardSequenceDataset(
                shard_obj,
                max_seq_len=max_seq_len,
                class_counts=global_counts,
            )

            # ★ 数据驱动 over-sample
            sampler = WeightedRandomSampler(
                weights=ds.sample_weights,
                num_samples=len(ds),
                replacement=True,
            )

            loader = DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

            # shard sampler 状态打印
            w_arr = np.array(ds.sample_weights)
            if sc == 1 or sc % 20 == 0:
                log.info(
                    f"  Shard sampler: min_w={w_arr.min():.2f} max_w={w_arr.max():.2f} "
                    f"mean_w={w_arr.mean():.2f} | high_risk_seq={int((w_arr > 1.5).sum())}/{len(ds)}"
                )

            pbar = tqdm(
                loader,
                desc=f"E{epoch+1} S{sc:>3}/{len(shards)} ({ds.n_orders:,}ord)",
                leave=False,
                dynamic_ncols=True,
                unit="b",
            )

            for bi, (X, y, lens) in enumerate(pbar):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                lens = lens.to(device)
                gate = X[:, :, GATE_INDICES]

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lq1, lq2, lq3, cls_logits, bnd_logit = model(X, lens, gate)

                    # 主 / 辅助 loss
                    l_ord, l_cls, l_bnd, l_align, l_sharp, _ = criterion(
                        lq1, lq2, lq3, cls_logits, bnd_logit, y
                    )

                    # ★ 主任务固定 + 辅助任务自适应
                    loss, weight_info, contrib_info = aux_weight(
                        main_loss=l_ord,
                        aux_losses={
                            "cls":   l_cls,
                            "bnd":   l_bnd,
                            "align": l_align,
                            "sharp": l_sharp,
                        },
                        disabled_keys=disabled,
                    )

                if torch.isnan(loss) or torch.isinf(loss):
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
                t_ord  += float(l_ord.item())
                t_cls  += float(l_cls.item())
                t_bnd  += float(l_bnd.item())
                t_cons += 0.0  # consistency 不再单独显式返回时，可先记 0
                nb += 1

                if device.type == "cuda" and bi % 20 == 0:
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix(
                        loss=f"{t_loss/nb:.4f}",
                        ord=f"{t_ord/nb:.3f}",
                        cls=f"{t_cls/nb:.3f}",
                        bnd=f"{t_bnd/nb:.3f}",
                        gpu=f"{alloc:.1f}G",
                    )

            del ds, loader, shard_obj

            if sc % 30 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # ★ shard 级中间日志：权重 + 有效贡献
            if sc % 30 == 0:
                elapsed = time.time() - epoch_start
                log.info(
                    f"  [E{epoch+1} S{sc}/{len(shards)}] "
                    f"loss={t_loss/nb:.4f} "
                    f"(ord={t_ord/nb:.3f} cls={t_cls/nb:.3f} bnd={t_bnd/nb:.3f}) | "
                    f"aux_w: cls={weight_info.get('cls',0):.3f} "
                    f"bnd={weight_info.get('bnd',0):.3f} "
                    f"align={weight_info.get('align',0):.3f} "
                    f"sharp={weight_info.get('sharp',0):.3f} | "
                    f"aux_r: cls={contrib_info.get('cls',0):.3f} "
                    f"bnd={contrib_info.get('bnd',0):.3f} "
                    f"align={contrib_info.get('align',0):.3f} "
                    f"sharp={contrib_info.get('sharp',0):.3f} | "
                    f"speed={nb/elapsed:.1f}b/s"
                )

        pf.close()
        epoch_time = time.time() - epoch_start
        a = lambda x: x / max(nb, 1)

        log.info(f"\n--- Epoch {epoch+1} Training ---")
        log.info(f"  Batches={nb:,} | Time={epoch_time:.0f}s | Speed={nb/max(epoch_time,1):.1f}b/s")
        log.info(f"  Loss={a(t_loss):.4f} (ord={a(t_ord):.3f} cls={a(t_cls):.3f} bnd={a(t_bnd):.3f})")

        log.info(f"  Adaptive aux weights (epoch {epoch+1}):")
        for name in ["cls", "bnd", "align", "sharp"]:
            log.info(
                f"    {name:>8s}: "
                f"w={weight_info.get(name, 0):.4f} | "
                f"rel_contrib={contrib_info.get(name, 0):.4f}"
            )

        # 验证
        log.info(f"\n--- Epoch {epoch+1} Validation ---")
        val = evaluate_on_shards(
            model, val_shards, device, criterion,
            batch_size * 2, max_seq_len,
            verbose=True, tau=0.55
        )

        # ★ 更任务对齐的 score
        # ★ 更温和的 gate：前期放宽到 0.45，后期收紧到 0.38
        dmr_gate = 0.45 if epoch < 3 else 0.38

        if val["ord_dmr"] > dmr_gate:
            score = -1.0
            log.info(
                f"Score gated to -1 because ord_dmr={val['ord_dmr']:.4f} > {dmr_gate:.2f} "
                f"(epoch {epoch + 1}, gate={'relaxed' if epoch < 3 else 'tightened'})"
            )
        else:
            score = (
                0.50 * (1.0 - val["ord_dmr"]) +
                0.20 * (1.0 - val["ord_s4u"]) +
                0.20 * val["ord_macro_f1"] +
                0.10 * val["cls_macro_f1"]
            )
            log.info(
                f"Score={score:.4f} | "
                f"(1-DMR)={1.0 - val['ord_dmr']:.4f} "
                f"(1-s4u)={1.0 - val['ord_s4u']:.4f} "
                f"ord_mF1={val['ord_macro_f1']:.4f} "
                f"cls_mF1={val['cls_macro_f1']:.4f}"
            )

        # TensorBoard
        writer.add_scalar("Loss/train", a(t_loss), epoch)
        writer.add_scalar("M/ord_mf1", val["ord_macro_f1"], epoch)
        writer.add_scalar("M/cls_mf1", val["cls_macro_f1"], epoch)
        writer.add_scalar("M/DMR", val["ord_dmr"], epoch)
        writer.add_scalar("M/score", score, epoch)

        for name in ["cls", "bnd", "align", "sharp"]:
            writer.add_scalar(f"AuxWeight/{name}", weight_info.get(name, 0), epoch)
            writer.add_scalar(f"AuxContrib/{name}", contrib_info.get(name, 0), epoch)

        save_training_checkpoint(
            model, optimizer, scheduler, scaler,
            aux_weight, epoch, score, latest_ckpt
        )

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_training_checkpoint(
                model, optimizer, scheduler, scaler,
                aux_weight, epoch, score, best_ckpt
            )
            no_improve = 0
            log.info(f"  ★ BEST score={best_score:.4f}")
        else:
            no_improve += 1
            log.info(f"  No improve ({no_improve}/{patience})")

        if no_improve >= patience:
            log.info("  Early stopping.")
            break

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_state:
        model.load_state_dict(best_state)
        log.info("Restored best checkpoint.")

    # τ 搜索
    best_tau, tau_df = search_best_tau(
        model, val_shards, device, criterion,
        batch_size * 2, max_seq_len
    )
    tau_df.to_csv("tau_search_results.csv", index=False)
    log.info(f"\n=== Final Validation (τ={best_tau:.2f}) ===")
    evaluate_on_shards(
        model, val_shards, device, criterion,
        batch_size * 2, max_seq_len,
        verbose=True, tau=best_tau
    )

    writer.close()
    return model


# ================================================================
# 10. 保存 / 加载
# ================================================================

def save_wrc_model(model, tag="ordpri_v1"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_DIR, f"stage1_ordpri_{tag}_{ts}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "dense_dim": model.dense_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "model_type": "triple_ordinal_primary_adaptive",
        "best_tau": float(model.best_tau.item()),
    }, path)
    log.info(f"Saved: {path} (τ={model.best_tau.item():.2f})")
    return path


def load_wrc_model(path, device=None):
    if device is None: device = DEVICE
    d = torch.load(path, map_location=device)
    model = TripleExpertWDRNet(
        d["dense_dim"], d["hidden_dim"], d["num_layers"]
    ).to(device)
    model.load_state_dict(d["model_state"])
    model.eval()
    if "best_tau" in d:
        model.best_tau.fill_(d["best_tau"])
    log.info(f"Loaded: {path} (τ={model.best_tau.item():.2f})")
    return model


# ================================================================
# 11. 推理（Stage 2 主输出 = ordinal probability）
# ================================================================

def predict_proba_wrc(model, df, max_seq_len=WRC_MAX_SEQ_LEN,
                      batch_size=512, device=None):
    if device is None: device = DEVICE
    model.eval(); model.to(device)

    mean_dict, std_dict, feature_cols = load_stats()
    mean_s = pd.Series(mean_dict); std_s = pd.Series(std_dict)

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
    pred_proba  = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    pred_q2     = np.zeros(n, dtype=np.float32)
    pred_q3     = np.zeros(n, dtype=np.float32)
    pred_reject = np.zeros(n, dtype=bool)
    pred_status = np.ones(n, dtype=np.float32)

    tau = float(model.best_tau.item())

    grouped = df.groupby(["order_id", "day"], sort=False)
    keys = list(grouped.groups.keys())

    for start in tqdm(range(0, len(keys), batch_size),
                      desc="[OrdPri predict]", dynamic_ncols=True, unit="b"):
        bk = keys[start:start + batch_size]
        seqs, ri, lens = [], [], []

        for key in bk:
            grp  = grouped.get_group(key)
            idxs = grp["_row_idx"].values
            feats = scaled[idxs]
            if len(feats) > max_seq_len:
                feats, idxs = feats[:max_seq_len], idxs[:max_seq_len]
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            seqs.append(torch.FloatTensor(feats)); ri.append(idxs); lens.append(len(feats))

        so = sorted(range(len(lens)), key=lambda i: lens[i], reverse=True)
        seqs = [seqs[i] for i in so]; ri = [ri[i] for i in so]; lens = [lens[i] for i in so]

        sp = pad_sequence(seqs, batch_first=True, padding_value=0.0).to(device)
        lt = torch.LongTensor(lens).to(device)
        gt = sp[:, :, GATE_INDICES]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                # ★ 只用 ordinal 主头产出 Stage 2/3 所需概率
                lq1, lq2, lq3, _, _ = model(sp, lt, gt)
                probs, q1_t, q2_t, q3_t = model.get_ordinal_probs(lq1, lq2, lq3)

                # hard label 也从 ordinal 主头来
                cdf = probs.cumsum(dim=-1)
                pred_cls_t = (cdf >= tau).float().argmax(dim=-1)

                B, T = lq1.shape
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                mask  = lt.unsqueeze(1) > t_idx

                eps = 1e-10
                safe_p  = probs.clamp_min(eps)
                entropy = -(safe_p * torch.log(safe_p)).sum(dim=-1)
                entropy = torch.where(mask, entropy, torch.zeros_like(entropy))
                should_reject = (entropy > 1.2) & mask

                probs_np  = probs.cpu().numpy()
                q2_np     = q2_t.cpu().numpy()
                q3_np     = q3_t.cpu().numpy()
                reject_np = should_reject.cpu().numpy()
                cls_np    = pred_cls_t.cpu().numpy()

        for b, (L, idxs) in enumerate(zip(lens, ri)):
            pred_proba[idxs]  = probs_np[b, :L]
            pred_q2[idxs]     = q2_np[b, :L]
            pred_q3[idxs]     = q3_np[b, :L]
            pred_reject[idxs] = reject_np[b, :L]
            pred_status[idxs] = cls_np[b, :L].astype(np.float32) + 1.0

    for k in range(NUM_CLASSES):
        df[f"pred_p{k+1}"] = pred_proba[:, k]

    df["pred_status"] = pred_status.astype("float32")
    eps = 1e-10
    safe_p = pred_proba.clip(min=eps)
    df["pred_entropy"] = -(safe_p * np.log(safe_p)).sum(1).astype("float32")
    df["pred_omega"]   = (pred_proba[:, 2] + pred_proba[:, 3] * 3).astype("float32")

    # ★ Stage 2 / Stage 3 主输入
    df["pred_cong_prob"] = pred_q2.astype("float32")
    df["pred_risk_prob"] = pred_q3.astype("float32")
    df["pred_reject"]    = pred_reject

    df.drop(columns=["_row_idx"], inplace=True)
    return df


