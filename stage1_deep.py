"""
stage1_deep.py — 累积序数版 Final
  1. 单调累积序数: q1=P(y>=2), q2=P(y>=3), q3=P(y>=4)
  2. base-delta 保证 q1>=q2>=q3
  3. Improved ordinal pw: effective-number + 分层收缩
  4. Head-specific focal gamma: q1 轻, q2 中, q3 强
  5. Quantile decision + τ 搜索
  6. Bi-GRU + 解耦衰减门 + AMP + shard 流式训练
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
    split_train_val_shards,
    load_stats,
    SingleShardSequenceDataset,
    collate_fn,
    ShardPrefetcher,
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

PERIODIC_FEATURES = frozenset({
    "sin_slice", "cos_slice",
    "sin_arr_slice", "cos_arr_slice",
})

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

_COST_MATRIX_NP = np.array([
    [  0,    1,    2,    3   ],
    [  1,    0,    1.5,  2.5 ],
    [  6,    4,    0,    1   ],
    [ 12,    8,    3,    0   ],
], dtype=np.float32)


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
                nn.Linear(gate_input_dim, bottleneck),
                nn.Tanh(),
                nn.Linear(bottleneck, out_dim),
                nn.Sigmoid(),
            ).float()

        self.forward_gate = make_gate(rnn_hidden_dim)
        self.backward_gate = make_gate(rnn_hidden_dim)

    def forward(self, gru_out, gate_info, lengths):
        B, T, H2 = gru_out.shape
        H = H2 // 2
        dev = gru_out.device
        orig_dtype = gru_out.dtype

        fwd_gru = gru_out[:, :, :H]
        bwd_gru = gru_out[:, :, H:]

        t_indices = torch.arange(T, device=dev, dtype=torch.long).unsqueeze(0)
        mask_2d = (lengths.unsqueeze(1) > t_indices).float().unsqueeze(-1)

        with torch.cuda.amp.autocast(enabled=False):
            assert self.forward_gate[0].weight.dtype == torch.float32, \
                "Gate weights must be float32."

            gate_f32 = gate_info.float()
            fwd_f32 = fwd_gru.float()
            bwd_f32 = bwd_gru.float()

            gate_flat = gate_f32.reshape(B * T, -1)
            decay_f_all = self.forward_gate(gate_flat).reshape(B, T, H)
            decay_b_all = self.backward_gate(gate_flat).reshape(B, T, H)

            h_fwd = torch.zeros(B, H, device=dev, dtype=torch.float32)
            fwd_steps = [None] * T
            for t in range(T):
                decay = decay_f_all[:, t, :]
                h_new = (1.0 - decay) * h_fwd + decay * fwd_f32[:, t, :]
                active = mask_2d[:, t, :]
                h_fwd = active * h_new + (1.0 - active) * h_fwd
                fwd_steps[t] = h_fwd
            fwd_out = torch.stack(fwd_steps, dim=1)

            h_bwd = torch.zeros(B, H, device=dev, dtype=torch.float32)
            bwd_steps = [None] * T
            for t in range(T - 1, -1, -1):
                decay = decay_b_all[:, t, :]
                h_new = (1.0 - decay) * h_bwd + decay * bwd_f32[:, t, :]
                active = mask_2d[:, t, :]
                h_bwd = active * h_new + (1.0 - active) * h_bwd
                bwd_steps[t] = h_bwd
            bwd_out = torch.stack(bwd_steps, dim=1)

        return torch.cat([fwd_out, bwd_out], dim=-1).to(orig_dtype)


# ================================================================
# 2. 单调序数输出头
# ================================================================

class MonotonicOrdinalHead(nn.Module):
    def __init__(self, fusion_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.1),
        )
        self.head_base = nn.Linear(hidden_dim, 1)
        self.head_delta1 = nn.Linear(hidden_dim, 1)
        self.head_delta2 = nn.Linear(hidden_dim, 1)

    def forward(self, fused):
        h = self.shared(fused)
        base = self.head_base(h).squeeze(-1)
        delta1 = F.softplus(self.head_delta1(h).squeeze(-1))
        delta2 = F.softplus(self.head_delta2(h).squeeze(-1))

        logit_q1 = base
        logit_q2 = base - delta1
        logit_q3 = base - delta1 - delta2
        return logit_q1, logit_q2, logit_q3


# ================================================================
# 3. 累积序数 WDR 网络
# ================================================================

class CumulativeOrdinalWDRNet(nn.Module):
    def __init__(self, dense_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dense_dim = dense_dim

        self.wide = nn.Linear(dense_dim, hidden_dim)
        self.deep = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=dense_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.decay_gate = BidirectionalDecoupledDecayGate(rnn_hidden_dim=hidden_dim)
        fusion_dim = hidden_dim * 4
        self.ordinal_head = MonotonicOrdinalHead(fusion_dim, hidden_dim)
        self.register_buffer("cost_matrix", torch.tensor(_COST_MATRIX_NP, dtype=torch.float32))
        self.register_buffer("best_tau", torch.tensor(0.70))

    def forward(self, x, lengths, gate_info):
        B, T, D = x.shape
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        decay_out = self.decay_gate(gru_out, gate_info, lengths)

        x_flat = x.reshape(B * T, D)
        wide_out = self.wide(x_flat).reshape(B, T, self.hidden_dim)
        deep_out = self.deep(x_flat).reshape(B, T, self.hidden_dim)
        fused = torch.cat([wide_out, deep_out, decay_out], dim=-1)

        return self.ordinal_head(fused)

    def get_cumulative_probs(self, lq1, lq2, lq3):
        q1 = torch.sigmoid(lq1)
        q2 = torch.sigmoid(lq2)
        q3 = torch.sigmoid(lq3)
        p4 = q3
        p3 = F.relu(q2 - q3)
        p2 = F.relu(q1 - q2)
        p1 = F.relu(1.0 - q1)
        probs = torch.stack([p1, p2, p3, p4], dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs, q1, q2, q3


# ================================================================
# 4. Focal Ordinal Loss（head-specific gamma）
# ================================================================

class FocalOrdinalLoss(nn.Module):
    """
    三门累积序数 loss:
      q1: gamma 轻（防 s2 塌陷）
      q2: gamma 中等
      q3: gamma 最强（聚焦极端拥堵 hard positive）
    """

    def __init__(self, pw_q1=2.0, pw_q2=5.0, pw_q3=10.0,
                 gamma_q1=0.0, gamma_q2=1.0, gamma_q3=2.0,
                 ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma_q1 = gamma_q1
        self.gamma_q2 = gamma_q2
        self.gamma_q3 = gamma_q3

        # ★ 用 0 维标量 buffer，而不是 shape=(1,)
        self.register_buffer("pw_q1", torch.tensor(pw_q1, dtype=torch.float32))
        self.register_buffer("pw_q2", torch.tensor(pw_q2, dtype=torch.float32))
        self.register_buffer("pw_q3", torch.tensor(pw_q3, dtype=torch.float32))

    def _focal_bce(self, logits, targets, pos_weight_tensor, gamma):
        """
        logits:  (N,)
        targets: (N,)
        """
        # ★ 只做设备/类型对齐，不 item()，不每次重建 tensor
        pos_weight = pos_weight_tensor.to(device=logits.device, dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pos_weight,
            reduction="none"
        )

        if gamma <= 0:
            return bce.mean()

        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal = (1.0 - pt).pow(gamma)
        return (focal * bce).mean()

    def forward(self, lq1, lq2, lq3, targets):
        y_flat = targets.reshape(-1)
        valid = y_flat != self.ignore_index

        q1f = lq1.reshape(-1)[valid]
        q2f = lq2.reshape(-1)[valid]
        q3f = lq3.reshape(-1)[valid]
        y = y_flat[valid]

        dev = q1f.device
        dt = q1f.dtype

        if len(y) == 0:
            zero = torch.tensor(0.0, device=dev, dtype=dt)
            return zero, zero, zero, zero

        loss_q1 = self._focal_bce(q1f, (y >= 1).float(), self.pw_q1, self.gamma_q1)
        loss_q2 = self._focal_bce(q2f, (y >= 2).float(), self.pw_q2, self.gamma_q2)
        loss_q3 = self._focal_bce(q3f, (y >= 3).float(), self.pw_q3, self.gamma_q3)

        total = loss_q1 + loss_q2 + loss_q3
        return total, loss_q1, loss_q2, loss_q3


# ================================================================
# 5. Quantile Decision
# ================================================================

def ordinal_quantile_decision(q1, q2, q3, lengths, tau=0.70,
                              reject_entropy_threshold=1.2):
    B, T = q1.shape
    dev = q1.device

    p4 = q3
    p3 = q2 - q3
    p2 = q1 - q2
    p1 = 1.0 - q1
    probs = torch.stack([p1, p2, p3, p4], dim=-1)

    cdf = probs.cumsum(dim=-1)
    pred_cls = (cdf >= tau).float().argmax(dim=-1)

    t_idx = torch.arange(T, device=dev).unsqueeze(0)
    mask = (lengths.unsqueeze(1) > t_idx)
    pred_cls = torch.where(mask, pred_cls, torch.zeros_like(pred_cls))

    eps = 1e-10
    safe_probs = probs.clamp_min(eps)
    entropy = -(safe_probs * torch.log(safe_probs)).sum(dim=-1)
    entropy = torch.where(mask, entropy, torch.zeros_like(entropy))
    should_reject = (entropy > reject_entropy_threshold) & mask

    return pred_cls, should_reject, entropy, probs


# ================================================================
# 6. Improved Ordinal Pos Weights
# ================================================================

def estimate_ordinal_pos_weights(train_shards, max_shards=5):
    """
    Effective-number + 分层收缩 + 强制单调。
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for info in train_shards[:max_shards]:
        shard = torch.load(info["path"], map_location="cpu")
        for y in shard["y_list"]:
            y_arr = np.asarray(y)
            y_arr = y_arr[(y_arr >= 0) & (y_arr < NUM_CLASSES)]
            counts += np.bincount(y_arr, minlength=NUM_CLASSES)
        del shard; gc.collect()

    counts = np.maximum(counts, 1)

    n_pos_q1 = counts[1] + counts[2] + counts[3]
    n_neg_q1 = counts[0]
    n_pos_q2 = counts[2] + counts[3]
    n_neg_q2 = counts[0] + counts[1]
    n_pos_q3 = counts[3]
    n_neg_q3 = counts[0] + counts[1] + counts[2]

    def eff(n, beta):
        return (1.0 - beta ** n) / (1.0 - beta)

    raw_q1 = eff(n_neg_q1, 0.999) / max(eff(n_pos_q1, 0.999), 1e-8)
    raw_q2 = eff(n_neg_q2, 0.9995) / max(eff(n_pos_q2, 0.9995), 1e-8)
    raw_q3 = eff(n_neg_q3, 0.9999) / max(eff(n_pos_q3, 0.9999), 1e-8)

    pw_q1 = float(np.clip(np.sqrt(raw_q1), 1.0, 4.0))
    pw_q2 = float(np.clip(np.sqrt(raw_q2) * 1.2, 2.0, 8.0))
    pw_q3 = float(np.clip(np.sqrt(raw_q3) * 1.5, 4.0, 15.0))

    pw_q2 = max(pw_q2, pw_q1 + 0.5)
    pw_q3 = max(pw_q3, pw_q2 + 1.0)

    log.info(f"Class counts: {counts.tolist()}")
    log.info(f"Ordinal pw: q1={pw_q1:.2f}  q2={pw_q2:.2f}  q3={pw_q3:.2f}")
    return pw_q1, pw_q2, pw_q3

# ================================================================
# 7. 工具
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
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_dmr": best_dmr, "best_macro_f1": best_macro_f1,
        "dense_dim": model.dense_dim, "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
    }, ckpt_path)


def load_training_checkpoint(ckpt_path, device):
    return torch.load(ckpt_path, map_location=device)


# ================================================================
# 8. 验证
# ================================================================

def evaluate_on_shards(model, shard_infos, device, criterion,
                       batch_size=256, max_seq_len=100, verbose=False, tau=None):
    model.eval()
    if tau is None:
        tau = float(model.best_tau.item())

    all_preds, all_labels, all_rejects = [], [], []
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
                    lq1, lq2, lq3 = model(X_b, lens_b, gate_b)
                    loss, _, _, _ = criterion(lq1, lq2, lq3, y_b)

                total_loss += float(loss.item())
                n_batches += 1

                _, q1, q2, q3 = model.get_cumulative_probs(lq1, lq2, lq3)
                pred_cls, should_reject, _, _ = ordinal_quantile_decision(
                    q1, q2, q3, lens_b, tau=tau,
                )

                for b in range(len(lens_b)):
                    L = lens_b[b].item()
                    all_preds.extend(pred_cls[b, :L].cpu().numpy())
                    all_labels.extend(y_b[b, :L].cpu().numpy())
                    all_rejects.extend(should_reject[b, :L].cpu().numpy())

        del ds, loader, shard_obj; gc.collect()
    pf.close()

    y_t = np.array(all_labels)
    y_p = np.array(all_preds)
    rej = np.array(all_rejects)
    m = y_t >= 0
    y_t, y_p, rej = y_t[m], y_p[m], rej[m]

    acc = accuracy_score(y_t, y_p)
    mf1 = f1_score(y_t, y_p, average="macro")
    high = y_t >= 2
    dmr = float((y_p < 2)[high].sum() / max(high.sum(), 1))
    s4a = y_t == 3
    s4u = float((y_p < 3)[s4a].sum() / max(s4a.sum(), 1))
    rr = float(rej.mean())
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2, 3])
    rec = [cm[i, i] / max(cm[i].sum(), 1) for i in range(4)]

    if verbose:
        ls_n = [f"s{k}" for k in STATUS_CLASSES]
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in ls_n],
                             columns=[f"pred_{l}" for l in ls_n])
        log.info(f"Confusion Matrix (τ={tau:.2f}):\n{cm_df.to_string()}")
        log.info(f"\n{classification_report(y_t, y_p, target_names=ls_n, digits=4)}")
        for i, l in enumerate(ls_n):
            log.info(f"  Recall {l}: {rec[i]:.4f}")
        log.info(f"  DMR={dmr:.4f} | s4u={s4u:.4f} | reject={rr:.4f} | τ={tau:.2f}")

    return {"loss": total_loss / max(n_batches, 1), "accuracy": acc,
            "macro_f1": mf1, "dangerous_miss": dmr,
            "s4_underestimate": s4u, "recall_per_class": rec, "reject_rate": rr}


# ================================================================
# 9. τ 搜索（GPU flatten 版）
# ================================================================

def search_best_tau(model, val_shards, device, criterion,
                    batch_size=512, max_seq_len=200, tau_candidates=None):
    if tau_candidates is None:
        tau_candidates = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    log.info("\n" + "=" * 60)
    log.info("TAU SEARCH")
    log.info("=" * 60)

    model.eval()
    all_q1, all_q2, all_q3, all_y = [], [], [], []

    pf = ShardPrefetcher(val_shards)
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
                y_b = y_b.to(device)
                lens_b = lens_b.to(device)
                gate_b = X_b[:, :, GATE_INDICES]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lq1, lq2, lq3 = model(X_b, lens_b, gate_b)

                q1 = torch.sigmoid(lq1)
                q2 = torch.sigmoid(lq2)
                q3 = torch.sigmoid(lq3)

                B, T = q1.shape
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                label_mask = (lens_b.unsqueeze(1) > t_idx) & (y_b >= 0)

                all_q1.append(q1[label_mask].cpu())
                all_q2.append(q2[label_mask].cpu())
                all_q3.append(q3[label_mask].cpu())
                all_y.append(y_b[label_mask].cpu())

        del ds, loader, shard_obj; gc.collect()
    pf.close()

    q1_np = torch.cat(all_q1).numpy()
    q2_np = torch.cat(all_q2).numpy()
    q3_np = torch.cat(all_q3).numpy()
    y_np = torch.cat(all_y).numpy()

    log.info(f"Collected {len(y_np):,} samples")

    p4 = q3_np; p3 = q2_np - q3_np; p2 = q1_np - q2_np; p1 = 1.0 - q1_np
    probs_all = np.stack([p1, p2, p3, p4], axis=-1)
    cdf_all = probs_all.cumsum(axis=-1)

    high = y_np >= 2
    s4a = y_np == 3
    results = []

    log.info(f"\n{'tau':>6s} {'Acc':>8s} {'mF1':>8s} {'DMR':>8s} {'s4u':>8s} "
             f"{'R_s1':>8s} {'R_s2':>8s} {'R_s3':>8s} {'R_s4':>8s}")
    log.info("-" * 78)

    for tau in tau_candidates:
        pred = (cdf_all >= tau).argmax(axis=1)
        acc = accuracy_score(y_np, pred)
        mf1 = f1_score(y_np, pred, average="macro")
        dmr = float((pred < 2)[high].sum() / max(high.sum(), 1))
        s4u = float((pred < 3)[s4a].sum() / max(s4a.sum(), 1))
        cm = confusion_matrix(y_np, pred, labels=[0, 1, 2, 3])
        rec = [cm[i, i] / max(cm[i].sum(), 1) for i in range(4)]

        results.append({"tau": tau, "accuracy": acc, "macro_f1": mf1,
                        "DMR": dmr, "s4_under": s4u,
                        "recall_s1": rec[0], "recall_s2": rec[1],
                        "recall_s3": rec[2], "recall_s4": rec[3]})

        log.info(f"{tau:>6.2f} {acc:>8.4f} {mf1:>8.4f} {dmr:>8.4f} {s4u:>8.4f} "
                 f"{rec[0]:>8.4f} {rec[1]:>8.4f} {rec[2]:>8.4f} {rec[3]:>8.4f}")

    df = pd.DataFrame(results)
    safe = df[df["DMR"] < 0.15]
    best_row = safe.loc[safe["macro_f1"].idxmax()] if len(safe) > 0 else df.loc[df["DMR"].idxmin()]
    best_tau = float(best_row["tau"])

    log.info(f"\n★ Best τ = {best_tau:.2f}")
    log.info(f"  DMR={best_row['DMR']:.4f} | mF1={best_row['macro_f1']:.4f} | "
             f"s2={best_row['recall_s2']:.4f} | s4={best_row['recall_s4']:.4f}")

    model.best_tau.fill_(best_tau)
    return best_tau, df


# ================================================================
# 10. 训练
# ================================================================

def train_wrc_from_shards(
    hidden_dim=WRC_HIDDEN_DIM, num_layers=WRC_NUM_LAYERS,
    batch_size=WRC_BATCH_SIZE, epochs=WRC_EPOCHS,
    lr=WRC_LR, max_seq_len=WRC_MAX_SEQ_LEN,
    device=None, resume=True,
):
    if device is None: device = DEVICE

    log.info("=" * 70)
    log.info("Cumulative Ordinal WDR Final: Focal + Improved PW + τ Search")
    log.info("=" * 70)
    log.info(f"device={device} | hidden={hidden_dim} | layers={num_layers} | "
             f"batch={batch_size} | lr={lr} | seq={max_seq_len} | epochs={epochs}")

    _, _, feature_cols = load_stats()
    train_shards, val_shards, _ = split_train_val_shards(val_ratio=0.15, seed=42)
    log.info(f"Train: {len(train_shards)} | Val: {len(val_shards)}")

    steps_per_epoch = sum(math.ceil(s["n_orders"] / batch_size) for s in train_shards)
    total_steps = epochs * steps_per_epoch
    log.info(f"steps/epoch={steps_per_epoch} | total={total_steps}")

    pw_q1, pw_q2, pw_q3 = estimate_ordinal_pos_weights(train_shards, min(5, len(train_shards)))

    model = CumulativeOrdinalWDRNet(len(feature_cols), hidden_dim, num_layers).to(device)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    log_gpu_memory("Init ")

    criterion = FocalOrdinalLoss(
        pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
        gamma_q1=0.0, gamma_q2=1.0, gamma_q3=2.0,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"ordinal_{ts}")
    writer = SummaryWriter(log_dir=tb_dir)
    log.info(f"TensorBoard: tensorboard --logdir {tb_dir}")

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "ordinal_latest.pt")
    best_ckpt = os.path.join(CHECKPOINT_DIR, "ordinal_best.pt")

    start_epoch, best_dmr, best_mf1, best_state = 0, 1.0, 0.0, None
    global_step, no_improve, patience = 0, 0, 3

    if resume and os.path.exists(latest_ckpt):
        ckpt = load_training_checkpoint(latest_ckpt, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            try: scheduler.load_state_dict(ckpt["scheduler_state"])
            except: log.warning("Scheduler re-init")
        if ckpt.get("scaler_state"): scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_dmr, best_mf1 = ckpt["best_dmr"], ckpt["best_macro_f1"]
        log.info(f"Resume epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t_loss = t_q1 = t_q2 = t_q3 = 0.0
        nb = 0
        epoch_start = time.time()

        shards = train_shards.copy()
        np.random.RandomState(epoch + 42).shuffle(shards)

        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch+1}/{epochs} | {len(shards)} shard(s) | lr={optimizer.param_groups[0]['lr']:.1e}")
        log.info(f"{'='*60}")

        pf = ShardPrefetcher(shards); sc = 0

        while True:
            _, shard_obj = pf.next()
            if shard_obj is None: break
            sc += 1

            ds = SingleShardSequenceDataset(shard_obj, max_seq_len)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                collate_fn=collate_fn, num_workers=0,
                                pin_memory=(device.type == "cuda"))

            pbar = tqdm(loader, desc=f"E{epoch+1} S{sc:>3}/{len(shards)} ({ds.n_orders:,}ord)",
                        leave=False, dynamic_ncols=True, unit="b")

            for bi, (X, y, lens) in enumerate(pbar):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                lens = lens.to(device)
                gate = X[:, :, GATE_INDICES]

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lq1, lq2, lq3 = model(X, lens, gate)
                    loss, l1, l2, l3 = criterion(lq1, lq2, lq3, y)

                if torch.isnan(loss) or torch.isinf(loss): continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if global_step < total_steps: scheduler.step()
                global_step += 1

                t_loss += float(loss.item())
                t_q1 += float(l1.item()); t_q2 += float(l2.item()); t_q3 += float(l3.item())
                nb += 1

                if device.type == "cuda" and bi % 20 == 0:
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix(loss=f"{t_loss/nb:.4f}", q1=f"{t_q1/nb:.3f}",
                                     q2=f"{t_q2/nb:.3f}", q3=f"{t_q3/nb:.3f}",
                                     gpu=f"{alloc:.1f}G", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

            del ds, loader, shard_obj
            if sc % 15 == 0:
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
            if sc % 30 == 0:
                elapsed = time.time() - epoch_start
                log.info(f"  [E{epoch+1} S{sc}/{len(shards)}] loss={t_loss/nb:.4f} "
                         f"(q1={t_q1/nb:.3f} q2={t_q2/nb:.3f} q3={t_q3/nb:.3f}) | "
                         f"speed={nb/elapsed:.1f}b/s | step={global_step}")

        pf.close()
        epoch_time = time.time() - epoch_start
        a = lambda x: x / max(nb, 1)

        log.info(f"\n--- Epoch {epoch+1} Training ---")
        log.info(f"  Batches={nb:,} | Time={epoch_time:.0f}s | Speed={nb/max(epoch_time,1):.1f}b/s")
        log.info(f"  Loss={a(t_loss):.4f} (q1={a(t_q1):.3f} q2={a(t_q2):.3f} q3={a(t_q3):.3f})")
        log_gpu_memory(f"  E{epoch+1} ")

        log.info(f"\n--- Epoch {epoch+1} Validation (τ=0.70) ---")
        val = evaluate_on_shards(model, val_shards, device, criterion,
                                 batch_size * 2, max_seq_len, verbose=False, tau=0.70)
        log.info(f"  Val loss={val['loss']:.4f} | Acc={val['accuracy']:.4f} | "
                 f"mF1={val['macro_f1']:.4f} | DMR={val['dangerous_miss']:.4f} | "
                 f"s4u={val['s4_underestimate']:.4f} | reject={val['reject_rate']:.4f}")
        log.info(f"  Recall: [{', '.join(f's{i+1}={r:.3f}' for i, r in enumerate(val['recall_per_class']))}]")

        writer.add_scalar("Loss/train", a(t_loss), epoch)
        writer.add_scalar("Loss/q1", a(t_q1), epoch)
        writer.add_scalar("Loss/q2", a(t_q2), epoch)
        writer.add_scalar("Loss/q3", a(t_q3), epoch)
        writer.add_scalar("Loss/val", val["loss"], epoch)
        writer.add_scalar("M/acc", val["accuracy"], epoch)
        writer.add_scalar("M/mf1", val["macro_f1"], epoch)
        writer.add_scalar("M/DMR", val["dangerous_miss"], epoch)
        for i, r in enumerate(val["recall_per_class"]):
            writer.add_scalar(f"R/s{i+1}", r, epoch)

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
            log.info(f"  ★ BEST: DMR={best_dmr:.4f} | mF1={best_mf1:.4f}")
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

    # τ 搜索
    best_tau, tau_df = search_best_tau(model, val_shards, device, criterion,
                                       batch_size * 2, max_seq_len)
    tau_df.to_csv("tau_search_results.csv", index=False)
    log.info("Saved: tau_search_results.csv")

    log.info(f"\n=== Final Validation (τ={best_tau:.2f}) ===")
    evaluate_on_shards(model, val_shards, device, criterion,
                       batch_size * 2, max_seq_len, verbose=True, tau=best_tau)

    writer.close()
    return model


# ================================================================
# 11. 保存 / 加载
# ================================================================

def save_wrc_model(model, tag="ordinal_v1"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_DIR, f"stage1_ordinal_{tag}_{ts}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "dense_dim": model.dense_dim, "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "feature_cols": STAGE1_FEATURE_COLS,
        "gate_indices": GATE_INDICES,
        "model_type": "cumulative_ordinal_focal",
        "best_tau": float(model.best_tau.item()),
    }, path)
    log.info(f"Saved: {path} (τ={model.best_tau.item():.2f})")
    return path


def load_wrc_model(path, device=None):
    if device is None: device = DEVICE
    d = torch.load(path, map_location=device)
    model = CumulativeOrdinalWDRNet(
        d["dense_dim"], d["hidden_dim"], d["num_layers"]
    ).to(device)
    model.load_state_dict(d["model_state"])
    model.eval()
    if "best_tau" in d:
        model.best_tau.fill_(d["best_tau"])
        log.info(f"Loaded: {path} (τ={d['best_tau']:.2f})")
    else:
        log.info(f"Loaded: {path} (τ=default 0.70)")
    return model


# ================================================================
# 12. 推理接口
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
            denom = std_s[col] if std_s[col] > 1e-8 else 1.0
            x[col] = (x[col] - mean_s[col]) / denom
    scaled = x.astype("float32").replace([np.inf, -np.inf], 0.0).fillna(0.0).values

    n = len(df)
    pred_proba = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    pred_q2_arr = np.zeros(n, dtype=np.float32)
    pred_q3_arr = np.zeros(n, dtype=np.float32)
    pred_reject_arr = np.zeros(n, dtype=bool)
    pred_status_arr = np.ones(n, dtype=np.float32)

    tau = float(model.best_tau.item())

    grouped = df.groupby(["order_id", "day"], sort=False)
    keys = list(grouped.groups.keys())

    for start in tqdm(range(0, len(keys), batch_size),
                      desc="[Ordinal predict]", dynamic_ncols=True, unit="b"):
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
            ri.append(idxs); lens.append(len(feats))

        so = sorted(range(len(lens)), key=lambda i: lens[i], reverse=True)
        seqs = [seqs[i] for i in so]; ri = [ri[i] for i in so]; lens = [lens[i] for i in so]

        sp = pad_sequence(seqs, batch_first=True, padding_value=0.0).to(device)
        lt = torch.LongTensor(lens).to(device)
        gt = sp[:, :, GATE_INDICES]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                lq1, lq2, lq3 = model(sp, lt, gt)
                probs, q1_t, q2_t, q3_t = model.get_cumulative_probs(lq1, lq2, lq3)
                pred_cls_t, should_reject, _, _ = ordinal_quantile_decision(
                    q1_t, q2_t, q3_t, lt, tau=tau,
                )

                probs_np = probs.cpu().numpy()
                q2_np = q2_t.cpu().numpy()
                q3_np = q3_t.cpu().numpy()
                reject_np = should_reject.cpu().numpy()
                cls_np = pred_cls_t.cpu().numpy()

        for b, (L, idxs) in enumerate(zip(lens, ri)):
            pred_proba[idxs] = probs_np[b, :L]
            pred_q2_arr[idxs] = q2_np[b, :L]
            pred_q3_arr[idxs] = q3_np[b, :L]
            pred_reject_arr[idxs] = reject_np[b, :L]
            pred_status_arr[idxs] = cls_np[b, :L].astype(np.float32) + 1.0

    for k in range(NUM_CLASSES):
        df[f"pred_p{k+1}"] = pred_proba[:, k]

    df["pred_status"] = pred_status_arr.astype("float32")

    eps = 1e-10
    safe_p = pred_proba.clip(min=eps)
    df["pred_entropy"] = -(safe_p * np.log(safe_p)).sum(1).astype("float32")
    df["pred_omega"] = (pred_proba[:, 2] + pred_proba[:, 3] * 3).astype("float32")

    df["pred_cong_prob"] = pred_q2_arr.astype("float32")
    df["pred_risk_prob"] = pred_q3_arr.astype("float32")
    df["pred_reject"] = pred_reject_arr

    df.drop(columns=["_row_idx"], inplace=True)
    return df