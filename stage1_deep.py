"""
stage1_deep.py — 层级建模满分版 v3 (Bug Free 生产闭环版)
最终硬核修复:
  - 彻底移除了 forward 内部对 Sub-module 的 inplace 属性强转重赋值，通过在 __init__ 里注册
    并使用 PyTorch 算子自身的 Dtype 匹配机制，100% 保护 Optimizer 的参数梯度链（修复最新硬伤）。
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
    #  s1    s2    s3    s4  (预测)
    [  0,    1,    2,    3  ],   # true s1
    [  1,    0,    2,    3  ],   # true s2
    [  8,    5,    0,    1  ],   # true s3 -> 漏判代价高
    [ 15,   10,    3,    0  ],   # true s4 -> 漏判代价最高
], dtype=np.float32)


# ================================================================
# 1. 双向解耦衰减门 (彻底修复 Optimizer 权重解绑硬伤)
# ================================================================

class BidirectionalDecoupledDecayGate(nn.Module):
    """
    性能优化版：
      1. 双向语义保持不变（前向/反向分别衰减）
      2. 门控 MLP 改为整段序列一次性并行计算
      3. 门控核心仍强制 float32，保障数值稳定
      4. 保持与现有外部接口完全兼容
    """

    def __init__(self, rnn_hidden_dim: int):
        super().__init__()
        gate_input_dim = len(GATE_FEATURE_NAMES)
        single_hidden = rnn_hidden_dim
        bottleneck = max(single_hidden // 2, 16)

        def make_gate(out_dim):
            # ★ 在 __init__ 就固定为 float32，不在 forward 里重复转换
            return nn.Sequential(
                nn.Linear(gate_input_dim, bottleneck),
                nn.Tanh(),
                nn.Linear(bottleneck, out_dim),
                nn.Sigmoid(),
            ).float()

        self.forward_gate = make_gate(single_hidden)
        self.backward_gate = make_gate(single_hidden)

    def forward(
        self,
        gru_out: torch.Tensor,     # (B, T, 2H)
        gate_info: torch.Tensor,   # (B, T, G)
        lengths: torch.Tensor      # (B,)
    ) -> torch.Tensor:
        B, T, H2 = gru_out.shape
        H = H2 // 2
        dev = gru_out.device
        orig_dtype = gru_out.dtype

        # 拆分双向 GRU 输出
        fwd_gru = gru_out[:, :, :H]   # (B, T, H)
        bwd_gru = gru_out[:, :, H:]   # (B, T, H)

        # lengths 放到同一设备
        lengths_dev = lengths.to(dev)

        with torch.cuda.amp.autocast(enabled=False):
            # ----------------------------------------------------
            # 1) 门控输入和 GRU 输出都转 float32
            # ----------------------------------------------------
            gate_f32 = gate_info.float()
            fwd_f32 = fwd_gru.float()
            bwd_f32 = bwd_gru.float()

            # ----------------------------------------------------
            # 2) 一次性并行计算整段序列的 decay
            #    原来: 每个时间步都调一次 MLP
            #    现在: 直接对 (B*T, G) 批量算
            # ----------------------------------------------------
            gate_flat = gate_f32.reshape(B * T, -1)                    # (B*T, G)
            decay_f_all = self.forward_gate(gate_flat).reshape(B, T, H)   # (B, T, H)
            decay_b_all = self.backward_gate(gate_flat).reshape(B, T, H)  # (B, T, H)

            # ----------------------------------------------------
            # 3) 预计算 mask（真实位置=1, padding=0）
            # ----------------------------------------------------
            t_indices = torch.arange(T, device=dev, dtype=torch.long).unsqueeze(0)   # (1, T)
            mask_2d = (lengths_dev.unsqueeze(1) > t_indices).float().unsqueeze(-1)   # (B, T, 1)

            # ----------------------------------------------------
            # 4) 前向衰减：t = 0 → T-1
            # ----------------------------------------------------
            h_fwd = torch.zeros(B, H, device=dev, dtype=torch.float32)
            fwd_steps = [None] * T

            for t in range(T):
                decay = decay_f_all[:, t, :]                         # (B, H)
                h_new = (1.0 - decay) * h_fwd + decay * fwd_f32[:, t, :]
                active_mask = mask_2d[:, t, :]                       # (B, 1)
                h_fwd = active_mask * h_new + (1.0 - active_mask) * h_fwd
                fwd_steps[t] = h_fwd

            fwd_out = torch.stack(fwd_steps, dim=1)                  # (B, T, H)

            # ----------------------------------------------------
            # 5) 反向衰减：t = T-1 → 0
            # ----------------------------------------------------
            h_bwd = torch.zeros(B, H, device=dev, dtype=torch.float32)
            bwd_steps = [None] * T

            for t in range(T - 1, -1, -1):
                decay = decay_b_all[:, t, :]                         # (B, H)
                h_new = (1.0 - decay) * h_bwd + decay * bwd_f32[:, t, :]
                active_mask = mask_2d[:, t, :]                       # (B, 1)
                h_bwd = active_mask * h_new + (1.0 - active_mask) * h_bwd
                bwd_steps[t] = h_bwd

            bwd_out = torch.stack(bwd_steps, dim=1)                  # (B, T, H)

            # ----------------------------------------------------
            # 6) 拼接并转回主干原始精度
            # ----------------------------------------------------
            out = torch.cat([fwd_out, bwd_out], dim=-1).to(orig_dtype)   # (B, T, 2H)

        return out

# ================================================================
# 2. WDR 网络
# ================================================================

class HierarchicalWDRNet(nn.Module):
    def __init__(self, dense_dim: int, hidden_dim: int = 64, num_layers: int = 2):
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

        self.head_cong = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.head_severe = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head_mild = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.register_buffer("cost_matrix", torch.tensor(_COST_MATRIX_NP, dtype=torch.float32))

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

        logit_cong   = self.head_cong(fused).squeeze(-1)
        logit_severe = self.head_severe(fused).squeeze(-1)
        logit_mild   = self.head_mild(fused).squeeze(-1)

        return logit_cong, logit_severe, logit_mild

    def get_hierarchical_probs(self, logit_cong, logit_severe, logit_mild):
        p_cong   = torch.sigmoid(logit_cong)
        p_severe = torch.sigmoid(logit_severe)
        p_mild   = torch.sigmoid(logit_mild)

        p4 = p_cong * p_severe
        p3 = p_cong * (1.0 - p_severe)
        p2 = (1.0 - p_cong) * p_mild
        p1 = (1.0 - p_cong) * (1.0 - p_mild)

        probs = torch.stack([p1, p2, p3, p4], dim=-1)
        return probs, p_cong, p4

    def cost_matrix_decision(self, probs, lengths, reject_entropy_threshold=1.2):
        B, T, _ = probs.shape
        dev = probs.device

        C = self.cost_matrix.to(dtype=probs.dtype, device=dev)
        expected_cost = torch.einsum("btk,kj->btj", probs, C)

        t_indices = torch.arange(T, device=dev).unsqueeze(0)
        mask = (lengths.unsqueeze(1) > t_indices).float().unsqueeze(-1)

        expected_cost = expected_cost * mask + (1.0 - mask) * 1e9
        pred_cls = expected_cost.argmin(dim=-1)

        eps = 1e-10
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
        entropy = entropy * mask.squeeze(-1)

        should_reject = (entropy > reject_entropy_threshold) & (mask.squeeze(-1) > 0)

        return pred_cls, should_reject, entropy


# ================================================================
# 3. 层级损失函数
# ================================================================

class HierarchicalSafetyLoss(nn.Module):
    def __init__(self, pw_cong=8.0, pw_severe=3.0, pw_mild=2.0, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index

        # ★ 不在构造时传 pos_weight，改用 register_buffer
        self.bce_cong   = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_severe = nn.BCEWithLogitsLoss(reduction="mean")
        self.bce_mild   = nn.BCEWithLogitsLoss(reduction="mean")

        # ★ 注册为 buffer，model.to(device) 时自动迁移
        self.register_buffer("pw_cong",   torch.tensor([pw_cong],   dtype=torch.float32))
        self.register_buffer("pw_severe", torch.tensor([pw_severe], dtype=torch.float32))
        self.register_buffer("pw_mild",   torch.tensor([pw_mild],   dtype=torch.float32))

    def forward(self, logit_cong, logit_severe, logit_mild, targets):
        y_flat = targets.reshape(-1)
        valid  = y_flat != self.ignore_index

        lc = logit_cong.reshape(-1)[valid]
        ls = logit_severe.reshape(-1)[valid]
        lm = logit_mild.reshape(-1)[valid]
        y  = y_flat[valid]

        dev = lc.device
        dt  = lc.dtype

        if len(y) == 0:
            zero = torch.tensor(0.0, device=dev, dtype=dt)
            return zero, zero, zero, zero

        # Level 1: 拥堵二分类
        target_cong = (y >= 2).float()
        loss_cong = F.binary_cross_entropy_with_logits(
            lc, target_cong,
            pos_weight=self.pw_cong,   # ★ 已在正确设备上
            reduction="mean"
        )

        # Level 2a: 极端拥堵
        cong_mask = y >= 2
        if cong_mask.sum() > 0:
            loss_severe = F.binary_cross_entropy_with_logits(
                ls[cong_mask],
                (y[cong_mask] == 3).float(),
                pos_weight=self.pw_severe,
                reduction="mean"
            )
        else:
            loss_severe = torch.tensor(0.0, device=dev, dtype=dt)

        # Level 2b: 缓行
        non_cong_mask = y < 2
        if non_cong_mask.sum() > 0:
            loss_mild = F.binary_cross_entropy_with_logits(
                lm[non_cong_mask],
                (y[non_cong_mask] == 1).float(),
                pos_weight=self.pw_mild,
                reduction="mean"
            )
        else:
            loss_mild = torch.tensor(0.0, device=dev, dtype=dt)

        total = loss_cong + loss_severe + loss_mild
        return total, loss_cong, loss_severe, loss_mild


# ================================================================
# 4. 类别分布估计
# ================================================================

def estimate_hierarchical_pos_weights(train_shards, max_shards=5):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for info in train_shards[:max_shards]:
        shard = torch.load(info["path"], map_location="cpu")
        for y in shard["y_list"]:
            y_arr = np.asarray(y)
            y_arr = y_arr[(y_arr >= 0) & (y_arr < NUM_CLASSES)]
            counts += np.bincount(y_arr, minlength=NUM_CLASSES)
        del shard; gc.collect()

    counts = np.maximum(counts, 1)
    n_cong     = counts[2] + counts[3]
    n_non_cong = counts[0] + counts[1]

    pw_cong = float(n_non_cong / max(n_cong, 1))
    pw_cong = float(np.clip(pw_cong, 2.0, 30.0))

    pw_severe = float(counts[2] / max(counts[3], 1))
    pw_severe = float(np.clip(pw_severe, 1.0, 10.0))

    pw_mild = float(counts[0] / max(counts[1], 1))
    pw_mild = float(np.clip(pw_mild, 1.0, 10.0))

    log.info(f"Class counts: {counts.tolist()}")
    log.info(f"pos_weights → cong(非拥堵/拥堵)={pw_cong:.2f}  severe(s3/s4)={pw_severe:.2f}  mild(s1/s2)={pw_mild:.2f}")
    return pw_cong, pw_severe, pw_mild


# ================================================================
# 5. 工具函数
# ================================================================

def log_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        p = torch.cuda.max_memory_allocated() / 1024**3
        log.info(f"{prefix}[GPU] alloc={a:.2f}GB reserved={r:.2f}GB peak={p:.2f}GB")


def save_training_checkpoint(model, optimizer, scheduler, scaler,
                             epoch, best_dmr, best_macro_f1, ckpt_path):
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "best_dmr": best_dmr, "best_macro_f1": best_macro_f1,
        "dense_dim": model.dense_dim, "hidden_dim": model.hidden_dim, "num_layers": model.num_layers,
    }, ckpt_path)


def load_training_checkpoint(ckpt_path, device):
    return torch.load(ckpt_path, map_location=device)


# ================================================================
# 6. 验证流
# ================================================================

def evaluate_on_shards(model, shard_infos, device, criterion,
                       batch_size=256, max_seq_len=100, verbose=False):
    model.eval()
    all_preds, all_labels, all_rejects = [], [], []
    total_loss, n_batches = 0.0, 0

    pf = ShardPrefetcher(shard_infos)
    while True:
        _, shard_obj = pf.next()
        if shard_obj is None:
            break

        ds = SingleShardSequenceDataset(shard_obj, max_seq_len=max_seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == "cuda"))

        with torch.no_grad():
            for X_b, y_b, lens_b in loader:
                X_b   = X_b.to(device, non_blocking=True)
                y_b   = y_b.to(device, non_blocking=True)
                lens_b = lens_b.to(device)
                gate_b = X_b[:, :, GATE_INDICES]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lc, ls, lm = model(X_b, lens_b, gate_b)
                    loss, _, _, _ = criterion(lc, ls, lm, y_b)

                total_loss += float(loss.item())
                n_batches  += 1

                probs, _, _ = model.get_hierarchical_probs(lc, ls, lm)
                pred_cls, should_reject, _ = model.cost_matrix_decision(probs, lens_b)

                for b in range(len(lens_b)):
                    L = lens_b[b].item()
                    all_preds.extend(pred_cls[b, :L].cpu().numpy())
                    all_labels.extend(y_b[b, :L].cpu().numpy())
                    all_rejects.extend(should_reject[b, :L].cpu().numpy())

        del ds, loader, shard_obj; gc.collect()
    pf.close()

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    reject  = np.array(all_rejects)
    mask    = y_true >= 0
    y_true, y_pred, reject = y_true[mask], y_pred[mask], reject[mask]

    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred, average="macro")
    high = y_true >= 2
    dmr  = float((y_pred < 2)[high].sum() / max(high.sum(), 1))
    s4a  = y_true == 3
    s4u  = float((y_pred < 3)[s4a].sum() / max(s4a.sum(), 1))
    rr   = float(reject.mean())
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    rec  = [cm[i, i] / max(cm[i].sum(), 1) for i in range(4)]

    if verbose:
        ls_n = [f"s{k}" for k in STATUS_CLASSES]
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in ls_n], columns=[f"pred_{l}" for l in ls_n])
        log.info(f"Confusion Matrix:\n{cm_df.to_string()}")
        log.info(f"\n{classification_report(y_true, y_pred, target_names=ls_n, digits=4)}")
        log.info(f"  DMR={dmr:.4f} | s4_under={s4u:.4f} | reject_rate={rr:.4f}")

    return {"loss": total_loss / max(n_batches, 1), "accuracy": acc, "macro_f1": mf1,
            "dangerous_miss": dmr, "s4_underestimate": s4u, "recall_per_class": rec, "reject_rate": rr}


# ================================================================
# 7. 训练主引擎
# ================================================================

def train_wrc_from_shards(
    hidden_dim=WRC_HIDDEN_DIM, num_layers=WRC_NUM_LAYERS,
    batch_size=WRC_BATCH_SIZE, epochs=WRC_EPOCHS,
    lr=WRC_LR, max_seq_len=WRC_MAX_SEQ_LEN, device=None, resume=True,
):
    if device is None: device = DEVICE

    log.info("=" * 70)
    log.info("Hierarchical WDR v3: Param Re-binding Bug Fixed")
    log.info("=" * 70)

    _, _, feature_cols = load_stats()
    train_shards, val_shards, _ = split_train_val_shards(val_ratio=0.15, seed=42)

    steps_per_epoch = sum(math.ceil(s["n_orders"] / batch_size) for s in train_shards)
    total_steps = epochs * steps_per_epoch

    pw_cong, pw_severe, pw_mild = estimate_hierarchical_pos_weights(train_shards, min(5, len(train_shards)))

    model = HierarchicalWDRNet(len(feature_cols), hidden_dim, num_layers).to(device)
    criterion = HierarchicalSafetyLoss(pw_cong=pw_cong, pw_severe=pw_severe, pw_mild=pw_mild).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"hier_wdr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    latest_ckpt, best_ckpt = os.path.join(CHECKPOINT_DIR, "hier_latest.pt"), os.path.join(CHECKPOINT_DIR, "hier_best.pt")

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

    for epoch in range(start_epoch, epochs):
        model.train()
        t_loss = t_cong = t_severe = t_mild = 0.0
        nb = 0

        shards = train_shards.copy()
        np.random.RandomState(epoch + 42).shuffle(shards)
        pf = ShardPrefetcher(shards); sc = 0

        while True:
            _, shard_obj = pf.next()
            if shard_obj is None: break
            sc += 1

            ds = SingleShardSequenceDataset(shard_obj, max_seq_len)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == "cuda"))
            pbar = tqdm(loader, desc=f"E{epoch+1} S{sc:>3}/{len(shards)}", leave=False, dynamic_ncols=True, unit="b")

            for bi, (X, y, lens) in enumerate(pbar):
                X, y, lens = X.to(device, non_blocking=True), y.to(device, non_blocking=True), lens.to(device)
                gate = X[:, :, GATE_INDICES]

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lc, ls, lm = model(X, lens, gate)
                    loss, l_cong, l_sev, l_mild = criterion(lc, ls, lm, y)

                if torch.isnan(loss) or torch.isinf(loss): continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if global_step < total_steps: scheduler.step()
                global_step += 1

                t_loss += float(loss.item()); t_cong += float(l_cong.item()); t_severe += float(l_sev.item()); t_mild += float(l_mild.item())
                nb += 1

                if device.type == "cuda" and bi % 30 == 0:
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix(loss=f"{t_loss/nb:.4f}", gpu=f"{alloc:.1f}G", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

            del ds, loader, shard_obj
            if sc % 15 == 0:
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()

        pf.close()
        a = lambda x: x / max(nb, 1)

        val = evaluate_on_shards(model, val_shards, device, criterion, batch_size * 2, max_seq_len, verbose=False)
        clr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Loss/val", val["loss"], epoch)
        writer.add_scalar("M/mf1", val["macro_f1"], epoch)
        writer.add_scalar("M/DMR", val["dangerous_miss"], epoch)

        log.info(f"Epoch {epoch+1:>2}/{epochs} | loss={a(t_loss):.4f} | val={val['loss']:.4f} | acc={val['accuracy']:.4f} | mf1={val['macro_f1']:.4f} | DMR={val['dangerous_miss']:.4f}")

        save_training_checkpoint(model, optimizer, scheduler, scaler, epoch, best_dmr, best_mf1, latest_ckpt)

        imp = (val["dangerous_miss"] < best_dmr or (abs(val["dangerous_miss"] - best_dmr) < 1e-8 and val["macro_f1"] > best_mf1))
        if imp:
            best_dmr, best_mf1 = val["dangerous_miss"], val["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_training_checkpoint(model, optimizer, scheduler, scaler, epoch, best_dmr, best_mf1, best_ckpt)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience: break

    if best_state: model.load_state_dict(best_state)
    evaluate_on_shards(model, val_shards, device, criterion, batch_size * 2, max_seq_len, True)
    writer.close()
    return model


# ================================================================
# 8. 模型固化接口
# ================================================================

def save_wrc_model(model, tag="hier_v3"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_DIR, f"stage1_hier_{tag}_{ts}.pt")
    torch.save({
        "model_state": model.state_dict(), "dense_dim": model.dense_dim, "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers, "feature_cols": STAGE1_FEATURE_COLS, "gate_indices": GATE_INDICES,
        "model_type": "hierarchical_v3",
    }, path)
    log.info(f"Hierarchical WDR v3 saved: {path}")
    return path


def load_wrc_model(path, device=None):
    if device is None: device = DEVICE
    d = torch.load(path, map_location=device)
    model = HierarchicalWDRNet(d["dense_dim"], d["hidden_dim"], d["num_layers"]).to(device)
    model.load_state_dict(d["model_state"])
    model.eval()
    return model


# ================================================================
# 9. 生产环境通用推理接口
# ================================================================

def predict_proba_wrc(model, df, max_seq_len=WRC_MAX_SEQ_LEN, batch_size=512, device=None):
    if device is None: device = DEVICE
    model.eval(); model.to(device)

    mean_dict, std_dict, feature_cols = load_stats()
    mean_s = pd.Series(mean_dict); std_s  = pd.Series(std_dict)

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

    pred_proba, pred_cong, pred_p4 = np.zeros((len(df), NUM_CLASSES), dtype=np.float32), np.zeros(len(df), dtype=np.float32), np.zeros(len(df), dtype=np.float32)
    pred_reject = np.zeros(len(df), dtype=bool)

    grouped = df.groupby(["order_id", "day"], sort=False)
    keys    = list(grouped.groups.keys())

    for start in tqdm(range(0, len(keys), batch_size), desc="[Hier predict]", dynamic_ncols=True, unit="b"):
        bk = keys[start:start + batch_size]
        seqs, ri, lens = [], [], []

        for key in bk:
            grp = grouped.get_group(key); idxs = grp["_row_idx"].values; feats = scaled[idxs]
            if len(feats) > max_seq_len: feats, idxs = feats[:max_seq_len], idxs[:max_seq_len]
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            seqs.append(torch.FloatTensor(feats)); ri.append(idxs); lens.append(len(feats))

        so  = sorted(range(len(lens)), key=lambda i: lens[i], reverse=True)
        seqs = [seqs[i] for i in so]; ri = [ri[i] for i in so]; lens = [lens[i] for i in so]

        sp = pad_sequence(seqs, batch_first=True, padding_value=0.0).to(device)
        lt = torch.LongTensor(lens).to(device)
        gt = sp[:, :, GATE_INDICES]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                lc, ls, lm = model(sp, lt, gt)

                probs, p_cong_t, p4_t = model.get_hierarchical_probs(lc, ls, lm)
                _, should_reject, _   = model.cost_matrix_decision(probs, lt)

                probs_np, cong_np, p4_np, reject_np = probs.cpu().numpy(), p_cong_t.cpu().numpy(), p4_t.cpu().numpy(), should_reject.cpu().numpy()

        for b, (L, idxs) in enumerate(zip(lens, ri)):
            pred_proba[idxs]  = probs_np[b, :L]
            pred_cong[idxs]   = cong_np[b, :L]
            pred_p4[idxs]     = p4_np[b, :L]
            pred_reject[idxs] = reject_np[b, :L]

    for k in range(NUM_CLASSES): df[f"pred_p{k+1}"] = pred_proba[:, k]

    df["pred_status"] = (pred_proba * np.array([[1, 2, 3, 4]])).sum(1).astype("float32")
    df["pred_entropy"] = -(pred_proba * np.log(pred_proba + 1e-10)).sum(1).astype("float32")
    df["pred_omega"]     = (pred_proba[:, 2] + pred_proba[:, 3] * 3).astype("float32")
    df["pred_cong_prob"] = pred_cong.astype("float32")

    df["pred_risk_prob"] = pred_p4.astype("float32")
    df["pred_reject"] = pred_reject

    df.drop(columns=["_row_idx"], inplace=True)
    return df