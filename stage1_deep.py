"""
stage1_deep.py — 级联架构 (Cascade) 版

设计:
  - 第一级: ng_head 二分类 (s1 vs non-s1)
  - 第二级: ordinal head 在 non-s1 样本上做三分类 (s2/s3/s4)
  - 推理时联合输出四类概率

损失:
  L = L_ng (全样本)
    + L_ord (仅 non-s1)
    + adaptive_weight(L_cls, L_bnd, L_align, L_sharp)  # 也仅 non-s1
    + boundary_weight * L_boundary (仅 s2/s3)
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
    MAX_TRAIN_SHARDS, MAX_VAL_SHARDS, MAX_EVAL_BATCHES,  # ★ 新增这三行
)
from feature_eng import STAGE1_FEATURE_COLS
from stage1_dataset import (
    split_train_val_shards, load_stats,
    SingleShardSequenceDataset, collate_fn, ShardPrefetcher,
)
from logger import get_logger
from feature_eng import STAGE1_FEATURE_COLS
# ★ 导入轻量 transform
from stage1_feature_transform import transform_stage1_feature_array

log = get_logger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GATE_FEATURE_NAMES = [
    "pos_ratio", "cum_travel_time",
    "sin_arr_slice", "cos_arr_slice",
    "downstream_cross_time",
]

# ★ 在文件顶部定义特征类型
RAW_ID_FEATURES = frozenset({
    "link_id",
    "slice_id",
    "arrival_slice_est",
    "link_current_status",
})
PERIODIC_FEATURES = frozenset({
    "sin_slice", "cos_slice",
    "sin_arr_slice", "cos_arr_slice",
})


GATE_INDICES = [STAGE1_FEATURE_COLS.index(n) for n in GATE_FEATURE_NAMES]



PERIODIC_FEATURES = frozenset({"sin_slice", "cos_slice", "sin_arr_slice", "cos_arr_slice"})
# ★ 替换原有的 PERIODIC_FEATURES 定义
from stage1_feature_schema import (
    RAW_ID_FEATURES,
    ORDINAL_RAW_FEATURES,
    PERIODIC_FEATURES,
    NON_STANDARDIZE_FEATURES,
    ID_CLIP_RANGES,
)
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Warm-up 配置
WARMUP_EPOCHS = 2
# 边界损失权重 (固定，不参与自适应)
BOUNDARY_WEIGHT = 0.15

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
# 3. 级联网络 (Cascade)
# ================================================================

class CascadeWDRNet(nn.Module):
    """
    级联架构:
      - ng_head: 二分类 s1 vs non-s1
      - ordinal_head: 在 non-s1 上做三分类 (s2/s3/s4)
    """
    def __init__(self, dense_dim, hidden_dim=64, num_layers=2, min_margin=0.3, status_emb_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dense_dim  = dense_dim
        # ★ 获取 link_current_status 在特征列中的位置
        self.status_idx = STAGE1_FEATURE_COLS.index("link_current_status")
        self.slice_idx = STAGE1_FEATURE_COLS.index("slice_id")
        self.arrival_idx = STAGE1_FEATURE_COLS.index("arrival_slice_est")

        self.status_emb_dim = status_emb_dim

        # ★ 定义 Embedding 层
        self.status_emb = nn.Embedding(5, 8)          # 0~4
        self.slice_emb = nn.Embedding(288, 8)         # 0~287
        self.arrival_emb = nn.Embedding(288, 8)       # 0~287
        # 计算总 Embedding 维度
        total_emb_dim = 8 + 8 + 8  # status + slice + arrival

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

        fusion_dim = hidden_dim * 4 + total_emb_dim
        adapter_dim = hidden_dim
        # 更新 Adapter 的输入维度
        self.ordinal_adapter = nn.Sequential(
            nn.Linear(fusion_dim, adapter_dim), nn.ReLU(), nn.Dropout(0.1),
        )
        self.cls_adapter = nn.Sequential(
            nn.Linear(fusion_dim, adapter_dim), nn.ReLU(), nn.Dropout(0.1),
        )
        # 共享的 adapter（可复用于各任务）
        self.shared_adapter = nn.Sequential(
            nn.Linear(fusion_dim, adapter_dim), nn.ReLU(), nn.Dropout(0.1),
        )
        # ng head (二分类)
        self.ng_head = nn.Linear(adapter_dim, 1)
        # ordinal head (三分类，但保留三个累计logit输出)
        self.ordinal_head = MonotonicOrdinalHead(adapter_dim, adapter_dim // 2, min_margin)
        # 辅助分类头 (仅用于辅助训练)
        self.cls_head = nn.Linear(adapter_dim, NUM_CLASSES)
        # boundary head (用于辅助)
        self.boundary_head = nn.Linear(adapter_dim // 2, 1)  # 需额外adapter，简化：直接接 ordinal 的 shared 输出？

        # 为了简化，我们复用 ordinal_head 的 shared 作为 boundary 特征
        # 但为了独立，我们再建一个小 adapter
        self.boundary_adapter = nn.Sequential(
            nn.Linear(fusion_dim, adapter_dim // 2), nn.ReLU(), nn.Dropout(0.1),
        )
        self.boundary_head = nn.Linear(adapter_dim // 2, 1)

        self.register_buffer("best_tau", torch.tensor(0.55))
        self.register_buffer("temperature", torch.tensor(1.0))

    def forward(self, x, lengths, gate_info):
        B, T, D = x.shape

        # ★ 提取三列离散特征（并做安全 clamp）
        status = x[:, :, self.status_idx].long()
        slice_id = x[:, :, self.slice_idx].long()
        arrival_id = x[:, :, self.arrival_idx].long()

        # ★ 安全钳位：防止超出词表范围
        status = status.clamp(0, 4)  # Embedding 词表大小 = 5 (0~4)
        slice_id = slice_id.clamp(0, 287)  # Embedding 词表大小 = 288 (0~287)
        arrival_id = arrival_id.clamp(0, 287)

        status_emb = self.status_emb(status)
        slice_emb = self.slice_emb(slice_id)
        arrival_emb = self.arrival_emb(arrival_id)

        # ★ 原始 x 保持不变，直接用于 wide/deep/gru
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.gru(packed)
        gru_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        decay_out = self.decay_gate(gru_out, gate_info, lengths)

        x_flat = x.reshape(B * T, -1)  # 维度 D
        wide_out = self.wide(x_flat).reshape(B, T, self.hidden_dim)
        deep_out = self.deep(x_flat).reshape(B, T, self.hidden_dim)

        # ★ 融合：wide + deep + decay + 三个 Embedding
        fused = torch.cat([wide_out, deep_out, decay_out, status_emb, slice_emb, arrival_emb], dim=-1)

        # 共享特征
        h_shared = self.shared_adapter(fused)

        # ng logit
        ng_logit = self.ng_head(h_shared).squeeze(-1)

        # ordinal logits
        lq1, lq2, lq3 = self.ordinal_head(h_shared)

        # cls logits
        cls_logits = self.cls_head(h_shared)

        # boundary logit
        h_bnd = self.boundary_adapter(fused)
        bnd_logit = self.boundary_head(h_bnd).squeeze(-1)

        return lq1, lq2, lq3, cls_logits, bnd_logit, ng_logit

    def get_cascade_probs(self, lq1, lq2, lq3, ng_logit, temperature=None):
        """
        级联概率输出：
        - ng_logit -> P(non-s1)
        - lq1,lq2,lq3 -> P(s2|non-s1), P(s3|non-s1), P(s4|non-s1)
        """
        if temperature is None:
            temperature = float(self.temperature.item())

        p_non_s1 = torch.sigmoid(ng_logit)

        # 温度缩放 ordinal logits
        lq1_s = lq1 / temperature
        lq2_s = lq2 / temperature
        lq3_s = lq3 / temperature
        q1 = torch.sigmoid(lq1_s)
        q2 = torch.sigmoid(lq2_s)
        q3 = torch.sigmoid(lq3_s)

        p4_cond = F.relu(q3)
        p3_cond = F.relu(q2 - q3)
        p2_cond = F.relu(q1 - q2)
        # p1_cond = 1 - q1 (但s1由ng处理，我们不用)

        cond_probs = torch.stack([p2_cond, p3_cond, p4_cond], dim=-1)
        cond_probs = cond_probs / cond_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        p_s1 = 1.0 - p_non_s1
        p_s2 = p_non_s1 * cond_probs[:, :, 0]
        p_s3 = p_non_s1 * cond_probs[:, :, 1]
        p_s4 = p_non_s1 * cond_probs[:, :, 2]

        probs = torch.stack([p_s1, p_s2, p_s3, p_s4], dim=-1)
        return probs, p_non_s1, cond_probs

    def get_ordinal_probs(self, lq1, lq2, lq3, temperature=None):
        """兼容旧接口，但此版本不建议使用，应使用 get_cascade_probs"""
        if temperature is None:
            temperature = float(self.temperature.item())
        lq1_s = lq1 / temperature
        lq2_s = lq2 / temperature
        lq3_s = lq3 / temperature
        q1 = torch.sigmoid(lq1_s)
        q2 = torch.sigmoid(lq2_s)
        q3 = torch.sigmoid(lq3_s)
        p4 = F.relu(q3)
        p3 = F.relu(q2 - q3)
        p2 = F.relu(q1 - q2)
        p1 = F.relu(1.0 - q1)
        probs = torch.stack([p1, p2, p3, p4], dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs, q1, q2, q3


# ================================================================
# 4. 自适应辅助权重 (Uncertainty Weighting)
# ================================================================

class AdaptiveAuxWeight(nn.Module):
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
# 5. 级联损失函数
# ================================================================

class CascadeLoss(nn.Module):
    """
    级联损失:
      - L_ng: 全样本二分类 (s1 vs non-s1)
      - L_ord, L_cls, L_bnd, L_align, L_sharp: 仅对 non-s1 样本
      - L_boundary: 仅对 s2/s3 样本 (固定权重)
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

    def _focal_bce(self, logits, targets, pw, gamma, weights=None):
        pw = pw.to(device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
        if weights is not None:
            bce = bce * weights
        if gamma <= 0:
            return bce.mean()
        p  = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        loss = (1 - pt).pow(gamma) * bce
        return loss.mean()

    def _focal_ce(self, logits, targets, alpha, gamma, class_weights=None):
        """
        class_weights: 类别权重张量 (num_classes,)，用于对每个类别的损失加权。
        """
        alpha = alpha.to(device=logits.device, dtype=logits.dtype)
        ce = F.cross_entropy(logits, targets, reduction="none")
        if class_weights is not None:
            # ★ 关键修复：按 targets 索引类别权重
            ce = ce * class_weights[targets]
        if gamma <= 0:
            return (alpha[targets] * ce).mean()
        probs = F.softmax(logits, dim=-1)
        pt    = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        loss = (alpha[targets] * (1 - pt).pow(gamma) * ce)
        return loss.mean()

    def _compute_class_weights(self, y, num_classes=4):
        counts = torch.bincount(y, minlength=num_classes).float() + 1e-6
        weights = 1.0 / counts
        weights = weights / weights.sum() * num_classes
        return weights

    def forward(self, lq1, lq2, lq3, cls_logits, bnd_logit, ng_logit, targets):
        y_flat = targets.reshape(-1)
        valid = y_flat != self.ignore_index

        dev = y_flat.device
        zero = torch.tensor(0.0, device=dev)

        if valid.sum() == 0:
            return zero, zero, zero, zero, zero, zero, zero, zero, zero, zero

        y = y_flat[valid].long()

        ngf = ng_logit.reshape(-1)[valid]
        q1f = lq1.reshape(-1)[valid]
        q2f = lq2.reshape(-1)[valid]
        q3f = lq3.reshape(-1)[valid]
        clf = cls_logits.reshape(-1, NUM_CLASSES)[valid]
        bndf = bnd_logit.reshape(-1)[valid]

        # ============================================================
        # 0. L_seq_cls：逐时间步分类辅助，覆盖所有有效 link
        # ============================================================
        cls_logits_flat = cls_logits.reshape(-1, NUM_CLASSES)
        targets_flat = targets.reshape(-1).long()

        l_seq_cls = F.cross_entropy(
            cls_logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            reduction="mean",
        )

        # ============================================================
        # 1. L_ng：s1 vs non-s1，全样本
        # ============================================================
        ng_target = (y >= 1).float()

        n_neg = (y == 0).sum().float()
        n_pos = (y >= 1).sum().float()
        pos_weight = (n_neg / torch.clamp(n_pos, min=1.0)).clamp(1.0, 20.0)

        l_ng = F.binary_cross_entropy_with_logits(
            ngf,
            ng_target,
            pos_weight=pos_weight.to(device=dev),
            reduction="mean",
        )

        # ============================================================
        # 2. L_under：underestimation-aware loss，全样本
        #    使用 ng + q2 + q3 形成期望严重度
        # ============================================================
        p_ng = torch.sigmoid(ngf)
        p_q2 = torch.sigmoid(q2f)
        p_q3 = torch.sigmoid(q3f)

        # 这里不使用 q1，因为你当前 q1 只在 non-s1 上 target=1，
        # 真正的 s1/non-s1 由 ng_logit 负责。
        p1 = 1.0 - p_ng
        p2 = p_ng * (1.0 - p_q2)
        p3 = p_ng * p_q2 * (1.0 - p_q3)
        p4 = p_ng * p_q2 * p_q3

        pred_sev = 0.0 * p1 + 1.0 * p2 + 2.0 * p3 + 3.0 * p4
        true_sev = y.float()

        under = torch.relu(true_sev - pred_sev)

        under_class_weights = torch.tensor(
            [0.0, 0.4, 1.0, 2.0],
            device=dev,
            dtype=pred_sev.dtype,
        )

        under_w = under_class_weights[y].clamp_min(0.0)

        if under_w.sum() > 0:
            l_under = (under_w * under.pow(2)).sum() / under_w.sum().clamp_min(1.0)
        else:
            l_under = zero

        # ============================================================
        # 后续损失只对 non-s1 样本
        # ============================================================
        non_s1_mask = y >= 1

        if non_s1_mask.sum() == 0:
            return l_ng, zero, zero, zero, zero, zero, zero, l_seq_cls, zero, l_under

        y_ns = y[non_s1_mask]
        q1f_ns = q1f[non_s1_mask]
        q2f_ns = q2f[non_s1_mask]
        q3f_ns = q3f[non_s1_mask]
        clf_ns = clf[non_s1_mask]
        bndf_ns = bndf[non_s1_mask]

        # ============================================================
        # 3. Optional S4 recall auxiliary
        # ============================================================
        s4_mask_ns = y_ns == 3

        if s4_mask_ns.sum() > 0:
            l_s4 = F.binary_cross_entropy_with_logits(
                q3f_ns[s4_mask_ns],
                torch.ones_like(q3f_ns[s4_mask_ns]),
                reduction="mean",
            )
        else:
            l_s4 = zero

        class_weights = self._compute_class_weights(y_ns, num_classes=NUM_CLASSES)

        # ============================================================
        # 4. L_ord
        # ============================================================
        l_q1 = self._focal_bce(
            q1f_ns,
            torch.ones_like(y_ns, dtype=torch.float),
            self.pw_q1,
            self.gamma_q1,
        )

        l_q2 = self._focal_bce(
            q2f_ns,
            (y_ns >= 2).float(),
            self.pw_q2,
            self.gamma_q2,
        )

        l_q3 = self._focal_bce(
            q3f_ns,
            (y_ns >= 3).float(),
            self.pw_q3,
            self.gamma_q3,
        )

        l_ord = l_q1 + l_q2 + l_q3

        # ============================================================
        # 5. L_cls
        # ============================================================
        l_cls = self._focal_ce(
            clf_ns,
            y_ns,
            self.cls_alpha,
            self.gamma_cls,
            class_weights=class_weights,
        )

        # ============================================================
        # 6. L_bnd: s2 vs s3
        # ============================================================
        mid_mask_ns = (y_ns == 1) | (y_ns == 2)

        if mid_mask_ns.sum() > 0:
            bnd_target = (y_ns[mid_mask_ns] == 2).float()

            bce_bnd = F.binary_cross_entropy_with_logits(
                bndf_ns[mid_mask_ns],
                bnd_target,
                reduction="none",
            )

            if self.gamma_bnd > 0:
                p_b = torch.sigmoid(bndf_ns[mid_mask_ns])
                pt_b = p_b * bnd_target + (1.0 - p_b) * (1.0 - bnd_target)
                bce_bnd = (1.0 - pt_b).pow(self.gamma_bnd) * bce_bnd

            l_bnd = bce_bnd.mean()
        else:
            l_bnd = zero

        # ============================================================
        # 7. L_align
        # ============================================================
        p_cls = F.softmax(clf_ns, dim=-1)

        f_cls_1 = p_cls[:, 1] + p_cls[:, 2] + p_cls[:, 3]
        f_cls_2 = p_cls[:, 2] + p_cls[:, 3]
        f_cls_3 = p_cls[:, 3]

        with torch.no_grad():
            q1_detach = torch.sigmoid(q1f_ns)
            q2_detach = torch.sigmoid(q2f_ns)
            q3_detach = torch.sigmoid(q3f_ns)

        l_align = (
            F.mse_loss(f_cls_1, q1_detach)
            + F.mse_loss(f_cls_2, q2_detach)
            + F.mse_loss(f_cls_3, q3_detach)
        ) / 3.0

        # ============================================================
        # 8. L_sharp
        # ============================================================
        if mid_mask_ns.sum() > 0:
            sharp_target = (y_ns[mid_mask_ns] == 2).float()

            l_sharp = F.binary_cross_entropy_with_logits(
                q2f_ns[mid_mask_ns],
                sharp_target,
                reduction="mean",
            )
        else:
            l_sharp = zero

        # ============================================================
        # 9. L_boundary
        # ============================================================
        if mid_mask_ns.sum() > 0:
            q2_vals = torch.sigmoid(q2f_ns[mid_mask_ns])
            q3_vals = torch.sigmoid(q3f_ns[mid_mask_ns])

            diff = q2_vals - q3_vals

            target_diff = torch.where(
                y_ns[mid_mask_ns] == 1,
                torch.tensor(0.4, device=dev, dtype=diff.dtype),
                torch.tensor(-0.2, device=dev, dtype=diff.dtype),
            )

            l_boundary = F.smooth_l1_loss(
                diff,
                target_diff,
                reduction="mean",
            )
        else:
            l_boundary = zero

        return (
            l_ng,
            l_ord,
            l_cls,
            l_bnd,
            l_align,
            l_sharp,
            l_boundary,
            l_seq_cls,
            l_s4,
            l_under,
        )


# ================================================================
# 6. 工具函数
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
        "temperature": float(model.temperature.item()),
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
    return pw_q1, pw_q2, pw_q3, cls_alpha, counts



##工具函数patch
def _to_zero_based_labels(y):
    y = np.asarray(y).astype(np.int64)
    if y.size == 0:
        return y
    if y.min() >= 1 and y.max() <= 4:
        y = y - 1
    return y

def _fast_confusion_matrix(y_true, y_pred, num_classes=4):
    y_true = _to_zero_based_labels(y_true)
    y_pred = _to_zero_based_labels(y_pred)
    mask = (
        (y_true >= 0) & (y_true < num_classes) &
        (y_pred >= 0) & (y_pred < num_classes)
    )
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    idx = y_true * num_classes + y_pred
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).astype(np.int64)

def _metrics_from_cm(cm):
    cm = cm.astype(np.float64)
    total = cm.sum()
    diag = np.diag(cm)
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)

    acc = float(diag.sum() / max(total, 1.0))
    precision = diag / np.maximum(col_sum, 1.0)
    recall = diag / np.maximum(row_sum, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    mf1 = float(np.mean(f1))

    dangerous_total = row_sum[2] + row_sum[3]
    dangerous_miss = cm[2, 0] + cm[2, 1] + cm[3, 0] + cm[3, 1]
    dmr = float(dangerous_miss / max(dangerous_total, 1.0))

    s4_total = row_sum[3]
    s4_under = cm[3, 0] + cm[3, 1] + cm[3, 2]
    s4u = float(s4_under / max(s4_total, 1.0))

    return {
        "accuracy": acc,
        "macro_f1": mf1,
        "DMR": dmr,
        "s4_under": s4u,
        "recall": recall.astype(float),
        "f1": f1.astype(float),
        "s2_f1": float(f1[1]),
        "s3_f1": float(f1[2]),
        "f1_s23": float((f1[1] + f1[2]) / 2.0),
    }


# ================================================================
# 7. 验证
# ================================================================

def evaluate_on_shards(model, shard_infos, device, criterion,
                       batch_size=256, max_seq_len=100, verbose=False, tau=None,
                       temperature=None, max_eval_batches=None):
    model.eval()
    if tau is None:
        tau = float(model.best_tau.item())
    if temperature is None:
        temperature = float(model.temperature.item())

    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total_loss = 0.0
    n_batches = 0
    total_valid = 0
    total_reject = 0

    pf = ShardPrefetcher(shard_infos)
    while True:
        _, shard_obj = pf.next()
        if shard_obj is None:
            break

        ds = SingleShardSequenceDataset(shard_obj, max_seq_len=max_seq_len)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )

        with torch.no_grad():
            for X_b, y_b, lens_b in loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                lens_b = lens_b.to(device)
                gate_b = X_b[:, :, GATE_INDICES]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lq1, lq2, lq3, cls_logits, bnd_logit, ng_logit = model(X_b, lens_b, gate_b)

                    if criterion is not None:
                        losses = criterion(
                            lq1, lq2, lq3,
                            cls_logits,
                            bnd_logit,
                            ng_logit,
                            y_b,
                        )

                        l_ng = losses[0]
                        l_ord = losses[1]

                        total_loss += float(l_ng.item()) + float(l_ord.item())

                probs, _, _ = model.get_cascade_probs(lq1, lq2, lq3, ng_logit, temperature=temperature)
                cdf = probs.cumsum(dim=-1)
                pred = (cdf >= tau).float().argmax(dim=-1)

                # reject
                eps = 1e-10
                safe_p = probs.clamp_min(eps)
                entropy = -(safe_p * torch.log(safe_p)).sum(dim=-1)
                B, T = lq1.shape
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                mask = lens_b.unsqueeze(1) > t_idx
                entropy = torch.where(mask, entropy, torch.zeros_like(entropy))
                should_reject = (entropy > 1.2) & mask

                for b in range(len(lens_b)):
                    L = lens_b[b].item()
                    y_true = y_b[b, :L].cpu().numpy()
                    y_pred = pred[b, :L].cpu().numpy()
                    valid_mask = y_true >= 0
                    y_true = y_true[valid_mask]
                    y_pred = y_pred[valid_mask]
                    if len(y_true) > 0:
                        cm_total += _fast_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)
                        total_valid += len(y_true)
                        total_reject += should_reject[b, :L].cpu().numpy()[valid_mask].sum()

                n_batches += 1
                if max_eval_batches is not None and n_batches >= max_eval_batches:
                    break

        del ds, loader, shard_obj
        gc.collect()
    pf.close()

    # 从混淆矩阵计算指标
    metrics = _metrics_from_cm(cm_total)
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["reject_rate"] = total_reject / max(total_valid, 1)

    if verbose:
        cm_df = pd.DataFrame(
            cm_total,
            index=[f"true_{l}" for l in STATUS_CLASSES],
            columns=[f"pred_{l}" for l in STATUS_CLASSES],
        )
        rec = metrics["recall"]
        log.info(f"\n--- Cascade (τ={tau:.2f}, T={temperature:.2f}) ---")
        log.info(f"CM:\n{cm_df.to_string()}")
        log.info(
            f"Acc={metrics['accuracy']:.4f} | "
            f"mF1={metrics['macro_f1']:.4f} | "
            f"DMR={metrics['DMR']:.4f} | "
            f"s4u={metrics['s4_under']:.4f}"
        )
        log.info(
            f"Recall: [s1={rec[0]:.4f}, s2={rec[1]:.4f}, "
            f"s3={rec[2]:.4f}, s4={rec[3]:.4f}]"
        )
        log.info(
            f"★ s2 F1={metrics['s2_f1']:.4f} | "
            f"s3 F1={metrics['s3_f1']:.4f} | "
            f"s2+s3 F1={metrics['f1_s23']:.4f}"
        )

    # 兼容原有返回值
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "dangerous_miss": metrics["DMR"],
        "s4_underestimate": metrics["s4_under"],
        "recall_per_class": metrics["recall"].tolist(),
        "f1_s2": metrics["s2_f1"],
        "f1_s3": metrics["s3_f1"],
        "f1_s23": metrics["f1_s23"],
        "reject_rate": metrics["reject_rate"],
        "cm": cm_total,
    }


# ================================================================
# 8. τ 和温度搜索
# ================================================================

def search_best_tau_and_temperature(model, val_shards, device, criterion,
                                    batch_size=512, max_seq_len=200,
                                    tau_candidates=None, temp_candidates=None):
    """
    联合搜索 τ 和 Temperature，解耦选择逻辑：
        - Temperature：基于 NLL，选择 Top-3 温度带
        - Tau：在温度带内，基于安全门控 + 中间类质量约束
    """
    if tau_candidates is None:
        tau_candidates = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    if temp_candidates is None:
        temp_candidates = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]

    log.info("\n" + "=" * 60)
    log.info("JOINT SEARCH: τ (safety) + Temperature (calibration)")
    log.info("=" * 60)

    model.eval()
    all_lq1, all_lq2, all_lq3, all_ng, all_y = [], [], [], [], []

    pf = ShardPrefetcher(val_shards)
    while True:
        _, shard_obj = pf.next()
        if shard_obj is None:
            break
        ds = SingleShardSequenceDataset(shard_obj, max_seq_len=max_seq_len)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )
        with torch.no_grad():
            for X_b, y_b, lens_b in loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device)
                lens_b = lens_b.to(device)
                gate_b = X_b[:, :, GATE_INDICES]
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lq1, lq2, lq3, _, _, ng = model(X_b, lens_b, gate_b)
                B, T = lq1.shape
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                mask = (lens_b.unsqueeze(1) > t_idx) & (y_b >= 0)
                all_lq1.append(lq1[mask].cpu())
                all_lq2.append(lq2[mask].cpu())
                all_lq3.append(lq3[mask].cpu())
                all_ng.append(ng[mask].cpu())
                all_y.append(y_b[mask].cpu())
        del ds, loader, shard_obj
        gc.collect()
    pf.close()

    lq1n = torch.cat(all_lq1).numpy()
    lq2n = torch.cat(all_lq2).numpy()
    lq3n = torch.cat(all_lq3).numpy()
    ngn = torch.cat(all_ng).numpy()
    yn = torch.cat(all_y).numpy().astype(np.int64)
    yn_idx = _to_zero_based_labels(yn)
    log.info(f"Collected {len(yn):,} samples")

    high = yn >= 2
    s4a = yn == 3
    s2_mask = yn == 1
    s3_mask = yn == 2

    results = []

    for temp in temp_candidates:
        # ★ p_non_s1 也使用温度
        p_non_s1 = 1 / (1 + np.exp(-ngn / temp))
        q1 = 1 / (1 + np.exp(-lq1n / temp))
        q2 = 1 / (1 + np.exp(-lq2n / temp))
        q3 = 1 / (1 + np.exp(-lq3n / temp))

        p4_cond = np.maximum(q3, 0)
        p3_cond = np.maximum(q2 - q3, 0)
        p2_cond = np.maximum(q1 - q2, 0)
        cond_sum = p2_cond + p3_cond + p4_cond + 1e-8
        p2_cond /= cond_sum
        p3_cond /= cond_sum
        p4_cond /= cond_sum

        p_s1 = 1.0 - p_non_s1
        p_s2 = p_non_s1 * p2_cond
        p_s3 = p_non_s1 * p3_cond
        p_s4 = p_non_s1 * p4_cond
        probs = np.stack([p_s1, p_s2, p_s3, p_s4], axis=-1)

        # 计算 NLL（用于温度选择）
        eps = 1e-10
        safe_p = np.clip(probs, eps, 1.0)

        p_true = safe_p[np.arange(len(yn_idx)), yn_idx]
        nll = -np.log(p_true).mean()
        brier = np.mean(np.sum(safe_p ** 2, axis=1) - 2.0 * p_true + 1.0)

        for tau in tau_candidates:
            cdf = probs.cumsum(axis=-1)
            pred = (cdf >= tau).argmax(axis=-1)

            # ★ 使用 confusion matrix 计算指标（与 evaluate_on_shards 一致）
            cm = _fast_confusion_matrix(yn_idx, pred, num_classes=4)
            m = _metrics_from_cm(cm)

            results.append({
                "tau": tau,
                "temp": temp,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "DMR": m["DMR"],
                "s4_under": m["s4_under"],
                "f1_s23": m["f1_s23"],
                "s2_recall": m["recall"][1],
                "s3_recall": m["recall"][2],
                "s4_recall": m["recall"][3],
                "nll": nll,
                "brier": brier,
            })

    candidates = pd.DataFrame(results)

    # ---- 1) Temperature 选择：Top-3 NLL band ----
    temp_nll = candidates.groupby("temp")["nll"].mean().sort_values()
    best_nll = float(temp_nll.iloc[0])
    # 选择 NLL 在最优 1% 以内的温度，至少保留 3 个
    eligible_temps = temp_nll[temp_nll <= best_nll * 1.01].index.tolist()
    if len(eligible_temps) < 3:
        eligible_temps = temp_nll.head(3).index.tolist()

    log.info(f"\nEligible temperatures (NLL band): {eligible_temps}")

    # ---- 2) Tau 选择：在温度带内做安全门控 ----
    # ★ 收紧的安全门控
    safe = candidates[
        (candidates["temp"].isin(eligible_temps)) &
        (candidates["DMR"] <= 0.39) &  # 从 0.42 收紧
        (candidates["s4_under"] <= 0.50) &  # 从 0.52 收紧
        (candidates["accuracy"] >= 0.72) &  # 从 0.70 提升
        (candidates["f1_s23"] >= 0.18) &
        (candidates["s2_recall"] >= 0.45) &
        (candidates["s3_recall"] >= 0.25) &
        (candidates["s4_recall"] >= 0.45)
        ].copy()

    if len(safe) == 0:
        log.warning("No candidate satisfied all safety gates. Using fallback.")
        candidates["safe_loss"] = (
                0.42 * candidates["DMR"]
                + 0.24 * candidates["s4_under"]
                + 0.14 * np.maximum(0.18 - candidates["f1_s23"], 0)
                + 0.07 * np.maximum(0.72 - candidates["accuracy"], 0)
                + 0.06 * np.maximum(0.45 - candidates["s2_recall"], 0)
                + 0.06 * np.maximum(0.25 - candidates["s3_recall"], 0)
                + 0.06 * np.maximum(0.45 - candidates["s4_recall"], 0)
                - 0.05 * candidates["macro_f1"]
        )
        # 限制在温度带内选择
        temp_mask = candidates["temp"].isin(eligible_temps)
        if temp_mask.sum() > 0:
            best = candidates[temp_mask].sort_values("safe_loss", ascending=True).iloc[0]
        else:
            best = candidates.sort_values("safe_loss", ascending=True).iloc[0]
        log.warning(
            f"Fallback: tau={best['tau']}, temp={best['temp']}, "
            f"DMR={best['DMR']:.4f}, s4_under={best['s4_under']:.4f}, "
            f"f1_s23={best['f1_s23']:.4f}"
        )
    else:
        safe["safe_score"] = (
            0.35 * safe["f1_s23"]
            + 0.25 * (1.0 - safe["DMR"])
            + 0.20 * (1.0 - safe["s4_under"])
            + 0.20 * safe["macro_f1"]
        )
        best = safe.sort_values("safe_score", ascending=False).iloc[0]

    best_tau = float(best["tau"])
    best_temp = float(best["temp"])

    os.makedirs("results", exist_ok=True)
    candidates.to_csv(os.path.join("results", "tau_search_candidates.csv"), index=False)

    log.info(f"\n★ Final: τ={best_tau:.2f}, T={best_temp:.2f}")
    log.info(f"  DMR={best['DMR']:.4f}, s4_under={best['s4_under']:.4f}")
    log.info(f"  f1_s23={best['f1_s23']:.4f}, s2_recall={best['s2_recall']:.4f}, s3_recall={best['s3_recall']:.4f}")

    model.best_tau.fill_(best_tau)
    model.temperature.fill_(best_temp)

    return best_tau, best_temp

def _ramp_weight(base_w, start_epoch, ramp_epochs, epoch_idx):
    # epoch_idx 是 0-based epoch
    cur_epoch = epoch_idx + 1

    if cur_epoch < int(start_epoch):
        return 0.0

    ramp_pos = (cur_epoch - int(start_epoch) + 1) / max(float(ramp_epochs), 1.0)
    ramp = min(max(ramp_pos, 0.0), 1.0)

    return float(base_w) * ramp
# ================================================================
# 9. 训练主函数
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
    train_shards=None,  # ★ 新增
    val_shards=None,  # ★ 新增
):
    if device is None:
        device = DEVICE

    log.info("=" * 70)
    log.info("CASCADE WDR Training (v2 with proper cascade)")
    log.info("=" * 70)
    log.info(f"[Stage1] train_wrc_from_shards resume={resume}")
    _, _, feature_cols = load_stats()
    if train_shards is None or val_shards is None:
        train_shards, val_shards, _ = split_train_val_shards(val_ratio=0.15, seed=42)

    # ★ 新增：根据配置限制 shard 数量
    if MAX_TRAIN_SHARDS is not None and len(train_shards) > MAX_TRAIN_SHARDS:
        log.info(f"Limiting train shards from {len(train_shards)} to {MAX_TRAIN_SHARDS}")
        train_shards = train_shards[:MAX_TRAIN_SHARDS]

    if MAX_VAL_SHARDS is not None and len(val_shards) > MAX_VAL_SHARDS:
        log.info(f"Limiting val shards from {len(val_shards)} to {MAX_VAL_SHARDS}")
        val_shards = val_shards[:MAX_VAL_SHARDS]

    log.info(f"Train: {len(train_shards)} | Val: {len(val_shards)}")
    # 否则使用传入的 shards
    log.info(f"Train: {len(train_shards)} | Val: {len(val_shards)}")

    steps_per_epoch = sum(math.ceil(s["n_orders"] / batch_size) for s in train_shards)
    total_steps = epochs * steps_per_epoch
    log.info(f"steps/epoch={steps_per_epoch} | total={total_steps}")

    pw_q1, pw_q2, pw_q3, cls_alpha, global_counts = estimate_weights(
        train_shards, min(5, len(train_shards))
    )

    model = CascadeWDRNet(len(feature_cols), hidden_dim, num_layers).to(device)
    log.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    log_gpu_memory("Init ")

    criterion = CascadeLoss(
        pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3, cls_alpha=cls_alpha,
    ).to(device)

    # 自适应辅助权重 (不含 ng，因为 ng 已作为主任务之一，但 L_ng 不参与自适应，直接加)
    # 我们只对 cls, bnd, align, sharp 做自适应
    aux_weight = AdaptiveAuxWeight(
        task_names=["cls", "bnd", "align", "sharp", "seq_cls"],
        max_weights={
            "cls":   0.40,
            "bnd":   0.55,
            "align": 0.05,
            "sharp": 0.15,
            "seq_cls": 0.30,
        },
        min_weights={                     # ★ 新增这一整块
            "cls":   0.08,
            "bnd":   0.10,
            "align": 0.00,
            "sharp": 0.03,
            "seq_cls": 0.05,
        },
        init_weights={
            "cls":   0.20,
            "bnd":   0.30,
            "align": 0.01,
            "sharp": 0.05,
            "seq_cls": 0.15,
        },
    ).to(device)

    log.info("AdaptiveAuxWeight initial weights:")
    for name in ["cls", "bnd", "align", "sharp","seq_cls"]:
        w = aux_weight.get_weight(name).item()
        log.info(f"  {name:>8s}: {w:.4f}")

    # 优化器：模型参数 + aux_weight 参数
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
    tb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", f"cascade_{ts}")
    writer = SummaryWriter(log_dir=tb_dir)
    log.info(f"TensorBoard: tensorboard --logdir {tb_dir}")

    latest_ckpt = os.path.join(CHECKPOINT_DIR, "cascade_latest.pt")
    best_ckpt   = os.path.join(CHECKPOINT_DIR, "cascade_best.pt")

    start_epoch, best_score, best_state = 0, -1.0, None
    global_step, no_improve, patience = 0, 0, 3
    skip_tau_search = True
    if resume and os.path.exists(latest_ckpt):
        ckpt = load_training_checkpoint(latest_ckpt, device)
        log.info(f"[Stage1] Loading checkpoint from {latest_ckpt}")
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
                log.warning("AuxWeight state incompatible, using fresh weights")
            if "best_tau" in ckpt and "temperature" in ckpt:
                model.best_tau.fill_(ckpt["best_tau"])
                model.temperature.fill_(ckpt["temperature"])
                skip_tau_search = True
                log.info(
                    f"Loaded best_tau={ckpt['best_tau']:.2f}, temperature={ckpt['temperature']:.2f} from checkpoint. Skipping tau search.")
            else:
                log.info("No best_tau in checkpoint, will perform tau search.")
        else:
            log.info("[Stage1] Training from scratch; checkpoint resume disabled or checkpoint not found.")
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", -1.0)
        if "temperature" in ckpt:
            model.temperature.fill_(ckpt["temperature"])
        log.info(f"Resume epoch {start_epoch} | best_score={best_score:.4f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        t_loss_ng = t_loss_ord = t_loss_cls = t_loss_bnd = t_loss_align = t_loss_sharp = 0.0
        t_loss_boundary = t_loss_seq_cls = t_loss_s4 = t_loss_under = 0.0
        nb = 0
        epoch_start = time.time()

        # warm-up: 前 WARMUP_EPOCHS 关闭 align 和 sharp (但 cls, bnd 保持)
        disabled = set()

# cls auxiliary 当前长期为 0，没有有效贡献，先禁用。
        disabled.add("cls")

        if epoch < WARMUP_EPOCHS:
            disabled.add("align")
        # disabled = {"align"} if epoch < WARMUP_EPOCHS else set()
        # 同时，boundary loss 也在前1个epoch关闭
        boundary_weight = 0.0 if epoch < 1 else BOUNDARY_WEIGHT

        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch+1}/{epochs} | {len(train_shards)} shard(s)")
        log.info(f"Warm-up disabled: {sorted(disabled) if disabled else 'None'}")
        log.info(f"Boundary weight: {boundary_weight:.2f}")
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
                pin_memory=False,
            )

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
                    lq1, lq2, lq3, cls_logits, bnd_logit, ng_logit = model(X, lens, gate)
                    l_ng, l_ord, l_cls, l_bnd, l_align, l_sharp, l_boundary, l_seq_cls, l_s4, l_under = criterion(
                        lq1, lq2, lq3, cls_logits, bnd_logit, ng_logit, y
                    )

                    # 主任务：L_ng + L_ord (固定权重)
                    main_loss = l_ng + l_ord

                    # 辅助任务 (自适应)
                    aux_losses = {
                        "cls": l_cls,
                        "bnd": l_bnd,
                        "align": l_align,
                        "sharp": l_sharp,
                        "seq_cls": l_seq_cls,  # ★ 新增
                    }
                    aux_total, weight_info, contrib_info = aux_weight(
                        main_loss=main_loss,
                        aux_losses=aux_losses,
                        disabled_keys=disabled,
                    )
                    from config import (
                        S4_AUX_WEIGHT,
                        S4_AUX_START_EPOCH,
                        S4_AUX_RAMP_EPOCHS,
                        UNDER_AUX_WEIGHT,
                        UNDER_AUX_START_EPOCH,
                        UNDER_AUX_RAMP_EPOCHS,
                    )

                    s4_w = _ramp_weight(
                        S4_AUX_WEIGHT,
                        S4_AUX_START_EPOCH,
                        S4_AUX_RAMP_EPOCHS,
                        epoch,
                    )

                    under_w = _ramp_weight(
                        UNDER_AUX_WEIGHT,
                        UNDER_AUX_START_EPOCH,
                        UNDER_AUX_RAMP_EPOCHS,
                        epoch,
                    )

                    loss = (
                        aux_total
                        + boundary_weight * l_boundary
                        + s4_w * l_s4
                        + under_w * l_under
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

                t_loss_ng += float(l_ng.item())
                t_loss_ord += float(l_ord.item())
                t_loss_cls += float(l_cls.item())
                t_loss_bnd += float(l_bnd.item())
                t_loss_align += float(l_align.item())
                t_loss_sharp += float(l_sharp.item())
                t_loss_boundary += float(l_boundary.item())
                t_loss_seq_cls += float(l_seq_cls.item())
                t_loss_s4 += float(l_s4.item())
                t_loss_under += float(l_under.item())
                nb += 1

                if device.type == "cuda" and bi % 20 == 0:
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        ng=f"{l_ng.item():.3f}",
                        ord=f"{l_ord.item():.3f}",
                        cls=f"{l_cls.item():.3f}",
                        seq=f"{l_seq_cls.item():.3f}",
                        bnd=f"{l_bnd.item():.3f}",
                        gpu=f"{alloc:.1f}G",
                    )

            del ds, loader, shard_obj
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if sc % 30 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if sc == 1 or sc % 10 == 0 or sc == len(shards):
                elapsed = time.time() - epoch_start
                log.info(
                    f"  [E{epoch+1} S{sc}/{len(shards)}] "
                    f"ng={t_loss_ng/nb:.4f} ord={t_loss_ord/nb:.4f} "
                    f"cls={t_loss_cls/nb:.4f} bnd={t_loss_bnd/nb:.4f} "
                    f"align={t_loss_align/nb:.4f} sharp={t_loss_sharp/nb:.4f} "
                    f"boundary={t_loss_boundary/nb:.4f} "
                    f"seq_cls={t_loss_seq_cls/nb:.4f} "
                    f"s4={t_loss_s4/nb:.4f} under={t_loss_under/nb:.4f} | "
                    f"aux_w: cls={weight_info.get('cls',0):.3f} "
                    f"bnd={weight_info.get('bnd',0):.3f} "
                    f"align={weight_info.get('align',0):.3f} "
                    f"sharp={weight_info.get('sharp',0):.3f} "
                    f"seq_cls={weight_info.get('seq_cls',0):.3f} | "
                    f"s4_w={s4_w:.4f} under_w={under_w:.4f} | "
                    f"speed={nb/elapsed:.1f}b/s"
                )

        pf.close()
        epoch_time = time.time() - epoch_start
        a = lambda x: x / max(nb, 1)

        log.info(f"\n--- Epoch {epoch+1} Training ---")
        log.info(f"  Batches={nb:,} | Time={epoch_time:.0f}s | Speed={nb/max(epoch_time,1):.1f}b/s")
        log.info(
            f"  ng={a(t_loss_ng):.4f} ord={a(t_loss_ord):.4f} "
            f"cls={a(t_loss_cls):.4f} bnd={a(t_loss_bnd):.4f} "
            f"align={a(t_loss_align):.4f} sharp={a(t_loss_sharp):.4f} "
            f"boundary={a(t_loss_boundary):.4f} "
            f"seq_cls={a(t_loss_seq_cls):.4f} "
            f"s4={a(t_loss_s4):.4f} under={a(t_loss_under):.4f}"
        )

        log.info(f"  Adaptive aux weights:")
        for name in ["cls", "bnd", "align", "sharp", "seq_cls"]:
            log.info(f"    {name:>8s}: w={weight_info.get(name, 0):.4f} | rel_contrib={contrib_info.get(name, 0):.4f}")

        # 验证 (使用临时 τ=0.55, T=1.0)
        log.info(f"\n--- Epoch {epoch+1} Validation (τ=0.55, T=1.0) ---")
        val = evaluate_on_shards(
            model, val_shards, device, criterion,
            batch_size * 2, max_seq_len,
            verbose=True, tau=0.55, temperature=1.0,
            max_eval_batches=MAX_EVAL_BATCHES,  # ★ 新增
        )

        # 评分
        dmr_gate = 0.45 if epoch < 3 else 0.38
        if val["dangerous_miss"] > dmr_gate:
            score = -1.0
            log.info(f"Score gated to -1 (DMR={val['dangerous_miss']:.4f} > {dmr_gate:.2f})")
        else:
            score = (
                0.30 * (1.0 - val["dangerous_miss"]) +
                0.20 * (1.0 - val["s4_underestimate"]) +
                0.20 * val["macro_f1"] +
                0.30 * val["f1_s23"]
            )
            log.info(f"Score={score:.4f} | (1-DMR)={1.0-val['dangerous_miss']:.4f} "
                     f"(1-s4u)={1.0-val['s4_underestimate']:.4f} "
                     f"mF1={val['macro_f1']:.4f} s2+s3_F1={val['f1_s23']:.4f}")

        # TensorBoard
        writer.add_scalar("Loss/ng", a(t_loss_ng), epoch)
        writer.add_scalar("Loss/ord", a(t_loss_ord), epoch)
        writer.add_scalar("Loss/cls", a(t_loss_cls), epoch)
        writer.add_scalar("Loss/bnd", a(t_loss_bnd), epoch)
        writer.add_scalar("Loss/align", a(t_loss_align), epoch)
        writer.add_scalar("Loss/sharp", a(t_loss_sharp), epoch)
        writer.add_scalar("Loss/boundary", a(t_loss_boundary), epoch)
        writer.add_scalar("M/DMR", val["dangerous_miss"], epoch)
        writer.add_scalar("M/s4u", val["s4_underestimate"], epoch)
        writer.add_scalar("M/macro_f1", val["macro_f1"], epoch)
        writer.add_scalar("M/f1_s23", val["f1_s23"], epoch)
        writer.add_scalar("M/score", score, epoch)
        writer.add_scalar("Loss/seq_cls", a(t_loss_seq_cls), epoch)

        writer.add_scalar("Loss/s4", a(t_loss_s4), epoch)
        writer.add_scalar("Loss/under", a(t_loss_under), epoch)
        writer.add_scalar("AuxWeight/s4_static", s4_w, epoch)
        writer.add_scalar("AuxWeight/under_static", under_w, epoch)

        for name in ["cls", "bnd", "align", "sharp", "seq_cls"]:
            writer.add_scalar(f"AuxWeight/{name}", weight_info.get(name, 0), epoch)

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

    # 联合搜索 τ 和 T
    # ---- 根据配置决定是否搜索 tau ----
    from config import SKIP_TAU_SEARCH, DEFAULT_TAU, DEFAULT_TEMPERATURE

    if SKIP_TAU_SEARCH:
        best_tau = DEFAULT_TAU
        best_temp = DEFAULT_TEMPERATURE
        model.best_tau.fill_(best_tau)
        model.temperature.fill_(best_temp)
        log.info(f"\n=== Skipping tau search (SKIP_TAU_SEARCH=True) ===")
        log.info(f"Using default: τ={best_tau:.2f}, T={best_temp:.2f}")

        # ★ 创建 DataFrame 保存默认值
        tau_df = pd.DataFrame([{"tau": best_tau, "temperature": best_temp, "note": "default_skip"}])
        tau_df.to_csv("tau_search_results.csv", index=False)
        log.info(f"Default tau saved to tau_search_results.csv")

        log.info(f"\n=== Final Validation (τ={best_tau:.2f}, T={best_temp:.2f}) ===")
        evaluate_on_shards(
            model, val_shards, device, criterion,
            batch_size * 2, max_seq_len,
            verbose=True,
            tau=best_tau,
            temperature=best_temp,  # ★ 新增
            max_eval_batches=MAX_EVAL_BATCHES,  # ★ 新增
        )
    else:
        best_tau, best_temp = search_best_tau_and_temperature(
            model, val_shards, device, criterion,
            batch_size * 2, max_seq_len

        )
        # 保存搜索结果
        pd.DataFrame([{"tau": best_tau, "temperature": best_temp}]).to_csv("tau_search_results.csv", index=False)
        log.info(f"\n=== Final Validation (τ={best_tau:.2f}) ===")
        evaluate_on_shards(
            model, val_shards, device, criterion,
            batch_size * 2, max_seq_len,
            verbose=True, tau=best_tau,
            temperature = best_temp,  # ★ 必须传
            max_eval_batches=MAX_EVAL_BATCHES,  # ★ 新增
        )

    writer.close()
    return model


# ================================================================
# 10. 保存 / 加载
# ================================================================

def save_wrc_model(model, tag="cascade_v1"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_DIR, f"stage1_cascade_{tag}_{ts}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "dense_dim": model.dense_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "model_type": "cascade",
        "best_tau": float(model.best_tau.item()),
        "temperature": float(model.temperature.item()),
    }, path)
    log.info(f"Saved: {path} (τ={model.best_tau.item():.2f}, T={model.temperature.item():.2f})")
    return path


def load_wrc_model(path, device=None):
    if device is None: device = DEVICE
    d = torch.load(path, map_location=device)
    model = CascadeWDRNet(
        d["dense_dim"], d["hidden_dim"], d["num_layers"]
    ).to(device)
    model.load_state_dict(d["model_state"])
    model.eval()
    if "best_tau" in d:
        model.best_tau.fill_(d["best_tau"])
    if "temperature" in d:
        model.temperature.fill_(d["temperature"])
    log.info(f"Loaded: {path} (τ={model.best_tau.item():.2f}, T={model.temperature.item():.2f})")
    return model


# ================================================================
# 11. 推理（Stage 2 主输出 = cascade probability）
# ================================================================

def predict_proba_wrc(model, df, max_seq_len=WRC_MAX_SEQ_LEN,
                      batch_size=None, device=None,
                      temperature=None):
    from config import WRC_INFER_BATCH_SIZE

    if batch_size is None:
        batch_size = int(WRC_INFER_BATCH_SIZE)
    if device is None: device = DEVICE
    model.eval();
    model.to(device)

    if temperature is None:
        temperature = float(model.temperature.item())

    mean_dict, std_dict, feature_cols = load_stats()

    df = df.copy().reset_index(drop=True)
    df["_row_idx"] = np.arange(len(df))

    # x = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    #
    # # ★ 核心修复：按特征类型分别处理
    # for col in feature_cols:
    #     if col in RAW_ID_FEATURES:
    #         # 离散 ID：不标准化，只填充缺失值（用 0 填充，代表 unknown）
    #         x[col] = x[col].fillna(0)
    #     elif col in PERIODIC_FEATURES:
    #         # 周期特征：只 clip 到 [-1, 1]
    #         x[col] = x[col].fillna(0).clip(-1.0, 1.0)
    #     else:
    #         # 连续特征：Z-score 标准化
    #         d = std_s[col] if std_s[col] > 1e-8 else 1.0
    #         x[col] = (x[col].fillna(mean_s[col]) - mean_s[col]) / d
    #
    # # ★ 对离散 ID 做 clip 保底（确保在 embedding 词表范围内）
    # x["link_current_status"] = x["link_current_status"].clip(0, 4)
    # x["slice_id"] = x["slice_id"].clip(0, 287)
    # x["arrival_slice_est"] = x["arrival_slice_est"].clip(0, 287)

    scaled = transform_stage1_feature_array(
        df=df,
        feature_cols=feature_cols,
        mean_dict=mean_dict,
        std_dict=std_dict,
    )

    # 获取离散列在特征矩阵中的位置（需要从 STAGE1_FEATURE_COLS 获取索引）
    status_idx = STAGE1_FEATURE_COLS.index("link_current_status")
    slice_idx = STAGE1_FEATURE_COLS.index("slice_id")
    arrival_idx = STAGE1_FEATURE_COLS.index("arrival_slice_est")

    # ★ 在 scaled 上做 clamp
    scaled[:, status_idx] = np.clip(scaled[:, status_idx], 0, 4)
    scaled[:, slice_idx] = np.clip(scaled[:, slice_idx], 0, 287)
    scaled[:, arrival_idx] = np.clip(scaled[:, arrival_idx], 0, 287)

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
                      desc=f"[Cascade predict T={temperature:.2f}]", dynamic_ncols=True, unit="b"):
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
                lq1, lq2, lq3, _, _, ng_logit = model(sp, lt, gt)
                probs, _, _ = model.get_cascade_probs(lq1, lq2, lq3, ng_logit, temperature=temperature)

                cdf = probs.cumsum(dim=-1)
                pred_cls_t = (cdf >= tau).float().argmax(dim=-1)

                B, T = lq1.shape
                t_idx = torch.arange(T, device=device).unsqueeze(0)
                mask = lt.unsqueeze(1) > t_idx

                eps = 1e-10
                safe_p = probs.clamp_min(eps)
                entropy = -(safe_p * torch.log(safe_p)).sum(dim=-1)
                entropy = torch.where(mask, entropy, torch.zeros_like(entropy))
                should_reject = (entropy > 1.2) & mask

                probs_np  = probs.cpu().numpy()
                reject_np = should_reject.cpu().numpy()
                cls_np    = pred_cls_t.cpu().numpy()
                # 输出 q2, q3 (用于 stage2)
                q2_np = torch.sigmoid(lq2 / temperature).cpu().numpy()
                q3_np = torch.sigmoid(lq3 / temperature).cpu().numpy()

        for b, (L, idxs) in enumerate(zip(lens, ri)):
            pred_proba[idxs]  = probs_np[b, :L]
            pred_q2[idxs]     = q2_np[b, :L]
            pred_q3[idxs]     = q3_np[b, :L]
            pred_reject[idxs] = reject_np[b, :L]
            pred_status[idxs] = cls_np[b, :L].astype(np.float32) + 1.0

    # ★ 概率语义修正
    for k in range(NUM_CLASSES):
        df[f"pred_p{k+1}"] = pred_proba[:, k].astype("float32")
    # q2/q3 are raw cascade/ordinal heads; they are NOT unconditional probabilities.
    df["pred_q2_raw"] = pred_q2.astype("float32")
    df["pred_q3_raw"] = pred_q3.astype("float32")

    # ★ Stage 2 应使用无条件类别概率
    df["pred_cong_prob"] = (pred_proba[:, 2] + pred_proba[:, 3]).astype("float32")  # P(s3)+P(s4)
    df["pred_risk_prob"] = pred_proba[:, 3].astype("float32")                      # P(s4)

    df["pred_expected_status"] = (
        pred_proba[:, 0] * 1.0
        + pred_proba[:, 1] * 2.0
        + pred_proba[:, 2] * 3.0
        + pred_proba[:, 3] * 4.0
    ).astype("float32")

    df["pred_status"] = pred_status.astype("float32")

    eps = 1e-10
    safe_p = pred_proba.clip(min=eps)
    df["pred_entropy"] = -(safe_p * np.log(safe_p)).sum(1).astype("float32")

    # omega 改为期望超阈状态（更平滑）
    df["pred_omega"] = (
        pred_proba[:, 2] * 1.0
        + pred_proba[:, 3] * 2.0
    ).astype("float32")

    df["pred_reject"] = pred_reject.astype(bool)

    df.drop(columns=["_row_idx"], inplace=True)
    return df


