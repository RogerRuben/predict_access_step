"""
Stage 1 特征类型定义（训练 + 推理共用）

避免 train / inference 标准化规则不一致。
"""

from config import HIST_FEATURE_COLS

RAW_ID_FEATURES = frozenset({
    "link_id",
    "slice_id",
    "arrival_slice_est",
})

ORDINAL_RAW_FEATURES = frozenset({
    # 0 = unknown, 1-4 = traffic status
    "link_current_status",
})

PERIODIC_FEATURES = frozenset({
    "sin_slice",
    "cos_slice",
    "sin_arr_slice",
    "cos_arr_slice",
})

NON_STANDARDIZE_FEATURES = RAW_ID_FEATURES | ORDINAL_RAW_FEATURES | PERIODIC_FEATURES

# 历史特征应该被标准化，因此不能进入 NON_STANDARDIZE_FEATURES
assert len(set(HIST_FEATURE_COLS) & NON_STANDARDIZE_FEATURES) == 0

ID_CLIP_RANGES = {
    "link_id": (0, None),
    "link_current_status": (0, 4),
    "slice_id": (0, 287),
    "arrival_slice_est": (0, 287),
}