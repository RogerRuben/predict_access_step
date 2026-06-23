"""
Stage 1 特征类型定义（训练 + 推理共用）

避免 train / inference 标准化规则不一致。
"""

RAW_ID_FEATURES = frozenset({
    "link_id",
    "slice_id",
    "arrival_slice_est",
})


ORDINAL_RAW_FEATURES = frozenset({
    # ordinal raw value, not an embedding id:
    # 0 = unknown, 1-4 = traffic status (speed-derived)
    "link_current_status",
})
PERIODIC_FEATURES = frozenset({
    "sin_slice",
    "cos_slice",
    "sin_arr_slice",
    "cos_arr_slice",
})

NON_STANDARDIZE_FEATURES = RAW_ID_FEATURES | ORDINAL_RAW_FEATURES | PERIODIC_FEATURES

# 合法值范围
ID_CLIP_RANGES = {
    "link_id": (0, None),  # 只保底
    "link_current_status": (0, 4),
    "slice_id": (0, 287),
    "arrival_slice_est": (0, 287),
}