#!/usr/bin/env python
"""
compute_stage2_day.py
独立子进程：计算单天 Stage 2 特征并写入 cache
"""

import argparse
import sys
import os
import gc
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import _process_day_stage2, _make_cache_key, _cache_paths
from config import STAGE1_MODEL
from loader import load_topology
from pipeline import _process_day_stage2
from logger import get_logger

log = get_logger()


def _load_stage1_model_for_stage2(stage1_model_path: str):
    """根据 STAGE1_MODEL 加载对应的模型"""
    if not os.path.exists(stage1_model_path):
        raise FileNotFoundError(f"Stage 1 model not found: {stage1_model_path}")

    if STAGE1_MODEL in ("wrc", "wdr", "hierarchical", "hier", "ordinal", "dualhead", "triple", "cascade"):
        from stage1_deep import load_wrc_model
        return load_wrc_model(stage1_model_path)

    if STAGE1_MODEL == "lgbm":
        from stage1_predict import load_stage1_model
        return load_stage1_model(stage1_model_path)

    raise ValueError(f"Unsupported STAGE1_MODEL={STAGE1_MODEL}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=str, required=True)
    parser.add_argument("--stage1-model-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--ct-threshold", type=float, required=True)
    parser.add_argument("--cross-global-mean", type=float, required=True)
    args = parser.parse_args()

    log.info(f"Starting Stage 2 for day {args.day} (mode={args.mode})")

    try:
        # ★ 不预加载 topology 和 model，交给 _process_day_stage2 lazy load
        s2_df, head_df = _process_day_stage2(
            day=args.day,
            topo=None,
            model=None,
            ct_threshold=args.ct_threshold,
            cross_global_mean=args.cross_global_mean,
            s1_model_path=args.stage1_model_path,
        )

        if s2_df is None or s2_df.empty:
            cache_key = _make_cache_key(
                day=args.day,
                s1_path=args.stage1_model_path,
                ct_thresh=args.ct_threshold,
                cross_mean=args.cross_global_mean,
            )

            s2_cache, head_cache = _cache_paths(cache_key)

            log.info(
                f"Day {args.day}: verifying cache_key={cache_key} | "
                f"s2={s2_cache} | head={head_cache}"
            )

            # 如果 _process_day_stage2 没有写 cache，这里兜底写入
            if not (os.path.exists(s2_cache) and os.path.exists(head_cache)):
                log.warning(
                    f"Day {args.day}: cache missing after _process_day_stage2; "
                    f"writing cache explicitly."
                )

                os.makedirs(os.path.dirname(s2_cache), exist_ok=True)

                s2_df.to_pickle(s2_cache)
                head_df.to_pickle(head_cache)

            # 再次强校验
            if not (os.path.exists(s2_cache) and os.path.exists(head_cache)):
                raise RuntimeError(
                    f"Day {args.day}: failed to create expected cache files: "
                    f"{s2_cache}, {head_cache}"
                )

            log.info(
                f"Day {args.day}: cache verified successfully. "
                f"cache_key={cache_key}, rows={len(s2_df)}"
            )

        log.info(f"Day {args.day}: Stage 2 completed with {len(s2_df)} orders")

    except Exception as e:
        log.error(f"Day {args.day}: Stage 2 failed: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)

    log.info(f"Day {args.day}: Stage 2 finished successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()