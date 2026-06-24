# -*- coding: utf-8 -*-

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from config import (
    HIST_CONTEXT_DIR,
    HIST_CONTEXT_VERSION,
    HIST_FEATURE_COLS,
    HIST_CONTEXT_WINDOWS,
    HIST_N_BUCKETS,
    HIST_BUCKET_CACHE_SIZE,
    USE_HIST_CONTEXT,
)
from logger import get_logger

log = get_logger()


def _target_bucket_path(day: str, bucket: int):
    day = str(day).zfill(2)
    return os.path.join(
        HIST_CONTEXT_DIR,
        HIST_CONTEXT_VERSION,
        "target_buckets",
        f"day_{day}",
        f"bucket_{bucket:03d}.pkl",
    )


def _make_hist_key(link_id, slice_id):
    return link_id.astype("int64") * 288 + slice_id.astype("int64")


@lru_cache(maxsize=HIST_BUCKET_CACHE_SIZE)
def load_historical_context_bucket(day: str, bucket: int):
    day = str(day).zfill(2)
    bucket = int(bucket)

    path = _target_bucket_path(day, bucket)

    if not os.path.exists(path):
        log.warning(f"[HistContext] missing target bucket: day={day}, bucket={bucket:03d}, path={path}")
        return None

    hist = pd.read_pickle(path)

    if hist is None or len(hist) == 0:
        return None

    for col in HIST_FEATURE_COLS:
        if col not in hist.columns:
            hist[col] = np.nan

    hist = hist[["hist_key"] + HIST_FEATURE_COLS].drop_duplicates("hist_key")
    hist = hist.set_index("hist_key")
    hist = hist.astype("float32", copy=False)

    return hist


def clear_historical_context_cache():
    load_historical_context_bucket.cache_clear()


def _fallback_values():
    fb = {
        "hist_status_mean": 1.5,
        "hist_cong_prob": 0.05,
        "hist_s4_prob": 0.01,
        "hist_entropy": 0.0,
        "hist_link_time_mean": 0.0,
        "hist_link_time_std": 0.0,
        "hist_n_log": 0.0,
        "hist_missing": 1.0,
    }

    for k in HIST_CONTEXT_WINDOWS:
        fb[f"hist_cong_win{k}"] = 0.05
        fb[f"hist_s4_win{k}"] = 0.01
        fb[f"hist_status_win{k}"] = 1.5

    return fb


def _fill_empty(df: pd.DataFrame):
    out = df.copy()
    fb = _fallback_values()

    for col in HIST_FEATURE_COLS:
        out[col] = fb.get(col, 0.0)

    if "hist_missing" in HIST_FEATURE_COLS:
        out["hist_missing"] = 1.0

    return out


def merge_historical_context(df: pd.DataFrame, day=None) -> pd.DataFrame:
    if not USE_HIST_CONTEXT:
        return df

    if df is None or df.empty:
        return df

    if day is None:
        if "day" not in df.columns:
            log.warning("[HistContext] df has no day column; using empty hist features.")
            return _fill_empty(df)

        days = df["day"].astype(str).str.zfill(2).unique()

        if len(days) > 1:
            parts = []
            for d, part in df.groupby("day", sort=False):
                parts.append(merge_historical_context(part, day=str(d).zfill(2)))
            return pd.concat(parts, ignore_index=True)

        day = days[0]

    day = str(day).zfill(2)

    if "arrival_slice_est" not in df.columns:
        log.warning("[HistContext] arrival_slice_est missing; using empty hist features.")
        return _fill_empty(df)

    out = df.copy()

    fb = _fallback_values()

    # 先填 fallback
    for col in HIST_FEATURE_COLS:
        out[col] = fb.get(col, 0.0)

    if "hist_missing" in HIST_FEATURE_COLS:
        out["hist_missing"] = 1.0

    link_ids = out["link_id"].astype("int64")
    slices = out["arrival_slice_est"].fillna(0).astype("int64")

    keys = _make_hist_key(link_ids, slices)
    buckets = (link_ids % HIST_N_BUCKETS).astype("int16")

    # 按 bucket lookup，避免加载一个巨大 day context
    for b in np.sort(buckets.unique()):
        mask = buckets == b
        if not mask.any():
            continue

        hist = load_historical_context_bucket(day, int(b))
        if hist is None or hist.empty:
            continue

        sub_keys = keys[mask]
        vals = hist.reindex(sub_keys)

        if vals is None or len(vals) == 0:
            continue

        # 命中的行：hist_n_log 非空
        if "hist_n_log" in vals.columns:
            hit = vals["hist_n_log"].notna().values
        else:
            hit = np.ones(len(vals), dtype=bool)

        idx = out.index[mask]

        for col in HIST_FEATURE_COLS:
            if col not in vals.columns:
                continue

            arr = vals[col].to_numpy(dtype="float32", copy=False)
            # 只覆盖非 NaN
            notna = ~np.isnan(arr)
            if notna.any():
                out.loc[idx[notna], col] = arr[notna]

        if "hist_missing" in HIST_FEATURE_COLS:
            out.loc[idx[hit], "hist_missing"] = 0.0

    for col in HIST_FEATURE_COLS:
        out[col] = out[col].fillna(fb.get(col, 0.0)).astype("float32")

    return out