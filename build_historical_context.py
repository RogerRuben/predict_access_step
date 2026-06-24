# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os

import numpy as np
import pandas as pd

from config import (
    HIST_CONTEXT_DIR,
    HIST_CONTEXT_VERSION,
    HIST_CONTEXT_WINDOWS,
    HIST_SMOOTH_PRIOR,
    HIST_USE_ARRIVAL_SLICE,
    HIST_BUILD_BATCH_SIZE_ORDERS,
    HIST_FEATURE_COLS,
)

from config import (
    HIST_N_BUCKETS,
    HIST_TARGET_MIN_COUNT,
)
from loader import load_day
from logger import get_logger

log = get_logger()

STATUS_VALUES = [1, 2, 3, 4]

def _daily_bucket_path(day: str, bucket: int):
    day = str(day).zfill(2)
    out_dir = os.path.join(HIST_CONTEXT_DIR, HIST_CONTEXT_VERSION, "daily_buckets", f"day_{day}")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"bucket_{bucket:03d}.pkl")


def _target_bucket_path(day: str, bucket: int):
    day = str(day).zfill(2)
    out_dir = os.path.join(HIST_CONTEXT_DIR, HIST_CONTEXT_VERSION, "target_buckets", f"day_{day}")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"bucket_{bucket:03d}.pkl")


def _target_bucket_meta_path(day: str):
    day = str(day).zfill(2)
    out_dir = os.path.join(HIST_CONTEXT_DIR, HIST_CONTEXT_VERSION, "target_buckets", f"day_{day}")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "meta.json")

def _parse_days(s: str):
    if not s:
        return []
    return [x.strip().zfill(2) for x in s.split(",") if x.strip()]


def _parse_windows(s: str):
    if not s:
        return list(HIST_CONTEXT_WINDOWS)
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})


def _out_dir():
    d = os.path.join(HIST_CONTEXT_DIR, HIST_CONTEXT_VERSION)
    os.makedirs(d, exist_ok=True)
    return d


def _daily_base_path(day: str):
    day = str(day).zfill(2)
    return os.path.join(_out_dir(), f"daily_base_{day}.pkl")


def _target_base_path(day: str):
    day = str(day).zfill(2)
    return os.path.join(_out_dir(), f"hist_base_day_{day}.pkl")


def _target_meta_path(day: str):
    day = str(day).zfill(2)
    return os.path.join(_out_dir(), f"hist_meta_day_{day}.json")


def _merge_sum_acc(acc, part, keys, sum_cols):
    if part is None or len(part) == 0:
        return acc

    part = part[list(keys) + list(sum_cols)].copy()

    if acc is None or len(acc) == 0:
        return part

    tmp = pd.concat([acc, part], ignore_index=True)

    out = (
        tmp.groupby(list(keys), sort=False)[list(sum_cols)]
        .sum()
        .reset_index()
    )

    del acc, part, tmp
    gc.collect()

    return out


def _prepare_hist_link_batch(day, head_b, link_b, cross_b):
    """
    构造历史聚合所需的 batch-level link 表。

    slice_key 必须和 feature_eng.py 的 arrival_slice_est 对齐：
      wt_with_cross = link_time + link_ratio + downstream_cross_time
      cum_travel_time = groupby(order_id, day).cumsum(wt_with_cross)
      slice_key = round(slice_id + cum_travel_time / 300) % 288
    """
    day = str(day).zfill(2)

    head = head_b[["order_id", "day", "slice_id"]].copy()
    link = link_b[
        [
            "order_id",
            "day",
            "link_id",
            "link_time",
            "link_ratio",
            "link_current_status",
            "link_arrival_status",
        ]
    ].copy()

    cross = cross_b[["order_id", "day", "cross_id", "cross_time"]].copy()

    head["order_id"] = head["order_id"].astype("int64")
    head["day"] = head["day"].astype(str).str.zfill(2)
    head["slice_id"] = head["slice_id"].fillna(0).astype("int16")

    link["order_id"] = link["order_id"].astype("int64")
    link["day"] = link["day"].astype(str).str.zfill(2)
    link["link_id"] = link["link_id"].astype("int64")
    link["link_time"] = link["link_time"].fillna(0).astype("float32")
    link["link_ratio"] = link["link_ratio"].fillna(0).astype("float32")
    link["link_current_status"] = link["link_current_status"].fillna(0).astype("int8")
    link["link_arrival_status"] = link["link_arrival_status"].fillna(1).astype("int8")

    df = link.merge(head, on=["order_id", "day"], how="left")
    df["slice_id"] = df["slice_id"].fillna(0).astype("int16")

    grp = df.groupby(["order_id", "day"], sort=False)
    df["next_link_id"] = grp["link_id"].shift(-1)

    df["cross_key"] = ""
    mask = df["next_link_id"].notna()
    if mask.any():
        df.loc[mask, "cross_key"] = (
            df.loc[mask, "link_id"].astype(str)
            + "_"
            + df.loc[mask, "next_link_id"].astype("int64").astype(str)
        )

    cross["order_id"] = cross["order_id"].astype("int64")
    cross["day"] = cross["day"].astype(str).str.zfill(2)
    cross["cross_id"] = cross["cross_id"].astype(str)
    cross["cross_time"] = cross["cross_time"].fillna(0).astype("float32")

    cross_rename = cross.rename(
        columns={
            "cross_id": "cross_key",
            "cross_time": "downstream_cross_time",
        }
    )

    df = df.merge(
        cross_rename,
        on=["order_id", "day", "cross_key"],
        how="left",
    )

    df["downstream_cross_time"] = df["downstream_cross_time"].fillna(0).astype("float32")

    grp = df.groupby(["order_id", "day"], sort=False)
    df["wt_with_cross"] = (
        df["link_time"] + df["link_ratio"] + df["downstream_cross_time"]
    ).astype("float32")

    df["cum_travel_time"] = grp["wt_with_cross"].cumsum().astype("float32")

    if HIST_USE_ARRIVAL_SLICE:
        df["slice_key"] = (
            df["slice_id"].astype("float32") + df["cum_travel_time"] / 300.0
        ).round().astype("int16") % 288
    else:
        df["slice_key"] = df["slice_id"].astype("int16")

    out = df[
        [
            "link_id",
            "slice_key",
            "link_time",
            "link_arrival_status",
        ]
    ].copy()

    del head, link, cross, df, grp
    gc.collect()

    return out


def _aggregate_hist_batch(hist_df):
    hist_df = hist_df[hist_df["link_arrival_status"].isin(STATUS_VALUES)].copy()

    if hist_df.empty:
        return pd.DataFrame()

    hist_df["hist_n"] = 1
    hist_df["status_sum"] = hist_df["link_arrival_status"].astype("float32")
    hist_df["cong_sum"] = (hist_df["link_arrival_status"] >= 3).astype("float32")
    hist_df["s4_sum"] = (hist_df["link_arrival_status"] == 4).astype("float32")
    hist_df["link_time_sum"] = hist_df["link_time"].astype("float32")
    hist_df["link_time_sumsq"] = (
        hist_df["link_time"].astype("float32") ** 2
    ).astype("float32")

    for s in STATUS_VALUES:
        hist_df[f"cnt_s{s}"] = (hist_df["link_arrival_status"] == s).astype("int32")

    base = (
        hist_df.groupby(["link_id", "slice_key"], sort=False)
        .agg(
            hist_n=("hist_n", "sum"),
            status_sum=("status_sum", "sum"),
            cong_sum=("cong_sum", "sum"),
            s4_sum=("s4_sum", "sum"),
            link_time_sum=("link_time_sum", "sum"),
            link_time_sumsq=("link_time_sumsq", "sum"),
            cnt_s1=("cnt_s1", "sum"),
            cnt_s2=("cnt_s2", "sum"),
            cnt_s3=("cnt_s3", "sum"),
            cnt_s4=("cnt_s4", "sum"),
        )
        .reset_index()
    )

    base["link_id"] = base["link_id"].astype("int64")
    base["slice_key"] = base["slice_key"].astype("int16")

    int_cols = ["hist_n", "cnt_s1", "cnt_s2", "cnt_s3", "cnt_s4"]
    for c in int_cols:
        base[c] = base[c].astype("int32")

    float_cols = [
        "status_sum",
        "cong_sum",
        "s4_sum",
        "link_time_sum",
        "link_time_sumsq",
    ]
    for c in float_cols:
        base[c] = base[c].astype("float32")

    del hist_df
    gc.collect()

    return base


def build_daily_base_agg(day: str, overwrite=False):
    day = str(day).zfill(2)
    out_path = _daily_base_path(day)

    if os.path.exists(out_path) and not overwrite:
        log.info(f"[HistDaily] day={day}: exists, skip.")
        return

    log.info("=" * 70)
    log.info(f"[HistDaily] Building daily aggregate for day {day}")
    log.info("=" * 70)

    head, link, cross = load_day(day)

    unique_orders = link["order_id"].unique()
    log.info(
        f"[HistDaily] day={day}: orders={len(unique_orders):,}, "
        f"link_rows={len(link):,}"
    )

    keys = ["link_id", "slice_key"]
    sum_cols = [
        "hist_n",
        "status_sum",
        "cong_sum",
        "s4_sum",
        "link_time_sum",
        "link_time_sumsq",
        "cnt_s1",
        "cnt_s2",
        "cnt_s3",
        "cnt_s4",
    ]

    acc = None

    for i in range(0, len(unique_orders), HIST_BUILD_BATCH_SIZE_ORDERS):
        batch_orders = unique_orders[i:i + HIST_BUILD_BATCH_SIZE_ORDERS]

        link_b = link[link["order_id"].isin(batch_orders)]
        head_b = head[head["order_id"].isin(batch_orders)]
        cross_b = cross[cross["order_id"].isin(batch_orders)]

        hist_link = _prepare_hist_link_batch(day, head_b, link_b, cross_b)
        part = _aggregate_hist_batch(hist_link)

        acc = _merge_sum_acc(acc, part, keys, sum_cols)

        log.info(
            f"[HistDaily] day={day}: processed {min(i + HIST_BUILD_BATCH_SIZE_ORDERS, len(unique_orders)):,}/"
            f"{len(unique_orders):,} orders | acc_rows={0 if acc is None else len(acc):,}"
        )

        del link_b, head_b, cross_b, hist_link, part
        gc.collect()

    if acc is None or len(acc) == 0:
        raise RuntimeError(f"[HistDaily] day={day}: empty aggregate")

    acc.to_pickle(out_path)

    log.info(f"[HistDaily] saved {out_path}, rows={len(acc):,}")

    del head, link, cross, acc
    gc.collect()


def _sum_daily_aggs_for_bucket(source_days, bucket: int):
    keys = ["link_id", "slice_key"]
    sum_cols = [
        "hist_n",
        "status_sum",
        "cong_sum",
        "s4_sum",
        "link_time_sum",
        "link_time_sumsq",
        "cnt_s1",
        "cnt_s2",
        "cnt_s3",
        "cnt_s4",
    ]

    acc = None

    for d in source_days:
        d = str(d).zfill(2)
        p = _daily_bucket_path(d, bucket)

        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing daily bucket: {p}")

        part = pd.read_pickle(p)
        part = part[keys + sum_cols].copy()

        if acc is None or len(acc) == 0:
            acc = part
        else:
            tmp = pd.concat([acc, part], ignore_index=True)

            # 分 int / float 聚合，避免 pandas 对宽表做 float64 downcast 临时数组
            int_cols = ["hist_n", "cnt_s1", "cnt_s2", "cnt_s3", "cnt_s4"]
            float_cols = [
                "status_sum",
                "cong_sum",
                "s4_sum",
                "link_time_sum",
                "link_time_sumsq",
            ]

            g_int = (
                tmp.groupby(keys, sort=False)[int_cols]
                .sum()
                .reset_index()
            )
            g_float = (
                tmp.groupby(keys, sort=False)[float_cols]
                .sum()
                .reset_index()
            )

            acc = g_int.merge(g_float, on=keys, how="inner")

            del tmp, g_int, g_float, part
            gc.collect()

        log.info(
            f"[HistContext] bucket={bucket:03d}: added day={d}, "
            f"rows={0 if acc is None else len(acc):,}"
        )

    if acc is None:
        return pd.DataFrame(columns=keys + sum_cols)

    # 低样本 key 直接丢弃，prepare 时 fallback
    if HIST_TARGET_MIN_COUNT and HIST_TARGET_MIN_COUNT > 1:
        before = len(acc)
        acc = acc[acc["hist_n"] >= HIST_TARGET_MIN_COUNT].copy()
        log.info(
            f"[HistContext] bucket={bucket:03d}: min_count={HIST_TARGET_MIN_COUNT}, "
            f"rows {before:,} -> {len(acc):,}"
        )

    return acc


def _finalize_base_features(base_raw: pd.DataFrame):
    prior = float(HIST_SMOOTH_PRIOR)

    n = base_raw["hist_n"].astype("float64").values
    n_safe = np.maximum(n, 1.0)

    total_n = float(base_raw["hist_n"].sum())

    global_status = float(base_raw["status_sum"].sum() / max(total_n, 1.0))
    global_cong = float(base_raw["cong_sum"].sum() / max(total_n, 1.0))
    global_s4 = float(base_raw["s4_sum"].sum() / max(total_n, 1.0))

    global_counts = np.array(
        [
            base_raw["cnt_s1"].sum(),
            base_raw["cnt_s2"].sum(),
            base_raw["cnt_s3"].sum(),
            base_raw["cnt_s4"].sum(),
        ],
        dtype="float64",
    )
    global_probs = global_counts / max(global_counts.sum(), 1.0)

    local_status = base_raw["status_sum"].values.astype("float64") / n_safe
    local_cong = base_raw["cong_sum"].values.astype("float64") / n_safe
    local_s4 = base_raw["s4_sum"].values.astype("float64") / n_safe

    out = base_raw[["link_id", "slice_key", "hist_n"]].copy()

    out["hist_status_mean"] = (
        (n * local_status + prior * global_status) / (n + prior)
    ).astype("float32")

    out["hist_cong_prob"] = (
        (n * local_cong + prior * global_cong) / (n + prior)
    ).astype("float32")

    out["hist_s4_prob"] = (
        (n * local_s4 + prior * global_s4) / (n + prior)
    ).astype("float32")

    probs = []
    for idx, s in enumerate(STATUS_VALUES):
        cnt = base_raw[f"cnt_s{s}"].values.astype("float64")
        p = (cnt + prior * global_probs[idx]) / (n + prior)
        probs.append(p)

    P = np.vstack(probs).T
    P = np.clip(P, 1e-8, 1.0)

    out["hist_entropy"] = (-np.sum(P * np.log(P), axis=1)).astype("float32")

    mean = base_raw["link_time_sum"].values.astype("float64") / n_safe
    ex2 = base_raw["link_time_sumsq"].values.astype("float64") / n_safe
    var = np.maximum(ex2 - mean * mean, 0.0)

    out["hist_link_time_mean"] = mean.astype("float32")
    out["hist_link_time_std"] = np.sqrt(var).astype("float32")
    out["hist_n_log"] = np.log1p(out["hist_n"].values.astype("float64")).astype("float32")
    out["hist_missing"] = np.zeros(len(out), dtype="float32")

    return out


def _add_window_features_streaming(base: pd.DataFrame, windows):
    out = base.copy()

    src = base[
        [
            "link_id",
            "slice_key",
            "hist_n",
            "hist_cong_prob",
            "hist_s4_prob",
            "hist_status_mean",
        ]
    ].copy()

    src["hist_n"] = src["hist_n"].clip(lower=1).astype("float32")

    keys = ["link_id", "slice_key"]
    sum_cols = ["_w", "_cong_w", "_s4_w", "_status_w"]

    for k in windows:
        log.info(f"[HistContext] building window k={k}")

        acc = None

        for offset in range(-k, k + 1):
            part = src.copy()

            part["slice_key"] = (
                (part["slice_key"].astype("int32") + offset) % 288
            ).astype("int16")

            w = part["hist_n"].values.astype("float32")
            part["_w"] = w
            part["_cong_w"] = (part["hist_cong_prob"].values * w).astype("float32")
            part["_s4_w"] = (part["hist_s4_prob"].values * w).astype("float32")
            part["_status_w"] = (part["hist_status_mean"].values * w).astype("float32")

            part = part[keys + sum_cols]

            acc = _merge_sum_acc(acc, part, keys, sum_cols)

            gc.collect()

        denom = np.maximum(acc["_w"].values.astype("float32"), 1e-6)

        acc[f"hist_cong_win{k}"] = (acc["_cong_w"].values / denom).astype("float32")
        acc[f"hist_s4_win{k}"] = (acc["_s4_w"].values / denom).astype("float32")
        acc[f"hist_status_win{k}"] = (acc["_status_w"].values / denom).astype("float32")

        keep = [
            "link_id",
            "slice_key",
            f"hist_cong_win{k}",
            f"hist_s4_win{k}",
            f"hist_status_win{k}",
        ]

        out = out.merge(acc[keep], on=["link_id", "slice_key"], how="left")

        del acc
        gc.collect()

    del src
    gc.collect()

    return out


def build_target_context(target_day, train_days, test_days, windows, leave_day_out=True, overwrite=False):
    target_day = str(target_day).zfill(2)
    train_days = [str(d).zfill(2) for d in train_days]
    test_days = {str(d).zfill(2) for d in test_days}

    if leave_day_out and target_day in train_days:
        source_days = [d for d in train_days if d != target_day]
    else:
        source_days = list(train_days)

    if target_day in test_days:
        source_days = list(train_days)

    if not source_days:
        raise RuntimeError(f"No source days for target_day={target_day}")

    log.info("=" * 70)
    log.info(f"[HistContext] target_day={target_day}")
    log.info(f"[HistContext] source_days={source_days}")
    log.info(f"[HistContext] windows={windows}")
    log.info(f"[HistContext] bucketed=True, n_buckets={HIST_N_BUCKETS}")
    log.info("=" * 70)

    total_rows = 0

    for b in range(HIST_N_BUCKETS):
        out_path = _target_bucket_path(target_day, b)

        if os.path.exists(out_path) and not overwrite:
            log.info(f"[HistContext] target={target_day} bucket={b:03d}: exists, skip.")
            continue

        log.info("-" * 70)
        log.info(f"[HistContext] target={target_day}: building bucket {b:03d}/{HIST_N_BUCKETS - 1:03d}")

        base_raw = _sum_daily_aggs_for_bucket(source_days, b)

        if base_raw is None or len(base_raw) == 0:
            empty = pd.DataFrame(columns=["hist_key"] + HIST_FEATURE_COLS)
            empty.to_pickle(out_path)
            log.warning(f"[HistContext] target={target_day} bucket={b:03d}: empty")
            continue

        hist = _finalize_base_features(base_raw)
        del base_raw
        gc.collect()

        hist = _add_window_features_streaming(hist, windows)

        hist["hist_key"] = (
            hist["link_id"].astype("int64") * 288
            + hist["slice_key"].astype("int64")
        )

        for col in HIST_FEATURE_COLS:
            if col not in hist.columns:
                hist[col] = np.nan

        hist = hist[["hist_key"] + HIST_FEATURE_COLS].drop_duplicates("hist_key")

        for col in HIST_FEATURE_COLS:
            hist[col] = hist[col].astype("float32")

        hist.to_pickle(out_path)
        total_rows += len(hist)

        log.info(
            f"[HistContext] target={target_day} bucket={b:03d}: "
            f"saved rows={len(hist):,}"
        )

        del hist
        gc.collect()

    meta = {
        "target_day": target_day,
        "source_days": source_days,
        "windows": windows,
        "version": HIST_CONTEXT_VERSION,
        "n_buckets": HIST_N_BUCKETS,
        "target_min_count": HIST_TARGET_MIN_COUNT,
        "total_rows": int(total_rows),
        "feature_cols": HIST_FEATURE_COLS,
    }

    with open(_target_bucket_meta_path(target_day), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info(
        f"[HistContext] target_day={target_day} bucketed context complete. "
        f"total_rows={total_rows:,}"
    )
def split_daily_base_to_buckets(day: str, overwrite: bool = False):
    day = str(day).zfill(2)
    src_path = _daily_base_path(day)

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Missing daily base aggregate: {src_path}")

    # 如果 bucket 已存在且不覆盖，直接跳过
    exists = all(os.path.exists(_daily_bucket_path(day, b)) for b in range(HIST_N_BUCKETS))
    if exists and not overwrite:
        log.info(f"[HistBucket] day={day}: all buckets exist, skip.")
        return

    log.info("=" * 70)
    log.info(f"[HistBucket] Splitting daily base day={day} into {HIST_N_BUCKETS} buckets")
    log.info("=" * 70)

    df = pd.read_pickle(src_path)
    df["bucket"] = (df["link_id"].astype("int64") % HIST_N_BUCKETS).astype("int16")

    for b in range(HIST_N_BUCKETS):
        part = df[df["bucket"] == b].drop(columns=["bucket"]).copy()
        part.to_pickle(_daily_bucket_path(day, b))
        log.info(f"[HistBucket] day={day} bucket={b:03d} rows={len(part):,}")
        del part
        gc.collect()

    del df
    gc.collect()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--build-daily", action="store_true")
    parser.add_argument("--build-target", action="store_true")
    parser.add_argument("--split-daily", action="store_true")
    parser.add_argument("--day", type=str, default="")
    parser.add_argument("--target-day", type=str, default="")
    parser.add_argument("--train-days", type=str, default="")
    parser.add_argument("--test-days", type=str, default="")
    parser.add_argument("--windows", type=str, default="")
    parser.add_argument("--leave-day-out", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    windows = _parse_windows(args.windows)

    if args.build_daily:
        if not args.day:
            raise ValueError("--build-daily requires --day")
        build_daily_base_agg(args.day, overwrite=args.overwrite)
        return

    if args.build_target:
        if not args.target_day:
            raise ValueError("--build-target requires --target-day")

        build_target_context(
            target_day=args.target_day,
            train_days=_parse_days(args.train_days),
            test_days=_parse_days(args.test_days),
            windows=windows,
            leave_day_out=args.leave_day_out,
            overwrite=args.overwrite,
        )
        return
    if args.split_daily:
        if not args.day:
            raise ValueError("--split-daily requires --day")
        split_daily_base_to_buckets(args.day, overwrite=args.overwrite)
        return
    raise ValueError("Specify either --build-daily or --build-target")


if __name__ == "__main__":
    main()