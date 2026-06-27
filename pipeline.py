"""
pipeline.py — 完整版
集成功能：
1. Stage 1 训练 / 加载
2. ★ 新增 Stage 1 test evaluation（train/test 模式都可打印）
3. Stage 2 缓存 + 订单级难度特征
4. Stage 3 难度锚定与评估
"""

import gc
import os
import glob
import hashlib
import pandas as pd
import numpy as np
from config import DATA_DIR
from config import (
    TRAIN_DAYS, TEST_DAYS,
    CROSS_TIME_QUANTILE,
    TRAIN_SAMPLE_RATE,
    BATCH_SIZE_ORDERS,
    MODEL_DIR, CACHE_DIR,
    STAGE1_MODEL,
    WRC_HIDDEN_DIM, WRC_NUM_LAYERS, WRC_BATCH_SIZE,
    WRC_EPOCHS, WRC_LR, WRC_MAX_SEQ_LEN, FIGURES_DIR, RESULTS_DIR,
)

from loader import load_day, load_split_files, load_topology
from feature_eng import (
    build_stage1_features_batch,
    STAGE1_FEATURE_COLS, STAGE1_TARGET,
)

from stage2_assess import (
    compute_deterministic,
    compute_uncertainty,
    compute_safety_risk,
    compute_night,
    compute_markov_features,
    compute_weather_features,
    STAGE2_ALL_COLS,
)
from logger import get_logger
from config import USE_KFOLD, KFOLD_SPLITS, KFOLD_SEED
from stage1_dataset import split_kfold_shards
from stage3_anchor import DifficultyModelXGB

log = get_logger()
from config import LOG_DIR

# ============================================================
# 工具函数
# ============================================================

import subprocess
import sys

def _run_stage2_day_isolated(day, s1_model_path, mode, ct_threshold, cross_global_mean):
    """在独立子进程中运行单天 Stage 2"""

    day = str(day).zfill(2)

    # ---- 父进程先检查 cache 是否已存在 ----
    if _stage2_cache_exists(day, s1_model_path, ct_threshold, cross_global_mean):
        cache_key = _make_cache_key(
            day=day,
            s1_path=s1_model_path,
            ct_thresh=ct_threshold,
            cross_mean=cross_global_mean,
        )
        log.info(f"[Isolated] Day {day}: cache exists, skip subprocess. cache_key={cache_key}")
        return

    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "compute_stage2_day.py",
    )

    cmd = [
        sys.executable,
        script_path,
        "--day", day,
        "--stage1-model-path", os.path.abspath(s1_model_path),
        "--mode", mode,
        "--ct-threshold", repr(float(ct_threshold)),
        "--cross-global-mean", repr(float(cross_global_mean)),
    ]

    log.info(f"[Isolated] Launching Stage 2 for Day {day}: {' '.join(cmd)}")

    # ---- 子进程日志写入单独文件 ----
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"stage2_day_{day}_{mode}.log")

    with open(log_path, "w", encoding="utf-8") as fh:
        result = subprocess.run(
            cmd,
            check=False,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode != 0:
        tail = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                tail = "".join(fh.readlines()[-80:])
        except Exception as read_err:
            tail = f"<failed to read child log: {read_err}>"

        log.error(
            f"[Isolated] Day {day} failed with exit code {result.returncode}. "
            f"See {log_path}\n"
            f"---- child log tail ----\n{tail}\n"
            f"---- end child log tail ----"
        )
        raise RuntimeError(f"Stage 2 Day {day} failed in isolated process")

    # ---- 子进程完成后再次检查 cache ----
    cache_key = _make_cache_key(
        day=day,
        s1_path=s1_model_path,
        ct_thresh=ct_threshold,
        cross_mean=cross_global_mean,
    )
    s2_cache, head_cache = _cache_paths(cache_key)

    s2_exists = os.path.exists(s2_cache)
    head_exists = os.path.exists(head_cache)

    log.info(
        f"[Isolated] Day {day}: parent verifies cache_key={cache_key} | "
        f"s2_exists={s2_exists} | head_exists={head_exists}"
    )

    if not (s2_exists and head_exists):
        tail = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                tail = "".join(fh.readlines()[-100:])
        except Exception as read_err:
            tail = f"<failed to read child log: {read_err}>"

        log.error(
            f"[Isolated] Day {day} completed but cache not found.\n"
            f"Expected cache_key={cache_key}\n"
            f"Expected s2={s2_cache}\n"
            f"Expected head={head_cache}\n"
            f"Child log={log_path}\n"
            f"---- child log tail ----\n{tail}\n"
            f"---- end child log tail ----"
        )
        raise RuntimeError(f"Stage 2 Day {day} completed but cache missing")

    log.info(f"[Isolated] Day {day} completed successfully. cache_key = {cache_key}")

def ensure_columns(df, required_cols, default_values=None):
    """
    确保 DataFrame 包含所有 required_cols，缺失则用默认值填充。

    Args:
        df: pd.DataFrame
        required_cols: list of column names
        default_values: dict, {col_name: default_value}

    Returns:
        pd.DataFrame (修改后的副本)
    """
    if df is None:
        df = pd.DataFrame()
    if default_values is None:
        default_values = {}

    df = df.copy()
    for col in required_cols:
        if col not in df.columns:
            default_val = default_values.get(col, 0.0)
            df[col] = default_val
            log.debug(f"Added missing column: {col} (default={default_val})")
    return df


def ensure_stage2_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    强健补全 Stage 2 的所有列。
    如果 df 为 None 或空，返回完整的空 DataFrame。
    """
    from stage2_assess import STAGE2_ALL_COLS as S2_COLS

    if df is None or len(df) == 0:
        df = pd.DataFrame()

    df = df.copy()
    for col in S2_COLS:
        if col not in df.columns:
            df[col] = 0.0
            log.debug(f"Added missing Stage 2 column: {col}")

    # 确保 order_id 和 day 存在
    if 'order_id' not in df.columns:
        df['order_id'] = 0
    if 'day' not in df.columns:
        df['day'] = 0

    return df


def _find_latest_model(prefix: str, ext: str) -> str:
    pattern = os.path.join(MODEL_DIR, f"{prefix}*{ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No saved model: {pattern}")
    latest = files[-1]
    log.info(f"Found model: {latest}")
    return latest


# pipeline.py

import hashlib
import json
import os
from functools import lru_cache

@lru_cache(maxsize=16)
def _get_model_fingerprint(path: str) -> str:
    if not path or not os.path.exists(path):
        return "no_model"

    try:
        import torch
        ckpt = torch.load(path, map_location="cpu")

        h = hashlib.sha1()

        meta = {
            "best_tau": ckpt.get("best_tau", None),
            "temperature": ckpt.get("temperature", None),
            "dense_dim": ckpt.get("dense_dim", None),
            "hidden_dim": ckpt.get("hidden_dim", None),
            "num_layers": ckpt.get("num_layers", None),
            "model_type": ckpt.get("model_type", None),
        }
        h.update(json.dumps(meta, sort_keys=True, default=str).encode("utf-8"))

        state = (
            ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
            or ckpt.get("model")
        )

        if isinstance(state, dict):
            for k in sorted(state.keys()):
                v = state[k]
                if hasattr(v, "detach"):
                    arr = v.detach().cpu().contiguous().numpy()
                    h.update(k.encode("utf-8"))
                    h.update(str(arr.shape).encode("utf-8"))
                    h.update(str(arr.dtype).encode("utf-8"))
                    h.update(arr.tobytes())
                else:
                    h.update(k.encode("utf-8"))
                    h.update(str(v).encode("utf-8"))

        return h.hexdigest()[:12]

    except Exception:
        # fallback: 文件内容 hash
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:12]

def _make_cache_key(
    day,
    s1_path=None,
    ct_thresh=None,
    cross_mean=None,
    **kwargs,
):
    if s1_path is None:
        s1_path = kwargs.get("s1_model_path", None)
    if ct_thresh is None:
        ct_thresh = kwargs.get("ct_threshold", None)
    if cross_mean is None:
        cross_mean = kwargs.get("cross_global_mean", None)

    ct_val = float(ct_thresh) if ct_thresh is not None else -1.0
    cross_val = float(cross_mean) if cross_mean is not None else -1.0

    model_fp = _get_model_fingerprint(s1_path)

    raw = (
        f"day={str(day).zfill(2)}"
        f"|model={STAGE1_MODEL}"
        f"|model_fp={model_fp}"
        f"|ct={ct_val:.4f}"
        f"|cross_mean={cross_val:.4f}"
        f"|schema=v15_stage2_isolated"
    )

    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _cache_paths(cache_key):
    return (
        os.path.join(CACHE_DIR, f"s2_{cache_key}.pkl"),
        os.path.join(CACHE_DIR, f"head_{cache_key}.pkl"),
    )

def _run_difficulty_validation_split(df, split_name: str):
    """
    对 train/test 分别运行 difficulty validation，并单独保存 JSON。
    注意 generate_validation_report 内部可能仍保存默认 validation_report.json；
    这里额外保存 split-specific report，避免 train/test 混淆。
    """
    import json
    import os
    import pandas as pd
    from evaluation.difficulty_validation import (
        generate_validation_report,
        extreme_weather_validation,
    )

    split_name = str(split_name).lower()

    log.info("\n" + "=" * 70)
    log.info(f"RUNNING DIFFICULTY VALIDATION [{split_name.upper()}]")
    log.info("=" * 70)

    if df is None or df.empty:
        log.warning(f"[Stage3-{split_name}] empty dataframe, skip validation.")
        return {
            "split": split_name,
            "n": 0,
            "report": None,
            "extreme_weather": None,
        }

    report = generate_validation_report(df)
    extreme_results = extreme_weather_validation(df)

    out = {
        "split": split_name,
        "n": int(len(df)),
        "report": report,
        "extreme_weather": extreme_results,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"validation_report_{split_name}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    log.info(f"[Stage3-{split_name}] validation report saved: {out_path}")

    return out

def _stage2_cache_exists(day, s1_model_path, ct_threshold, cross_global_mean):
    cache_key = _make_cache_key(
        day=day,
        s1_path=s1_model_path,
        ct_thresh=ct_threshold,
        cross_mean=cross_global_mean,
    )
    s2_cache, head_cache = _cache_paths(cache_key)
    return os.path.exists(s2_cache) and os.path.exists(head_cache)

def assert_no_bad_stage2_names(cols):
    """检查 Stage 2 列名是否包含历史拼写错误"""
    bad = [c for c in cols if ("topop" in c or "topp" in c)]
    if bad:
        raise ValueError(
            f"Bad Stage2 feature names found: {bad}. "
            "Use topo, not topop/topp. Check fix_stage2_column_aliases()."
        )

def fix_stage2_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    df = df.copy()

    alias_pairs = {
        "D9_topop_complex": "D9_topo_complex",
        "D9_topp_complex": "D9_topo_complex",
        "D10_topop_cluster": "D10_topo_cluster",
        "D10_topp_cluster": "D10_topo_cluster",
    }

    for old, new in alias_pairs.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
            log.debug(f"Stage2 column alias: {old} -> {new}")
        elif old in df.columns and new in df.columns:
            df[new] = df[new].where(df[new].notna(), df[old])

    # ★ 删除旧列
    drop_cols = [c for c in alias_pairs.keys() if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def ensure_stage2_columns(df: pd.DataFrame) -> pd.DataFrame:
    from stage2_assess import STAGE2_ALL_COLS as S2_COLS

    if df is None or len(df) == 0:
        df = pd.DataFrame()

    # ★ 先做 alias
    df = fix_stage2_column_aliases(df)

    df = df.copy()
    for col in S2_COLS:
        if col not in df.columns:
            df[col] = 0.0
            log.debug(f"Added missing Stage 2 column: {col}")

    if "order_id" not in df.columns:
        df["order_id"] = 0
    if "day" not in df.columns:
        df["day"] = 0

    # ★ 再做一次 alias 兜底
    df = fix_stage2_column_aliases(df)
    return df
# ============================================================
# Stage 2: 单天处理（带缓存）
# ============================================================

def _process_day_stage2(
    day,
    topo=None,
    model=None,
    ct_threshold=None,
    cross_global_mean=None,
    s1_model_path="",
):
    cache_key = _make_cache_key(
        day=day,
        s1_path=s1_model_path,
        ct_thresh=ct_threshold,
        cross_mean=cross_global_mean,
    )
    s2_cache, head_cache = _cache_paths(cache_key)

    if os.path.exists(s2_cache) and os.path.exists(head_cache):
        log.info(f"  Day {day}: cache HIT ({cache_key})")
        s2_full = pd.read_pickle(s2_cache)
        head_full = pd.read_pickle(head_cache)
        # ★ 修 alias + 补列
        s2_full = fix_stage2_column_aliases(s2_full)
        s2_full = ensure_stage2_columns(s2_full)
        return s2_full, head_full

    log.info(f"  Day {day}: cache MISS, computing ...")

    # ★ 先 load_day（内存最重的操作，先执行）
    try:
        head, link, cross = load_day(day)
    except Exception as e:
        log.error(f"  Day {day}: load_day failed: {e}")
        empty_s2 = pd.DataFrame(columns=["order_id", "day"] + STAGE2_ALL_COLS)
        empty_head = pd.DataFrame(columns=["order_id", "day", "driver_id", "slice_id", "ata", "simple_eta"])
        return empty_s2, empty_head
    if topo is None:
        topo = load_topology()

    if model is None:
        if not s1_model_path or not os.path.exists(s1_model_path):
            raise FileNotFoundError(f"Stage 1 model not found: {s1_model_path}")
        if STAGE1_MODEL in ("wrc", "wdr", "hierarchical", "hier", "ordinal", "dualhead", "triple", "cascade"):
            from stage1_deep import load_wrc_model
            model = load_wrc_model(s1_model_path)
        else:
            from stage1_predict import load_stage1_model
            model = load_stage1_model(s1_model_path)

    # ============================================================
    # 步骤1：合并天气数据（如果 head 中没有）
    # ============================================================
    if 'weather' not in head.columns:
        weather_path = os.path.join(DATA_DIR, 'weather.csv')
        if os.path.exists(weather_path):
            weather_df = pd.read_csv(weather_path)
            weather_df['day'] = weather_df['date'].astype(str)
            head = head.merge(weather_df[['day', 'weather', 'hightemp', 'lowtemp']],
                              on='day', how='left')
            log.info(f"  Day {day}: weather data merged")
        else:
            log.warning(f"  Weather file not found at {weather_path}. Using defaults.")
            head['weather'] = 'cloudy'
            head['hightemp'] = 25
            head['lowtemp'] = 15
    else:
        head['weather'] = head['weather'].fillna('cloudy')
        head['hightemp'] = head['hightemp'].fillna(25)
        head['lowtemp'] = head['lowtemp'].fillna(15)

    unique_orders = link["order_id"].unique()
    s2_parts = []
    head_parts = []

    # 选择预测函数
    if STAGE1_MODEL in ("wrc", "wdr", "hierarchical", "hier", "ordinal", "dualhead", "triple", "cascade"):
        from stage1_deep import predict_proba_wrc
        predict_fn = predict_proba_wrc
    else:
        from stage1_predict import predict_proba
        predict_fn = predict_proba

    for i in range(0, len(unique_orders), BATCH_SIZE_ORDERS):
        batch_orders = unique_orders[i:i + BATCH_SIZE_ORDERS]
        link_b = link[link["order_id"].isin(batch_orders)]
        head_b = head[head["order_id"].isin(batch_orders)]
        cross_b = cross[cross["order_id"].isin(batch_orders)]

        feat = build_stage1_features_batch(
            link_b, head_b, cross_b, topo,
            keep_extra_for_stage2=True,
        )

        # Stage 1 预测
        pred = predict_fn(model, feat)
        # pred 已经包含了 feat 的所有列 + 预测列，直接使用
        link_with_pred = pred

        # 计算各维度特征
        det = compute_deterministic(
            link_with_pred, cross_b, topo, ct_threshold, cross_global_mean
        )
        unc = compute_uncertainty(link_with_pred)
        risk = compute_safety_risk(link_with_pred)
        night = compute_night(head_b)
        markov = compute_markov_features(link_with_pred)
        weather = compute_weather_features(head_b)

        # 合并所有订单级特征
        merged = det.merge(unc, on=['order_id', 'day'], how='outer')
        merged = merged.merge(risk, on=['order_id', 'day'], how='outer')
        merged = merged.merge(night, on=['order_id', 'day'], how='outer')
        merged = merged.merge(markov, on=['order_id', 'day'], how='outer')
        merged = merged.merge(weather, on=['order_id', 'day'], how='outer')

        s2_parts.append(merged)
        print(f"head_b columns: {head_b.columns.tolist()}")
        head_slim_batch = head_b[['order_id', 'day', 'driver_id', 'slice_id', 'ata', 'simple_eta', 'distance']].copy()
        head_parts.append(head_slim_batch)

        # 每个 batch 后清理
        del link_b, head_b, cross_b
        del feat, pred, link_with_pred
        del det, unc, risk, night, markov, weather, merged

        if (i // BATCH_SIZE_ORDERS) % 5 == 0:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # 合并所有批次
    if s2_parts:
        s2_full = pd.concat(s2_parts, ignore_index=True)
        head_full = pd.concat(head_parts, ignore_index=True)
    else:
        s2_full = pd.DataFrame()
        head_full = pd.DataFrame()

    # ============================================================
    # 步骤2：强制确保所有 Stage 2 列都存在
    # ============================================================
    from stage2_assess import STAGE2_ALL_COLS as S2_COLS

    # 确保 order_id 和 day 存在
    if 'order_id' not in s2_full.columns:
        s2_full['order_id'] = 0
    if 'day' not in s2_full.columns:
        s2_full['day'] = 0

    # 补全所有 Stage 2 特征列
    for col in S2_COLS:
        if col not in s2_full.columns:
            s2_full[col] = 0.0
            log.debug(f"  Added missing column: {col} (filled with 0)")

    # 确保 head_full 包含必要列
    required_head_cols = ['order_id', 'day', 'driver_id', 'slice_id', 'ata', 'simple_eta', 'distance']
    for col in required_head_cols:
        if col not in head_full.columns:
            head_full[col] = 0 if col in ['order_id', 'day', 'driver_id', 'slice_id'] else 0.0

    # 保存缓存
    os.makedirs(CACHE_DIR, exist_ok=True)
    s2_full.to_pickle(s2_cache)
    head_full.to_pickle(head_cache)
    log.info(f"  Day {day}: cached {len(s2_full)} orders, {len(s2_full.columns)} columns")

    return s2_full, head_full


# ============================================================
# Stage 1: 训练
# ============================================================

def _run_stage1_train(topo):
    from config import USE_KFOLD, KFOLD_SPLITS, KFOLD_SEED
    from stage1_deep import train_wrc_from_shards, save_wrc_model
    import torch
    import gc

    if USE_KFOLD:
        log.info("=" * 70)
        log.info(f"STAGE 1: K-Fold Cross Validation (splits={KFOLD_SPLITS})")
        log.info("=" * 70)

        best_models = []
        for fold_idx, (train_shards, val_shards) in enumerate(split_kfold_shards(
                n_splits=KFOLD_SPLITS, seed=KFOLD_SEED
        )):
            log.info(f"\n{'=' * 60}")
            log.info(f"Training Fold {fold_idx + 1}/{KFOLD_SPLITS}")
            log.info(f"Train shards: {len(train_shards)}, Val shards: {len(val_shards)}")
            log.info(f"{'=' * 60}")

            from config import RESUME_STAGE1

            model = train_wrc_from_shards(
                train_shards=train_shards,
                val_shards=val_shards,
                resume=RESUME_STAGE1,  # ★ 从 config 读取
            )
            save_path = save_wrc_model(model, tag=f"fold{fold_idx + 1}")
            best_models.append(save_path)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log.info(f"\nK-Fold complete. Best models saved: {best_models}")
        from stage1_deep import load_wrc_model
        final_model = load_wrc_model(best_models[-1])
        final_path = best_models[-1]
        return final_model, final_path

    else:
        if STAGE1_MODEL in ("wrc", "wdr", "hierarchical", "hier", "ordinal", "dualhead", "triple", "cascade"):
            from stage1_deep import train_wrc_from_shards, save_wrc_model
            from config import RESUME_STAGE1
            log.info("=" * 70)
            log.info("STAGE 1: Single Training (no K-Fold)")
            log.info("=" * 70)
            model = train_wrc_from_shards(
                hidden_dim=WRC_HIDDEN_DIM,
                num_layers=WRC_NUM_LAYERS,
                batch_size=WRC_BATCH_SIZE,
                epochs=WRC_EPOCHS,
                lr=WRC_LR,
                max_seq_len=WRC_MAX_SEQ_LEN,
                resume=RESUME_STAGE1,
            )
            model_path = save_wrc_model(model, tag="v1")
            return model, model_path

        elif STAGE1_MODEL == "lgbm":
            from stage1_predict import train_model, save_stage1_model

            X_parts, y_parts = [], []
            for day in TRAIN_DAYS:
                try:
                    log.info(f"Extracting day {day} ...")
                    head, link, cross = load_day(day)
                    unique_orders = link["order_id"].unique()
                    day_samples = 0

                    for i in range(0, len(unique_orders), BATCH_SIZE_ORDERS):
                        batch_orders = unique_orders[i:i + BATCH_SIZE_ORDERS]
                        feat = build_stage1_features_batch(
                            link[link["order_id"].isin(batch_orders)],
                            head[head["order_id"].isin(batch_orders)],
                            cross[cross["order_id"].isin(batch_orders)],
                            topo,
                            keep_extra_for_stage2=False,
                        )
                        valid = feat[feat[STAGE1_TARGET].isin([1, 2, 3, 4])]
                        if len(valid) > 0:
                            X_parts.append(valid[STAGE1_FEATURE_COLS].values.astype("float32"))
                            y_parts.append((valid[STAGE1_TARGET].values - 1).astype("int32"))
                            day_samples += len(valid)
                        del feat, valid
                        gc.collect()

                    del head, link, cross
                    gc.collect()
                    log.info(f"  Day {day}: {day_samples:,} samples")
                except FileNotFoundError:
                    log.warning(f"Day {day} missing, skipped.")

            if not X_parts:
                raise RuntimeError("No training data for LightGBM")

            X_all = np.concatenate(X_parts)
            y_all = np.concatenate(y_parts)
            del X_parts, y_parts
            gc.collect()

            if TRAIN_SAMPLE_RATE < 1.0:
                n = int(len(y_all) * TRAIN_SAMPLE_RATE)
                idx = np.random.RandomState(42).choice(len(y_all), n, replace=False)
                X_all, y_all = X_all[idx], y_all[idx]

            model = train_model(X_all, y_all)
            model_path = save_stage1_model(model, tag="v3")
            del X_all, y_all
            gc.collect()
            return model, model_path

        else:
            raise ValueError(
                f"Unknown STAGE1_MODEL='{STAGE1_MODEL}'. "
                f"Supported: 'wdr', 'wrc', 'hier', 'hierarchical', "
                f"'ordinal', 'dualhead', 'triple', 'cascade', 'lgbm'."
            )


# ============================================================
# ★ 新增：Stage 1 单独评估
# ============================================================

def _build_stage1_eval_criterion(stage1_module, train_shards, device):
    """
    根据 stage1_deep.py 里实际存在的类，动态构建评估 criterion。
    兼容当前所有版本。
    """
    if hasattr(stage1_module, "estimate_weights"):
        result = stage1_module.estimate_weights(
            train_shards, min(5, len(train_shards))
        )
        if len(result) == 5:
            pw_q1, pw_q2, pw_q3, cls_alpha, _ = result
        else:
            pw_q1, pw_q2, pw_q3, cls_alpha = result
    else:
        pw_q1, pw_q2, pw_q3, cls_alpha = 1.0, 2.0, 4.0, np.ones(4, dtype=np.float32)

    if hasattr(stage1_module, "OrdinalPrimaryLoss"):
        criterion = stage1_module.OrdinalPrimaryLoss(
            pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
            cls_alpha=cls_alpha,
        ).to(device)
        return criterion
    if hasattr(stage1_module, "CascadeLoss"):
        try:
            criterion = stage1_module.CascadeLoss(
                pw_q1=pw_q1,
                pw_q2=pw_q2,
                pw_q3=pw_q3,
                cls_alpha=cls_alpha,
                gamma_q1=0.0,
                gamma_q2=2.0,
                gamma_q3=2.0,
                gamma_cls=2.0,
                gamma_bnd=1.5,
            ).to(device)
        except TypeError:
            criterion = stage1_module.CascadeLoss(
                pw_q1=pw_q1,
                pw_q2=pw_q2,
                pw_q3=pw_q3,
                cls_alpha=cls_alpha,
            ).to(device)
        return criterion
    if hasattr(stage1_module, "TripleExpertLoss"):
        criterion = stage1_module.TripleExpertLoss(
            pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
            cls_alpha=cls_alpha,
            gamma_q1=0.0, gamma_q2=2.0, gamma_q3=2.0,
            gamma_cls=2.0, gamma_bnd=1.5,
            lambda_cls=1.0, lambda_bnd=0.8, lambda_cons=0.0,
        ).to(device)
        return criterion

    if hasattr(stage1_module, "HybridOrdinalClassificationLoss"):
        criterion = stage1_module.HybridOrdinalClassificationLoss(
            pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
            cls_alpha=cls_alpha,
        ).to(device)
        return criterion

    if hasattr(stage1_module, "FocalOrdinalLoss"):
        criterion = stage1_module.FocalOrdinalLoss(
            pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
        ).to(device)
        return criterion

    if hasattr(stage1_module, "CumulativeOrdinalLoss"):
        criterion = stage1_module.CumulativeOrdinalLoss(
            pw_q1=pw_q1, pw_q2=pw_q2, pw_q3=pw_q3,
        ).to(device)
        return criterion

    raise RuntimeError(
        "Cannot build Stage 1 evaluation criterion. "
        "No recognized loss class found in stage1_deep.py. "
        f"Available: {[x for x in dir(stage1_module) if 'Loss' in x]}"
    )


def _run_stage1_test_evaluation(s1_model):
    """
    在 pipeline 中单独评估 Stage 1，打印 confusion matrix / recall / DMR。
    """
    if STAGE1_MODEL not in ("wrc", "wdr", "hierarchical", "hier", "ordinal", "dualhead", "triple", "cascade"):
        return

    log.info("=" * 70)
    log.info("STAGE 1: Test Evaluation")
    log.info("=" * 70)

    try:
        import stage1_deep as stage1_module

        train_shards, val_shards, test_shards = stage1_module.split_train_val_shards()

        eval_shards = test_shards if len(test_shards) > 0 else val_shards
        eval_name = "TEST" if len(test_shards) > 0 else "VAL(fallback)"

        criterion = _build_stage1_eval_criterion(stage1_module, train_shards, stage1_module.DEVICE)

        tau = float(s1_model.best_tau.item()) if hasattr(s1_model, "best_tau") else 0.55

        log.info(f"Evaluating Stage 1 on {eval_name} shards: {len(eval_shards)} | tau={tau:.2f}")

        stage1_eval = stage1_module.evaluate_on_shards(
            s1_model,
            eval_shards,
            device=stage1_module.DEVICE,
            criterion=criterion,
            batch_size=WRC_BATCH_SIZE * 2,
            max_seq_len=WRC_MAX_SEQ_LEN,
            verbose=True,
            tau=tau,
        )

        log.info(
            f"[Stage1-{eval_name}] "
            f"Integrated mF1={stage1_eval.get('macro_f1', stage1_eval.get('ord_macro_f1', 0)):.4f} "
            f"DMR={stage1_eval['dangerous_miss']:.4f} | "
            f"s4u={stage1_eval['s4_underestimate']:.4f}"
        )
        log.info(
            f"[Stage1-{eval_name}] Integrated Recall: "
            f"{[round(x, 4) for x in stage1_eval['recall_per_class']]}"
        )
        if "cls_recall" in stage1_eval:
            log.info(
                f"[Stage1-{eval_name}] Classifier Recall: "
                f"{[round(x, 4) for x in stage1_eval['cls_recall']]}"
            )
        log.info(f"[Stage1-{eval_name}] Reject rate: {stage1_eval['reject_rate']:.4f}")

    except Exception as e:
        log.warning(f"Stage 1 evaluation skipped due to error: {repr(e)}")


# ============================================================
# 全局统计量
# ============================================================

def _compute_global_cross_stats(train_days):
    log.info("Computing global cross_time statistics (Train set only) ...")
    ct_vals = []
    for day in train_days:
        try:
            c = load_split_files("cross", day)
            ct_vals.append(c["cross_time"].values)
            del c
        except FileNotFoundError:
            pass

    if ct_vals:
        all_ct = np.concatenate(ct_vals)
        ct_threshold = float(np.percentile(all_ct, CROSS_TIME_QUANTILE * 100))
        cross_global_mean = float(np.mean(all_ct))
        del all_ct
    else:
        ct_threshold = 30.0
        cross_global_mean = 25.0

    del ct_vals
    gc.collect()

    log.info(f"  cross_time P{int(CROSS_TIME_QUANTILE * 100)} = {ct_threshold:.2f}s")
    log.info(f"  cross_time global_mean = {cross_global_mean:.2f}s")
    return ct_threshold, cross_global_mean

#stage3def

def _dedup_keep_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _make_stage3_v2_input(df, s2_cols):
    """
    Save a clean Stage3-v2 input table:
    - keeps Stage2 features and service outcome columns
    - excludes old Stage3 output scores to avoid leakage
    """
    out = df.copy()

    required_supervision = ["ata", "simple_eta"]
    missing = [c for c in required_supervision if c not in out.columns]
    if missing:
        raise ValueError(
            f"Cannot create Stage3-v2 input. Missing supervision columns: {missing}"
        )

    out["ata"] = pd.to_numeric(out["ata"], errors="coerce")
    out["simple_eta"] = pd.to_numeric(out["simple_eta"], errors="coerce")

    eta_safe = out["simple_eta"].clip(lower=1e-6)
    out["ata_over_eta"] = out["ata"] / eta_safe
    out["eta_delay"] = np.maximum(out["ata_over_eta"] - 1.0, 0.0)

    base_cols = [
        "order_id",
        "day",
        "ata",
        "simple_eta",
        "ata_over_eta",
        "eta_delay",
        "distance",
        "driver_id",
        "slice_id",
    ]

    # Keep weather columns if they exist, even if not included in S2_COLS.
    extra_cols = [
        "weather_severity",
        "temp_avg",
        "temp_range",
        "is_extreme_weather",
        "is_high_temp",
        "is_low_temp",
    ]

    keep_cols = []
    keep_cols += [c for c in base_cols if c in out.columns]
    keep_cols += [c for c in s2_cols if c in out.columns]
    keep_cols += [c for c in extra_cols if c in out.columns]
    keep_cols = _dedup_keep_order(keep_cols)

    if len([c for c in s2_cols if c in out.columns]) < 8:
        log.warning(
            f"[Stage3-v2 input] Only found "
            f"{len([c for c in s2_cols if c in out.columns])} Stage2 feature columns. "
            f"Please check S2_COLS and train_final/test_final columns."
        )

    return out[keep_cols].copy()

# ============================================================
# 主流程
# ============================================================

def run_pipeline(mode: str = "train"):
    # ★ 初始化所有 DataFrame（避免 test 模式引用未定义变量）
    train_s2 = pd.DataFrame()
    train_head_slim = pd.DataFrame()
    test_s2 = pd.DataFrame()
    test_head_slim = pd.DataFrame()
    train_final = None
    test_final = None

    topo = load_topology()

    topo = load_topology()

    # ---- Stage 1: train or load ----
    if mode == "train":
        s1_model, s1_model_path = _run_stage1_train(topo)
    else:
        if STAGE1_MODEL in ("wrc", "wdr", "hierarchical", "hier", "ordinal", "dualhead", "triple", "cascade"):
            import stage1_deep as stage1_module

            for prefix in ["stage1_cascade", "stage1_ordpri", "stage1_triple", "stage1_dualhead", "stage1_ordinal"]:
                try:
                    s1_model_path = _find_latest_model(prefix, ".pt")
                    break
                except FileNotFoundError:
                    continue
            else:
                raise FileNotFoundError("No Stage 1 deep model found in saved_models/")

            s1_model = stage1_module.load_wrc_model(s1_model_path)
        else:
            from stage1_predict import load_stage1_model
            s1_model_path = _find_latest_model("stage1_lgbm", ".txt")
            s1_model = load_stage1_model(s1_model_path)

    # ---- ★ 新增：Stage 1 test evaluation ----
    _run_stage1_test_evaluation(s1_model)

    # ---- Stage 2 前释放父进程内存 ----
    log.info("[Memory] Releasing parent Stage1 model/topology before isolated Stage2.")

    s1_model_return = None

    try:
        del s1_model
    except NameError:
        pass

    try:
        del topo
    except NameError:
        pass

    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- Stage 2/3 ----
    # 先计算全局交叉路口统计
    ct_threshold, cross_global_mean = _compute_global_cross_stats(TRAIN_DAYS)

    # Stage 2 train
    train_final = None
    test_final = None  # ★ 新增这一行
    if mode == "train":
        log.info("=" * 70)
        log.info("STAGE 2: Difficulty Assessment (Train) - Isolated per day")
        log.info("=" * 70)

        failed_train_days = []
        for day in TRAIN_DAYS:
            try:
                _run_stage2_day_isolated(
                    day=day,
                    s1_model_path=s1_model_path,
                    mode="train",
                    ct_threshold=ct_threshold,
                    cross_global_mean=cross_global_mean,
                )
            except Exception as e:
                log.warning(f"Train day {day} Stage 2 failed: {e}")
                failed_train_days.append(str(day).zfill(2))

        if failed_train_days:
            log.warning(f"Failed train days: {failed_train_days}")

    # ---- Stage 2 test ----
    log.info("=" * 70)
    log.info("STAGE 2: Difficulty Assessment (Test) - Isolated per day")
    log.info("=" * 70)

    failed_test_days = []
    for day in TEST_DAYS:
        try:
            _run_stage2_day_isolated(
                day=day,
                s1_model_path=s1_model_path,
                mode="test",
                ct_threshold=ct_threshold,
                cross_global_mean=cross_global_mean,
            )
        except Exception as e:
            log.error(f"Test day {day} Stage 2 failed: {e}")
            failed_test_days.append(str(day).zfill(2))

    if failed_test_days:
        raise RuntimeError(f"Test days failed: {failed_test_days}")

    # ---- 从 cache 读取所有已处理 day ----
    train_s2_list = []
    train_head_list = []
    for day in TRAIN_DAYS:
        cache_key = _make_cache_key(
            day=day,
            s1_path=s1_model_path,
            ct_thresh=ct_threshold,
            cross_mean=cross_global_mean,
        )
        log.info(f"Day {day}: parent expects cache_key={cache_key}")
        s2_cache, head_cache = _cache_paths(cache_key)
        import os 
        if os.path.exists(s2_cache) and os.path.exists(head_cache):
            s2 = pd.read_pickle(s2_cache)
            head = pd.read_pickle(head_cache)
            s2 = fix_stage2_column_aliases(s2)
            s2 = ensure_stage2_columns(s2)
            train_s2_list.append(s2)
            train_head_list.append(head)
            log.info(f"Day {day}: loaded from cache")
        else:
            log.warning(f"Day {day}: cache not found, skipped.")

    test_s2_list = []
    test_head_list = []
    for day in TEST_DAYS:
        cache_key = _make_cache_key(
            day=day,
            s1_path=s1_model_path,
            ct_thresh=ct_threshold,
            cross_mean=cross_global_mean,
        )
        s2_cache, head_cache = _cache_paths(cache_key)
        if os.path.exists(s2_cache) and os.path.exists(head_cache):
            s2 = pd.read_pickle(s2_cache)
            head = pd.read_pickle(head_cache)
            s2 = fix_stage2_column_aliases(s2)
            s2 = ensure_stage2_columns(s2)
            test_s2_list.append(s2)
            test_head_list.append(head)
            log.info(f"Day {day}: loaded from cache")
        else:
            log.error(f"Day {day}: cache not found")
            raise RuntimeError(f"Test day {day} cache missing")

    # ---- 合并 ----
    train_s2 = pd.concat(train_s2_list, ignore_index=True) if train_s2_list else pd.DataFrame()
    train_head_slim = pd.concat(train_head_list, ignore_index=True) if train_head_list else pd.DataFrame()
    test_s2 = pd.concat(test_s2_list, ignore_index=True) if test_s2_list else pd.DataFrame()
    test_head_slim = pd.concat(test_head_list, ignore_index=True) if test_head_list else pd.DataFrame()

    # ---- Stage 3 前硬检查 ----
    if mode == "train" and train_s2.empty:
        raise RuntimeError("train_s2 is empty. Cannot train Stage 3.")

    if test_s2.empty:
        raise RuntimeError("test_s2 is empty. Cannot run Stage 3.")
    # ============================================================
    # Stage 3: 使用新的 XGBoost 模型 + 自动验证
    # ============================================================
    # Stage 3: 使用新的 XGBoost 模型 + 自动验证
    # ============================================================
    stage3_output_cols = [
        "difficulty",
        "grade",
        "bti_score",
        "safety_score",
        "eta_risk_score",
        "safety_risk_score",
        "uncertainty_score",
        "av_unsuitability_score",
        "av_reject_prob_proxy",
    ]

    log.info("=" * 70)
    log.info("STAGE 3: Difficulty Anchoring (XGBoost + SHAP)")
    log.info("=" * 70)

    from stage3_anchor import DifficultyModelXGB, STAGE2_ALL_COLS as S2_COLS

    diff_model = DifficultyModelXGB()

    if mode == "train":
        # ---- 1. 准备训练数据 ----
        if train_s2.empty:
            raise ValueError("train_s2 is empty. Cannot proceed with Stage 3 training.")

        train_final = train_s2.merge(train_head_slim, on=['order_id', 'day'], how='inner')
        if train_final.empty:
            train_final = train_s2.merge(train_head_slim, on=['order_id', 'day'], how='outer')

        if train_final.empty:
            raise ValueError("No data after merging train_s2 and train_head_slim.")

        required_cols = ['ata', 'simple_eta', 'distance', 'driver_id', 'slice_id'] + S2_COLS
        default_values = {
            'ata': 300.0,
            'simple_eta': 300.0,
            'distance': 10000.0,
            'driver_id': 0,
            'slice_id': 0,
        }
# 训练监督列不能用默认值伪造
        missing_sup = [c for c in ["ata", "simple_eta"] if c not in train_final.columns]
        if missing_sup:
            raise ValueError(
                f"Stage3 training requires real supervision columns {missing_sup}, "
                f"but they are missing before ensure_columns()."
            )

        train_final = ensure_columns(train_final, required_cols, default_values)
        log.info(f"Stage 3 training data: {len(train_final)} rows, {len(train_final.columns)} columns")

        # 保存 Stage3-v2 的干净输入表：只含 Stage2 features + 真实 outcome，不含旧 Stage3 输出
        train_stage3_v2_input = _make_stage3_v2_input(train_final, S2_COLS)

        # ---- 2. 训练模型 ----
        diff_model.fit(train_final)

        diff_model.save(tag='v_xgb')






        # ---- 3. 预测训练集（用于内部验证） ----
        train_pred = diff_model.predict(train_final)
        train_final['difficulty'] = train_pred['difficulty']
        train_final['grade'] = train_pred['grade']
        train_final['bti_score'] = train_pred['bti_score']
        train_final['safety_score'] = train_pred['safety_score']



        # ---- 4. ★ 预测测试集（这是最终输出） ----
        if test_s2.empty:
            raise ValueError("test_s2 is empty. Cannot predict test set.")

        test_final = test_s2.merge(test_head_slim, on=['order_id', 'day'], how='inner')
        if test_final.empty:
            test_final = test_s2.merge(test_head_slim, on=['order_id', 'day'], how='outer')

        if test_final.empty:
            raise ValueError("No data after merging test_s2 and test_head_slim.")

        missing_sup = [c for c in ["ata", "simple_eta"] if c not in test_final.columns]
        if missing_sup:
            raise ValueError(
                f"Stage3 test validation requires real supervision columns {missing_sup}, "
                f"but they are missing before ensure_columns()."
            )

        test_final = ensure_columns(test_final, required_cols, default_values)

        # 保存 Stage3-v2 的干净输入表：只含 Stage2 features + 真实 outcome，不含旧 Stage3 输出
        test_stage3_v2_input = _make_stage3_v2_input(test_final, S2_COLS)

        test_pred = diff_model.predict(test_final)

        test_final['difficulty'] = test_pred['difficulty']
        test_final['grade'] = test_pred['grade']
        test_final['bti_score'] = test_pred['bti_score']
        test_final['safety_score'] = test_pred['safety_score']

        # 训练集
        train_pred_slim = train_pred[["order_id", "day"] + [c for c in stage3_output_cols if c in train_pred.columns]]
        train_final = train_final.drop(columns=[c for c in stage3_output_cols if c in train_final.columns],
                                       errors="ignore")
        train_final = train_final.merge(train_pred_slim, on=["order_id", "day"], how="left")

        # 测试集
        # 测试集（同理）
        test_pred_slim = test_pred[["order_id", "day"] + [c for c in stage3_output_cols if c in test_pred.columns]]
        test_final = test_final.drop(columns=[c for c in stage3_output_cols if c in test_final.columns],
                                     errors="ignore")
        test_final = test_final.merge(test_pred_slim, on=["order_id", "day"], how="left")


        # ---- 5. 生成 SHAP 解释和验证报告 ----
        diff_model.explain(train_final, output_dir=FIGURES_DIR)

        import json
        import os

        os.makedirs(RESULTS_DIR, exist_ok=True)

        stage3_v2_train_path = os.path.join(RESULTS_DIR, "stage3_v2_input_train.csv")
        stage3_v2_test_path = os.path.join(RESULTS_DIR, "stage3_v2_input_test.csv")

        train_stage3_v2_input.to_csv(stage3_v2_train_path, index=False)
        test_stage3_v2_input.to_csv(stage3_v2_test_path, index=False)

        log.info(f"Stage3-v2 train input saved to {stage3_v2_train_path} "
                f"({len(train_stage3_v2_input)} rows, {len(train_stage3_v2_input.columns)} cols)")
        log.info(f"Stage3-v2 test input saved to {stage3_v2_test_path} "
                f"({len(test_stage3_v2_input)} rows, {len(test_stage3_v2_input.columns)} cols)")

        train_report = _run_difficulty_validation_split(train_final, "train")
        test_report = _run_difficulty_validation_split(test_final, "test")

        full_report = {
            "reports": {
                "train": train_report,
                "test": test_report,
            },
            "thresholds": {
                "safety_threshold": diff_model.safety_threshold,
                "boundary_g1g2": diff_model.boundaries[0] if diff_model.boundaries else None,
                "boundary_g2g3": diff_model.boundaries[1] if diff_model.boundaries else None,
                "entropy_weights": diff_model.entropy_weights,
            },
            "stage3_model": {
                "feature_cols": getattr(diff_model, "feature_cols", None),
                "all_cols_ordered": getattr(diff_model, "all_cols_ordered", None),
            },
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        os.makedirs(RESULTS_DIR, exist_ok=True)
        full_path = os.path.join(RESULTS_DIR, "validation_report_full.json")

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)

        log.info(f"Full validation report saved to {full_path}")

    else:
        # ---- 测试模式：加载模型并预测 ----
        if test_s2.empty:
            raise ValueError("test_s2 is empty. Cannot proceed with Stage 3 testing.")

        model_path = _find_latest_model("stage3_xgb", "pkl")
        diff_model = DifficultyModelXGB.load(model_path)

        test_final = test_s2.merge(test_head_slim, on=['order_id', 'day'], how='inner')
        if test_final.empty:
            test_final = test_s2.merge(test_head_slim, on=['order_id', 'day'], how='outer')

        if test_final.empty:
            raise ValueError("No data after merging test_s2 and test_head_slim.")

        required_cols = ['ata', 'simple_eta', 'distance', 'driver_id', 'slice_id'] + S2_COLS
        default_values = {
            'ata': 300.0,
            'simple_eta': 300.0,
            'distance': 10000.0,
            'driver_id': 0,
            'slice_id': 0,
        }
        test_final = ensure_columns(test_final, required_cols, default_values)

        test_pred = diff_model.predict(test_final)
        test_final['difficulty'] = test_pred['difficulty']
        test_final['grade'] = test_pred['grade']
        test_final['bti_score'] = test_pred['bti_score']
        test_final['safety_score'] = test_pred['safety_score']

        # 训练模式下 train_final 已定义，测试模式下无 train_final
        train_final = None

    # ---- 保存结果 ----
    output_cols = ["order_id", "day"] + [
        c for c in stage3_output_cols if c in test_final.columns
    ]

    test_final[output_cols].to_csv("difficulty_test.csv", index=False)
    log.info(f"Test slim results saved to difficulty_test.csv ({len(test_final)} orders)")

    if train_final is not None:
        train_output_cols = ["order_id", "day"] + [
            c for c in stage3_output_cols if c in train_final.columns
        ]
        train_final[train_output_cols].to_csv("difficulty_train.csv", index=False)
        log.info(f"Train slim results saved to difficulty_train.csv ({len(train_final)} orders)")
        log.info(f"Test results saved to difficulty_test.csv ({len(test_final)} orders)")

    if train_final is not None:
        train_final[output_cols].to_csv('difficulty_train.csv', index=False)
        log.info(f"Train results saved to difficulty_train.csv ({len(train_final)} orders)")

    return s1_model_return, diff_model, train_final, test_final