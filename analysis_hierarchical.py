"""
analysis_hierarchical.py
------------------------
层级模型专用评估模块

功能:
  1. 层级 Head 独立性能评估 (head_cong / head_severe / head_mild 各自的 AUROC、AUPRC、Recall)
  2. 概率层 vs 决策层对比分析 (模型概率质量 vs 成本矩阵决策效果)
  3. 成本矩阵网格搜索 (自动寻找最优 safety-discrimination tradeoff)
  4. 综合论文报告输出

用法:
  python analysis_hierarchical.py

  或在代码中:
  from analysis_hierarchical import run_full_analysis
  run_full_analysis()
"""

import os
import gc
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    accuracy_score, f1_score,
    confusion_matrix, classification_report,
)
from tqdm.auto import tqdm

from config import (
    MODEL_DIR, NUM_CLASSES, STATUS_CLASSES,
    WRC_BATCH_SIZE, WRC_MAX_SEQ_LEN,
    TRAIN_DAYS, TEST_DAYS,
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

# 门控特征索引
GATE_FEATURE_NAMES = [
    "pos_ratio", "cum_travel_time",
    "sin_arr_slice", "cos_arr_slice",
    "downstream_cross_time",
]
GATE_INDICES = [STAGE1_FEATURE_COLS.index(n) for n in GATE_FEATURE_NAMES]


# ================================================================
# 1. 从 shard 收集原始概率和标签
# ================================================================

def collect_predictions_from_shards(
    model,
    shard_infos: list,
    device=None,
    batch_size: int = 512,
    max_seq_len: int = 200,
) -> dict:
    """
    在指定 shard 上收集模型的所有原始输出。

    返回 dict:
      y_true:       np.array (N,)     真实标签 0-indexed
      p_cong:       np.array (N,)     P(拥堵)
      p_severe:     np.array (N,)     P(极端|拥堵)
      p_mild:       np.array (N,)     P(缓行|非拥堵)
      probs:        np.array (N, 4)   恢复的四类概率
      p4:           np.array (N,)     P(s4) = p_cong * p_severe
    """
    if device is None:
        device = DEVICE

    model.eval()
    model.to(device)

    all_y      = []
    all_pcong  = []
    all_psev   = []
    all_pmild  = []
    all_probs  = []
    all_p4     = []

    pf = ShardPrefetcher(shard_infos)

    while True:
        _, shard_obj = pf.next()
        if shard_obj is None:
            break

        ds = SingleShardSequenceDataset(shard_obj, max_seq_len=max_seq_len)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        with torch.no_grad():
            for X_b, y_b, lens_b in loader:
                X_b    = X_b.to(device, non_blocking=True)
                y_b    = y_b.to(device, non_blocking=True)
                lens_b = lens_b.to(device)
                gate_b = X_b[:, :, GATE_INDICES]

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    lc, ls, lm = model(X_b, lens_b, gate_b)

                    p_cong_t   = torch.sigmoid(lc)
                    p_severe_t = torch.sigmoid(ls)
                    p_mild_t   = torch.sigmoid(lm)

                    probs_t, _, p4_t = model.get_hierarchical_probs(lc, ls, lm)

                for b in range(len(lens_b)):
                    L = lens_b[b].item()

                    y_slice = y_b[b, :L].cpu().numpy()
                    valid   = y_slice >= 0

                    all_y.append(y_slice[valid])
                    all_pcong.append(p_cong_t[b, :L].cpu().numpy()[valid])
                    all_psev.append(p_severe_t[b, :L].cpu().numpy()[valid])
                    all_pmild.append(p_mild_t[b, :L].cpu().numpy()[valid])
                    all_probs.append(probs_t[b, :L].cpu().numpy()[valid])
                    all_p4.append(p4_t[b, :L].cpu().numpy()[valid])

        del ds, loader, shard_obj
        gc.collect()

    pf.close()

    return {
        "y_true":   np.concatenate(all_y),
        "p_cong":   np.concatenate(all_pcong),
        "p_severe": np.concatenate(all_psev),
        "p_mild":   np.concatenate(all_pmild),
        "probs":    np.concatenate(all_probs, axis=0),
        "p4":       np.concatenate(all_p4),
    }


# ================================================================
# 2. 层级 Head 独立性能评估
# ================================================================

def evaluate_hierarchical_heads(data: dict) -> dict:
    """
    分别评估三个层级 head 的概率质量。

    Head 1 (cong):   y >= 2 → positive
    Head 2a (severe): 在 y>=2 子集上, y==3 → positive
    Head 2b (mild):   在 y<2 子集上, y==1 → positive
    """
    y = data["y_true"]
    results = {}

    log.info("\n" + "=" * 70)
    log.info("HIERARCHICAL HEAD EVALUATION (Probability Quality)")
    log.info("=" * 70)

    # ---- Head 1: 拥堵门 ----
    log.info("\n--- Head 1: Congestion Gate (y>=2 vs y<2) ---")
    target_cong = (y >= 2).astype(int)
    p_cong = data["p_cong"]

    if len(np.unique(target_cong)) > 1:
        auroc_cong = roc_auc_score(target_cong, p_cong)
        auprc_cong = average_precision_score(target_cong, p_cong)

        # 找最优阈值（F1 最大化）
        prec, rec, thresholds = precision_recall_curve(target_cong, p_cong)
        f1_arr = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = np.argmax(f1_arr)
        best_thresh_cong = thresholds[min(best_idx, len(thresholds) - 1)]
        best_f1_cong = f1_arr[best_idx]

        # 高召回阈值（Recall >= 0.85）
        high_recall_thresh = None
        for i in range(len(rec) - 1, -1, -1):
            if rec[i] >= 0.85:
                high_recall_thresh = thresholds[min(i, len(thresholds) - 1)]
                break

        log.info(f"  AUROC:  {auroc_cong:.4f}")
        log.info(f"  AUPRC:  {auprc_cong:.4f}")
        log.info(f"  Best F1 threshold: {best_thresh_cong:.4f} → F1={best_f1_cong:.4f}")
        log.info(f"  High-recall (≥0.85) threshold: {high_recall_thresh:.4f}" if high_recall_thresh else "  High-recall threshold: N/A")
        log.info(f"  Positive rate: {target_cong.mean():.4f}")

        results["cong"] = {
            "auroc": auroc_cong, "auprc": auprc_cong,
            "best_f1_thresh": float(best_thresh_cong),
            "best_f1": float(best_f1_cong),
            "high_recall_thresh": float(high_recall_thresh) if high_recall_thresh else None,
        }
    else:
        log.warning("  Only one class present in cong target, skip AUROC/AUPRC")
        results["cong"] = None

    # ---- Head 2a: 极端拥堵（仅在 y>=2 子集上）----
    log.info("\n--- Head 2a: Severe (s4 vs s3, within congested) ---")
    cong_mask = y >= 2
    if cong_mask.sum() > 0:
        y_cong = y[cong_mask]
        target_severe = (y_cong == 3).astype(int)
        p_severe = data["p_severe"][cong_mask]

        if len(np.unique(target_severe)) > 1:
            auroc_sev = roc_auc_score(target_severe, p_severe)
            auprc_sev = average_precision_score(target_severe, p_severe)

            log.info(f"  AUROC:  {auroc_sev:.4f}")
            log.info(f"  AUPRC:  {auprc_sev:.4f}")
            log.info(f"  N samples: {cong_mask.sum():,} (s3={int((y_cong==2).sum()):,}, s4={int((y_cong==3).sum()):,})")

            results["severe"] = {"auroc": auroc_sev, "auprc": auprc_sev}
        else:
            log.warning("  Only one class in severe target")
            results["severe"] = None
    else:
        log.warning("  No congested samples")
        results["severe"] = None

    # ---- Head 2b: 缓行（仅在 y<2 子集上）----
    log.info("\n--- Head 2b: Mild (s2 vs s1, within non-congested) ---")
    non_cong_mask = y < 2
    if non_cong_mask.sum() > 0:
        y_non_cong = y[non_cong_mask]
        target_mild = (y_non_cong == 1).astype(int)
        p_mild = data["p_mild"][non_cong_mask]

        if len(np.unique(target_mild)) > 1:
            auroc_mild = roc_auc_score(target_mild, p_mild)
            auprc_mild = average_precision_score(target_mild, p_mild)

            log.info(f"  AUROC:  {auroc_mild:.4f}")
            log.info(f"  AUPRC:  {auprc_mild:.4f}")
            log.info(f"  N samples: {non_cong_mask.sum():,} (s1={int((y_non_cong==0).sum()):,}, s2={int((y_non_cong==1).sum()):,})")

            results["mild"] = {"auroc": auroc_mild, "auprc": auprc_mild}
        else:
            log.warning("  Only one class in mild target")
            results["mild"] = None
    else:
        log.warning("  No non-congested samples")
        results["mild"] = None

    # ---- P(s4) 全局 AUROC ----
    log.info("\n--- P(s4) Global Risk Score ---")
    target_s4 = (y == 3).astype(int)
    if len(np.unique(target_s4)) > 1:
        auroc_p4 = roc_auc_score(target_s4, data["p4"])
        auprc_p4 = average_precision_score(target_s4, data["p4"])
        log.info(f"  P(s4) AUROC: {auroc_p4:.4f}")
        log.info(f"  P(s4) AUPRC: {auprc_p4:.4f}")
        results["p4_global"] = {"auroc": auroc_p4, "auprc": auprc_p4}
    else:
        results["p4_global"] = None

    return results


# ================================================================
# 3. 成本矩阵网格搜索
# ================================================================

def cost_matrix_grid_search(data: dict, n_candidates: int = 12) -> pd.DataFrame:
    """
    搜索不同成本矩阵配置下的 DMR / macro_f1 / accuracy tradeoff。

    核心参数: C[s4→s1] 和 C[s3→s1] 的相对大小。
    """
    y = data["y_true"]
    probs = data["probs"]

    log.info("\n" + "=" * 70)
    log.info("COST MATRIX GRID SEARCH")
    log.info("=" * 70)

    # 搜索空间: 控制 s3→s1 和 s4→s1 的漏判惩罚
    s3_miss_costs = [4, 6, 8, 10]
    s4_miss_costs = [8, 12, 15, 20]

    results = []

    for c_s3_s1, c_s4_s1 in itertools.product(s3_miss_costs, s4_miss_costs):
        if c_s4_s1 <= c_s3_s1:
            continue

        C = np.array([
            [0,         1,         2,         3        ],
            [1,         0,         2,         3        ],
            [c_s3_s1,   c_s3_s1-2, 0,         1        ],
            [c_s4_s1,   c_s4_s1-3, 3,         0        ],
        ], dtype=np.float32)

        C_tensor = torch.tensor(C, dtype=torch.float32)
        probs_tensor = torch.tensor(probs, dtype=torch.float32)

        expected_cost = torch.einsum("nk,kj->nj", probs_tensor, C_tensor)
        pred_cls = expected_cost.argmin(dim=-1).numpy()

        acc = accuracy_score(y, pred_cls)
        mf1 = f1_score(y, pred_cls, average="macro")

        high = y >= 2
        dmr = float((pred_cls < 2)[high].sum() / max(high.sum(), 1))

        s4a = y == 3
        s4u = float((pred_cls < 3)[s4a].sum() / max(s4a.sum(), 1))

        cm = confusion_matrix(y, pred_cls, labels=[0, 1, 2, 3])
        rec = [cm[i, i] / max(cm[i].sum(), 1) for i in range(4)]

        # 计算平均期望代价
        avg_cost = float(expected_cost.min(dim=-1).values.mean())

        results.append({
            "C_s3_s1": c_s3_s1,
            "C_s4_s1": c_s4_s1,
            "accuracy": round(acc, 4),
            "macro_f1": round(mf1, 4),
            "DMR": round(dmr, 4),
            "s4_under": round(s4u, 4),
            "recall_s1": round(rec[0], 4),
            "recall_s2": round(rec[1], 4),
            "recall_s3": round(rec[2], 4),
            "recall_s4": round(rec[3], 4),
            "avg_cost": round(avg_cost, 4),
        })

    df = pd.DataFrame(results)
    df = df.sort_values("DMR")

    log.info("\nGrid Search Results (sorted by DMR):")
    log.info(df.to_string(index=False))

    # 推荐: DMR < 0.15 中 macro_f1 最高
    safe = df[df["DMR"] < 0.15]
    if len(safe) > 0:
        best = safe.loc[safe["macro_f1"].idxmax()]
        log.info(f"\n★ Recommended (DMR<0.15): C_s3={int(best['C_s3_s1'])}, C_s4={int(best['C_s4_s1'])}")
        log.info(f"  DMR={best['DMR']:.4f} | macro_f1={best['macro_f1']:.4f} | "
                 f"s4_recall={best['recall_s4']:.4f}")
    else:
        best = df.iloc[0]
        log.info(f"\n★ Lowest DMR config: C_s3={int(best['C_s3_s1'])}, C_s4={int(best['C_s4_s1'])}")

    return df


# ================================================================
# 4. 概率层 vs 决策层对比
# ================================================================

def compare_probability_vs_decision(data: dict, cost_matrix: np.ndarray = None):
    """
    对比:
      A) 纯概率 argmax 决策
      B) 当前成本矩阵决策
      C) 概率阈值决策 (仅用 p_cong 阈值)
    """
    y = data["y_true"]
    probs = data["probs"]
    p_cong = data["p_cong"]

    log.info("\n" + "=" * 70)
    log.info("PROBABILITY vs DECISION LAYER COMPARISON")
    log.info("=" * 70)

    high = y >= 2

    # ---- A: 纯 argmax ----
    pred_argmax = probs.argmax(axis=1)
    acc_a = accuracy_score(y, pred_argmax)
    mf1_a = f1_score(y, pred_argmax, average="macro")
    dmr_a = float((pred_argmax < 2)[high].sum() / max(high.sum(), 1))

    log.info(f"\n[A] Pure Argmax:")
    log.info(f"    Accuracy={acc_a:.4f} | Macro_F1={mf1_a:.4f} | DMR={dmr_a:.4f}")

    # ---- B: 成本矩阵 ----
    if cost_matrix is None:
        from stage1_deep import _COST_MATRIX_NP
        cost_matrix = _COST_MATRIX_NP

    C_tensor = torch.tensor(cost_matrix, dtype=torch.float32)
    probs_tensor = torch.tensor(probs, dtype=torch.float32)
    expected_cost = torch.einsum("nk,kj->nj", probs_tensor, C_tensor)
    pred_cost = expected_cost.argmin(dim=-1).numpy()

    acc_b = accuracy_score(y, pred_cost)
    mf1_b = f1_score(y, pred_cost, average="macro")
    dmr_b = float((pred_cost < 2)[high].sum() / max(high.sum(), 1))

    log.info(f"\n[B] Cost Matrix Decision:")
    log.info(f"    Accuracy={acc_b:.4f} | Macro_F1={mf1_b:.4f} | DMR={dmr_b:.4f}")

    # ---- C: P(cong) 阈值扫描 ----
    log.info(f"\n[C] P(cong) Threshold Scan:")
    log.info(f"    {'Threshold':>10s} {'Accuracy':>10s} {'Macro_F1':>10s} {'DMR':>10s} {'s4_recall':>10s}")

    for tau in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        pred_thresh = np.zeros(len(y), dtype=int)
        cong_mask = p_cong >= tau

        # 非拥堵区: s1/s2 按 argmax
        pred_thresh[~cong_mask] = probs[~cong_mask][:, :2].argmax(axis=1)
        # 拥堵区: s3/s4 按 argmax
        pred_thresh[cong_mask] = probs[cong_mask][:, 2:].argmax(axis=1) + 2

        acc_c = accuracy_score(y, pred_thresh)
        mf1_c = f1_score(y, pred_thresh, average="macro")
        dmr_c = float((pred_thresh < 2)[high].sum() / max(high.sum(), 1))
        s4_rec = float((pred_thresh == 3)[y == 3].sum() / max((y == 3).sum(), 1))

        log.info(f"    {tau:>10.2f} {acc_c:>10.4f} {mf1_c:>10.4f} {dmr_c:>10.4f} {s4_rec:>10.4f}")

    # ---- 对比总结 ----
    log.info(f"\n  Summary:")
    log.info(f"    Argmax      → DMR={dmr_a:.4f}, Macro_F1={mf1_a:.4f}")
    log.info(f"    Cost Matrix → DMR={dmr_b:.4f}, Macro_F1={mf1_b:.4f}")
    log.info(f"    Delta       → DMR {dmr_b - dmr_a:+.4f}, Macro_F1 {mf1_b - mf1_a:+.4f}")

    if dmr_b < dmr_a:
        log.info(f"    → Cost matrix reduces DMR by {(dmr_a - dmr_b) / dmr_a * 100:.1f}%")
    else:
        log.info(f"    → Cost matrix did not reduce DMR")


# ================================================================
# 5. 综合论文报告
# ================================================================

def generate_paper_report(data: dict, head_results: dict):
    """
    输出论文可直接引用的统计量。
    """
    y = data["y_true"]

    log.info("\n" + "=" * 70)
    log.info("KEY NUMBERS FOR PAPER")
    log.info("=" * 70)

    # 数据集统计
    counts = np.bincount(y, minlength=NUM_CLASSES)
    total = counts.sum()
    log.info(f"\nDataset:")
    log.info(f"  Total samples: {total:,}")
    for i, c in enumerate(counts):
        log.info(f"  s{i+1}: {c:,} ({c/total*100:.2f}%)")

    # 层级 head 性能
    log.info(f"\nHierarchical Head AUROC:")
    if head_results.get("cong"):
        log.info(f"  head_cong (拥堵门):     AUROC={head_results['cong']['auroc']:.4f}")
    if head_results.get("severe"):
        log.info(f"  head_severe (极端门):   AUROC={head_results['severe']['auroc']:.4f}")
    if head_results.get("mild"):
        log.info(f"  head_mild (缓行门):     AUROC={head_results['mild']['auroc']:.4f}")
    if head_results.get("p4_global"):
        log.info(f"  P(s4) global:           AUROC={head_results['p4_global']['auroc']:.4f}")

    # 最优阈值
    if head_results.get("cong") and head_results["cong"].get("best_f1_thresh"):
        log.info(f"\nOptimal P(cong) threshold:")
        log.info(f"  Best F1:         τ={head_results['cong']['best_f1_thresh']:.4f} → F1={head_results['cong']['best_f1']:.4f}")
        if head_results["cong"].get("high_recall_thresh"):
            log.info(f"  High Recall≥0.85: τ={head_results['cong']['high_recall_thresh']:.4f}")

    log.info("\n" + "=" * 70)


# ================================================================
# 6. 主函数
# ================================================================

def run_full_analysis(
    model=None,
    model_path: str = None,
    use_val: bool = True,
    batch_size: int = WRC_BATCH_SIZE,
    max_seq_len: int = WRC_MAX_SEQ_LEN,
):
    """
    完整评估流程。

    Parameters
    ----------
    model : 已加载的模型（可选）
    model_path : 模型文件路径（如果 model 为 None）
    use_val : True=用验证集, False=用测试集
    """
    import glob

    # 加载模型
    if model is None:
        if model_path is None:
            pattern = os.path.join(MODEL_DIR, "stage1_hier_*.pt")
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No hierarchical model found: {pattern}")
            model_path = files[-1]

        from stage1_deep import load_wrc_model
        model = load_wrc_model(model_path, device=DEVICE)

    # 获取 shard
    train_shards, val_shards, test_shards = split_train_val_shards()

    if use_val:
        eval_shards = val_shards
        eval_name = "Validation"
    else:
        eval_shards = test_shards if test_shards else val_shards
        eval_name = "Test" if test_shards else "Validation (no test shards)"

    log.info(f"\n{'='*70}")
    log.info(f"FULL HIERARCHICAL ANALYSIS on {eval_name} set")
    log.info(f"  Shards: {len(eval_shards)}")
    log.info(f"{'='*70}")

    # Step 1: 收集原始概率
    log.info("\nStep 1: Collecting raw predictions ...")
    data = collect_predictions_from_shards(
        model, eval_shards, device=DEVICE,
        batch_size=batch_size, max_seq_len=max_seq_len,
    )
    log.info(f"  Collected {len(data['y_true']):,} samples")

    # Step 2: 层级 Head 独立评估
    log.info("\nStep 2: Evaluating hierarchical heads ...")
    head_results = evaluate_hierarchical_heads(data)

    # Step 3: 概率层 vs 决策层对比
    log.info("\nStep 3: Probability vs Decision comparison ...")
    compare_probability_vs_decision(data)

    # Step 4: 成本矩阵网格搜索
    log.info("\nStep 4: Cost matrix grid search ...")
    grid_df = cost_matrix_grid_search(data)

    # Step 5: 论文报告
    log.info("\nStep 5: Paper report ...")
    generate_paper_report(data, head_results)

    # 保存网格搜索结果
    output_path = "cost_matrix_grid_results.csv"
    grid_df.to_csv(output_path, index=False)
    log.info(f"\nGrid search results saved: {output_path}")

    return {
        "data": data,
        "head_results": head_results,
        "grid_df": grid_df,
    }


# ================================================================
# 入口
# ================================================================

if __name__ == "__main__":
    run_full_analysis(use_val=True)