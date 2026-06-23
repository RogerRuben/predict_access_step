"""
config.py — 改进版
"""
import os
import re
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "output")
DATA_DIR = os.path.join(BASE_DIR, "data_split")
TOPO_PATH = os.path.join(BASE_DIR, "nextlinks", "nextlinks.txt")

SPLIT_SUBDIRS = {
    "head":  "head_split",
    "link":  "link_split",
    "cross": "cross_split",
}

# MISSING_DAYS = {"03","08","09","10"}
# TRAIN_DAYS = [f"{d:02d}" for d in range(1, 11) if f"{d:02d}" not in MISSING_DAYS]
# TEST_DAYS  = ["13"]
# MISSING_DAYS = {"03","08","09","10"}
# TRAIN_DAYS = [f"{d:02d}" for d in range(15, 16) if f"{d:02d}" not in MISSING_DAYS]
# TEST_DAYS  = ["31"]

# 日期配置 —— 保守版本，排除 Day 07（内存问题）和 Day 08/09/10（缺失）
TRAIN_DAYS = ["01", "02", "04", "05", "06", "11"]
TEST_DAYS = ["13"]

# 如果后续确认 Day 07 问题修复，可加回
# 如果确认 Day 31 数据存在，可切换测试日

# Stage 1
STATUS_CLASSES = [1, 2, 3, 4]
NUM_CLASSES = len(STATUS_CLASSES)
TRAIN_SAMPLE_RATE = 0.7
BATCH_SIZE_ORDERS = 1500

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": NUM_CLASSES,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,          # 63→127: 更强的拟合能力
    "learning_rate": 0.03,      # 0.05→0.03: 配合更多轮次
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 100,   # 防过拟合
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
    "is_unbalance": True,       # ★ 内置类别不均衡处理
}
LGBM_NUM_ROUNDS = 800           # 400→800
LGBM_EARLY_STOPPING = 60       # 40→60

# Stage 2
CONGESTION_PROB_THRESHOLD = 0.5
ENTROPY_HIGH_THRESHOLD = 1.0
CROSS_TIME_QUANTILE = 0.80
DEGREE_BASELINE = 2
DEGREE_HIGH = 4
CLUSTER_WINDOW_K = 5

# Stage 3
DRIVER_SHRINKAGE_PRIOR = 10
Y_TILDE_WINSORIZE = (0.01, 0.99)   # ★ 截断极端值

# ★ 新增：Stage 3 使用 float32 还是 float64
STAGE3_USE_FLOAT64 = False  # False=float32, 节省内存；True=float64, 数值更稳定

# 模型保存
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)
# 校验保存
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_figs")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 校验保存
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_result")
os.makedirs(RESULTS_DIR, exist_ok=True)
# 日志
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# dtype
HEAD_DTYPES = {
    "order_id": "int32", "ata": "int32", "distance": "int32",
    "simple_eta": "int32", "driver_id": "int32", "slice_id": "int16",
}
LINK_DTYPES = {
    "order_id": "int32", "link_id": "int32",
    "link_time": "float32", "link_ratio": "float32",
    "link_current_status": "int8", "link_arrival_status": "int8",
}
CROSS_DTYPES = {
    "order_id": "int32", "cross_id": "object", "cross_time": "float32",
}

COLUMN_RENAME = {
    "order id": "order_id", "cross id": "cross_id", "cross time": "cross_time",
    "link id": "link_id", "link time": "link_time", "link ratio": "link_ratio",
    "link current status": "link_current_status",
    "link arrival status": "link_arrival_status",
    "simple eta": "simple_eta", "driver id": "driver_id", "slice id": "slice_id",
}
# 缓存
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Stage 1 模型选择

STAGE1_MODEL = "wdr"   # 或 "wrc"
# WRC 超参数
WRC_HIDDEN_DIM = 128
WRC_NUM_LAYERS = 2
WRC_BATCH_SIZE = 512
WRC_EPOCHS = 5
WRC_LR = 8e-4
WRC_MAX_SEQ_LEN = 200
# DataLoader
num_workers = 2

# ============================================================
# Stage 1 WDR 安全损失与安全决策
# ============================================================

# loss 权重
# LOSS_LAMBDA_RISK = 0.50
# LOSS_LAMBDA_FN   = 0.80
# LOSS_LAMBDA_S4   = 0.60
FOCAL_GAMMA      = 2.0
LOSS_LAMBDA_RISK = 0.20
LOSS_LAMBDA_FN   = 0.30
LOSS_LAMBDA_S4   = 0.10
# 安全阈值（初始值，可后续网格搜索）
# SAFE_TAU_CONG       = 0.22   # P(s3)+P(s4) 超过此值，则按拥堵处理
# SAFE_TAU_RISK       = 0.35   # 风险头概率超过此值，则按拥堵处理
# SAFE_TAU_S4         = 0.12   # P(s4) 超过此值，则直接按 s4
# SAFE_TAU_RISK_HIGH  = 0.70   # 风险头极高，则直接按 s4
SAFE_TAU_CONG      = 0.35
SAFE_TAU_RISK      = 0.55
SAFE_TAU_S4        = 0.45
SAFE_TAU_RISK_HIGH = 0.90
# 是否启用安全决策规则
SAFE_DECISION = True
# ============================================================
# K折交叉验证配置（Stage 1）
# ============================================================
USE_KFOLD = False          # 是否启用 K折交叉验证（默认关闭，仅在最终调优时开启）
KFOLD_SPLITS = 5           # 折数
KFOLD_SEED = 42            # 随机种子

# ============================================================
# Stage 1: Tau 搜索开关
# ============================================================
SKIP_TAU_SEARCH = False          # True: 跳过搜索，使用默认值；False: 执行搜索
DEFAULT_TAU = 0.55              # 默认 tau 值
DEFAULT_TEMPERATURE = 1.0       # 默认温度值

# 运行模式
RUN_PROFILE = "smoke"   # smoke / debug / full

if RUN_PROFILE == "smoke":
    WRC_EPOCHS = 2
    MAX_TRAIN_SHARDS = 12
    MAX_VAL_SHARDS = 4
    MAX_EVAL_BATCHES = 200
elif RUN_PROFILE == "debug":
    WRC_EPOCHS = 3
    MAX_TRAIN_SHARDS = 24
    MAX_VAL_SHARDS = 8
    MAX_EVAL_BATCHES = 500
else:  # full
    WRC_EPOCHS = 5
    MAX_TRAIN_SHARDS = None
    MAX_VAL_SHARDS = None
    MAX_EVAL_BATCHES = None

    # ============================================================
    # Stage 1 训练控制
    # ============================================================
    RESUME_STAGE1 = False  # False: 从头训练, True: 从 checkpoint 恢复
    MAX_VAL_SHARDS = None  # None: 使用全部 val shards, 整数: 限制数量

    # Stage 1 验证日（用于 day-holdout 验证）
    # 如果为空，则使用随机 shard split
    STAGE1_VAL_DAYS = ["11"]  # 使用 Day 11 作为验证日