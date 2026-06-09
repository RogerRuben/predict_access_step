"""
config.py
"""
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "output")
DATA_DIR = os.path.join(BASE_DIR, "data_split")
TOPO_PATH = os.path.join(BASE_DIR, "nextlinks", "nextlinks.txt")

SPLIT_SUBDIRS = {
    "head":  "head_split",
    "link":  "link_split",
    "cross": "cross_split",
}

# 数据划分
MISSING_DAYS = {"03"}
TRAIN_DAYS = [f"{d:02d}" for d in range(4, 6) if f"{d:02d}" not in MISSING_DAYS]
TEST_DAYS  = ["16"]

# Stage 1 配置
STATUS_CLASSES = [1, 2, 3, 4]
NUM_CLASSES = len(STATUS_CLASSES)
TRAIN_SAMPLE_RATE = 0.7          # 内存紧张时可调低至 0.5
BATCH_SIZE_ORDERS = 3000         # 每批处理的订单数量（核心参数）

LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": NUM_CLASSES,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}
LGBM_NUM_ROUNDS = 400
LGBM_EARLY_STOPPING = 40

# Stage 2 配置
CONGESTION_PROB_THRESHOLD = 0.5
ENTROPY_HIGH_THRESHOLD = 1.0
CROSS_TIME_QUANTILE = 0.80
DEGREE_BASELINE = 2
DEGREE_HIGH = 4
CLUSTER_WINDOW_K = 5

# Stage 3 配置
DRIVER_SHRINKAGE_PRIOR = 10
LASSO_ALPHA = 0.01

# dtype（严格控制）
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