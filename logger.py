"""
logger.py — 统一日志模块
"""
import logging
import sys
from datetime import datetime
from config import LOG_DIR
import os


def get_logger(name: str = "pipeline") -> logging.Logger:
    """获取带控制台+文件双输出的 logger（全局单例）。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 已初始化，直接复用

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, f"run_{ts}.log"),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger