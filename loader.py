"""
loader.py — 支持逐天加载 + 批量轻量加载
"""

import os
import glob
import gc
import pandas as pd
from config import (
    DATA_DIR, TOPO_PATH, SPLIT_SUBDIRS,
    HEAD_DTYPES, LINK_DTYPES, CROSS_DTYPES, COLUMN_RENAME,
)


def _detect_sep(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()
    if "\t" in first_line:
        return "\t"
    elif "," in first_line:
        return ","
    return r"\s+"


def _read_single_file(filepath: str) -> pd.DataFrame:
    sep = _detect_sep(filepath)
    df = pd.read_csv(filepath, sep=sep, engine="python" if sep == r"\s+" else "c")
    df.columns = df.columns.str.strip()
    df = df.rename(columns=COLUMN_RENAME)
    return df


def _glob_day_files(file_type: str, day_id: str) -> list[str]:
    """
    兼容两种命名：
      1) head_01_1
      2) head01_1.csv
    """
    subdir = SPLIT_SUBDIRS[file_type]
    folder = os.path.join(DATA_DIR, subdir)

    patterns = [
        os.path.join(folder, f"{file_type}_{day_id}_*"),
        os.path.join(folder, f"{file_type}{day_id}_*"),
        os.path.join(folder, f"{file_type}_{int(day_id)}_*"),
        os.path.join(folder, f"{file_type}{int(day_id)}_*"),
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"No files found for {file_type} day={day_id}\n"
            f"  Tried patterns:\n    " + "\n    ".join(patterns)
        )

    return files


def load_split_files(file_type: str, day_id: str) -> pd.DataFrame:
    dtype_map = {"head": HEAD_DTYPES, "link": LINK_DTYPES, "cross": CROSS_DTYPES}[file_type]
    files = _glob_day_files(file_type, day_id)
    if not files:
        raise FileNotFoundError(f"No files: {file_type}_{day_id}_*")
    frames = [_read_single_file(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    for col, dt in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dt)
            except (ValueError, TypeError):
                pass
    return df


def load_day(day_id: str):
    """加载单天的 head/link/cross 数据。"""
    h = load_split_files("head", day_id)
    l = load_split_files("link", day_id)
    c = load_split_files("cross", day_id)
    h["day"] = day_id
    l["day"] = day_id
    c["day"] = day_id
    return h, l, c


def load_heads_only(day_list: list[str]) -> pd.DataFrame:
    """仅加载多天的 head 数据（内存轻量，用于 Stage 3）。"""
    frames = []
    for day in day_list:
        try:
            h = load_split_files("head", day)
            h["day"] = day
            frames.append(h)
        except FileNotFoundError:
            print(f"[loader] ⚠ day {day} head not found, skipping")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_topology() -> dict[int, list[int]]:
    topo = {}
    with open(TOPO_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            src = int(parts[0])
            dsts = [int(x) for x in parts[1].split(",") if x] if len(parts) > 1 else []
            topo[src] = dsts
    print(f"[loader] topology: {len(topo)} links")
    return topo