"""
main.py — 支持 train / test 模式
"""
import sys
from pipeline import run_pipeline
from logger import get_logger

log = get_logger()


def main():
    # 从命令行参数读取模式，默认 train
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    assert mode in ("train", "test"), f"Usage: python main.py [train|test], got '{mode}'"

    log.info(f"Starting pipeline in '{mode}' mode")

    s1_model, diff_model, train_result, test_result = run_pipeline(mode=mode)

    display_cols = [
        "order_id",
        "D1_cong_exposure", "D2_cong_severity", "D3_cong_persist",
        "D4_status_jump", "D5_coupling",
        "U1_path_entropy", "U2_high_unc_ratio",
        "difficulty", "grade",
    ]
    log.info(f"\nTest sample (15 orders):\n"
             f"{test_result[display_cols].head(15).to_string(index=False)}")


if __name__ == "__main__":
    main()