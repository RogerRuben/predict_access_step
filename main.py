"""
main.py
-------
入口：运行完整 Predict-then-Assess Pipeline
"""

from pipeline import run_pipeline


def main():
    model, diff_model, train_result, test_result = run_pipeline()

    # 示例输出
    display_cols = [
        "order_id",
        "D1_cong_exposure", "D2_cong_severity", "D3_cong_persist",
        "D4_status_jump", "D5_coupling",
        "U1_path_entropy", "U2_high_unc_ratio",
        "difficulty", "grade",
    ]
    print("\n📋 Test set sample (first 15 orders):\n")
    print(test_result[display_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()