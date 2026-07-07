from __future__ import annotations

import argparse

from src.workflow import print_summary, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Glue Block Color Correction Demo"
    )

    parser.add_argument(
        "--photo",
        default="qiangguang.jpg",
        help="实拍图（包含色卡和胶块）",
    )

    parser.add_argument(
        "--standard",
        default="standard_chart.png",
        help="标准色卡",
    )

    parser.add_argument(
        "--out",
        default="output_real_correction",
        help="输出目录",
    )

    parser.add_argument(
        "--model",
        default="linear_bias",
        choices=["linear_bias", "ccm", "poly2", "poly3", "root_poly2", "root_poly3"],
        help="颜色校正模型。linear_bias/ccm 是 CCM；root_poly2 推荐优先尝试。",
    )

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-6,
        help="Ridge 正则强度；0 表示普通最小二乘。小样本建议 1e-6 到 1e-3 之间对比。",
    )

    parser.add_argument(
        "--chart-sample-method",
        default="mean",
        choices=["mean", "median", "trimmed_mean"],
        help="色卡色块取色方式。mean为原始均值；median/trimmed_mean更抗异常点。",
    )

    parser.add_argument(
        "--chart-trim-percent",
        type=float,
        default=10.0,
        help="chart-sample-method=trimmed_mean 时的截尾比例。",
    )

    parser.add_argument(
        "--target-trim-percent",
        type=float,
        default=10.0,
        help="胶块目标区域 trimmed mean 截尾比例。",
    )

    parser.add_argument(
        "--target-sequence",
        default="all",
        help="待验证颜色顺序，如 all 或 W032,W048；manual 表示不使用内置批量顺序。",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="目标胶块通过阈值，默认 ΔE00 <= 5。",
    )

    return parser


def main():
    args = build_parser().parse_args()
    report = run(args)
    print_summary(report)


if __name__ == "__main__":
    main()
