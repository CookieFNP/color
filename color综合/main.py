from __future__ import annotations

import argparse

from src.workflow import print_summary, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="128 Glue Block Color Correction and Matching")

    parser.add_argument("--photo", required=True, help="实拍图，包含 ColorChecker 和 128 色胶块")
    parser.add_argument("--standard", default="standard_chart.png", help="标准 ColorChecker 图片")
    parser.add_argument("--data", default="data.csv", help="128 色标准 Lab CSV：编号,名称,LAB")
    parser.add_argument("--out", default="output_128", help="输出目录")

    parser.add_argument("--model", default="root_poly2", choices=["linear_bias", "poly2", "root_poly2"], help="颜色校正模型")
    parser.add_argument("--ridge-alpha", type=float, default=1e-6, help="岭回归强度")
    parser.add_argument("--correction-strength", type=float, default=1.0, help="校正强度 0~1，强光可试 0.5")

    parser.add_argument("--target-sequence", default="all", help="默认 all，即按 data.csv 全部顺序手动框选；也可 W001,W002")
    parser.add_argument("--top-k", type=int, default=10, help="输出最近标准色数量")

    parser.add_argument("--force-select-chart", action="store_true", help="强制重新点击 ColorChecker 四角")
    parser.add_argument("--force-select-rois", action="store_true", help="强制重新框选胶块 ROI")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    print_summary(report)


if __name__ == "__main__":
    main()
