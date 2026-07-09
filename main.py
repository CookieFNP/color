from __future__ import annotations

import argparse

from src.workflow import print_summary, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Glue Block Color Correction Demo"
    )

    parser.add_argument(
        "--photo",
        default="IMG_0800.jpg",
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
        default="root_poly2",
        choices=["linear_bias", "poly2", "root_poly2", "poly3", "root_poly3"],
        help="颜色校正模型，默认 root_poly2",
    )

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-6,
        help="岭回归正则强度，默认 1e-6",
    )

    parser.add_argument(
        "--chart-weight-mode",
        default="none",
        choices=["none", "gray", "light", "gray_light"],
        help=(
            "色卡拟合权重模式。"
            "none=普通root_poly2；gray=提高灰阶权重；"
            "light=提高浅色权重；gray_light=灰阶+浅色都提高。"
        ),
    )

    parser.add_argument(
        "--gray-weight",
        type=float,
        default=4.0,
        help="灰阶色块权重，默认 4.0。仅 chart-weight-mode 包含 gray 时生效。",
    )

    parser.add_argument(
        "--light-weight",
        type=float,
        default=2.5,
        help="浅色色块权重，默认 2.5。仅 chart-weight-mode 包含 light 时生效。",
    )

    parser.add_argument(
        "--light-l-threshold",
        type=float,
        default=70.0,
        help="Lab L 大于该值的色卡块视为浅色块，默认 70。",
    )

    return parser


def main():
    args = build_parser().parse_args()
    report = run(args)
    print_summary(report)


if __name__ == "__main__":
    main()
