from __future__ import annotations

import argparse

from src.workflow import print_summary, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Glue Block Color Correction Demo"
    )

    parser.add_argument(
        "--photo",
        # default="real_photo_iqoo.jpg",
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

    return parser


def main():
    args = build_parser().parse_args()
    report = run(args)
    print_summary(report)


if __name__ == "__main__":
    main()