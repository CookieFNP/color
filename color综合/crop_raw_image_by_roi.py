from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def strip_quotes(text: str) -> str:
    """移除拖入终端或复制路径时可能带上的引号。"""
    return text.strip().strip('"').strip("'")


def read_image_unicode(path: Path) -> np.ndarray:
    """兼容 Windows 中文路径读取图片。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise RuntimeError(f"读取图片失败：{path}") from exc

    if image is None:
        raise RuntimeError(f"无法识别图片：{path}")

    return image


def write_image_unicode(
    path: Path,
    image: np.ndarray,
    jpg_quality: int = 100,
) -> None:
    """兼容 Windows 中文路径保存 JPG。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, jpg_quality],
    )
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")

    encoded.tofile(str(path))


def load_roi_records(csv_path: Path) -> list[dict[str, str]]:
    """读取 manual_roi_cropper.py 生成的 roi_info.csv。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 ROI 数据文件：{csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("ROI CSV 没有表头。")

        required = {"number", "x", "y"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(
                "ROI CSV 至少需要包含 number、x、y 列。"
            )

        has_x2_y2 = {"x2", "y2"}.issubset(reader.fieldnames)
        has_wh = {"width", "height"}.issubset(reader.fieldnames)

        if not has_x2_y2 and not has_wh:
            raise ValueError(
                "ROI CSV 需要包含 x2、y2，或者 width、height。"
            )

        records = list(reader)

    if not records:
        raise ValueError("ROI CSV 中没有数据。")

    return records


def crop_by_roi_csv(
    image: np.ndarray,
    records: list[dict[str, str]],
    output_dir: Path,
    suffix: str,
) -> list[dict[str, int | str]]:
    image_h, image_w = image.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_records: list[dict[str, int | str]] = []

    for index, row in enumerate(records, start=1):
        try:
            number = int(row["number"])
            x1 = int(float(row["x"]))
            y1 = int(float(row["y"]))

            if row.get("x2", "").strip() and row.get("y2", "").strip():
                x2 = int(float(row["x2"]))
                y2 = int(float(row["y2"]))
            else:
                width = int(float(row["width"]))
                height = int(float(row["height"]))
                x2 = x1 + width
                y2 = y1 + height

        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"ROI CSV 第 {index + 1} 行数据格式错误：{row}"
            ) from exc

        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"编号 {number} 的 ROI 坐标无效："
                f"({x1}, {y1}) - ({x2}, {y2})"
            )

        if x2 > image_w or y2 > image_h:
            raise ValueError(
                f"编号 {number} 的 ROI 超出未处理原图范围。\n"
                f"ROI：({x1}, {y1}) - ({x2}, {y2})\n"
                f"未处理原图尺寸：{image_w} × {image_h}\n"
                "请确认处理前后两张图片的尺寸、裁切和方向完全一致。"
            )

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            raise RuntimeError(f"编号 {number} 裁剪结果为空。")

        filename = f"{number}_{suffix}.jpg"
        output_path = output_dir / filename
        write_image_unicode(output_path, crop, jpg_quality=100)

        saved_records.append(
            {
                "number": number,
                "filename": filename,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "x2": x2,
                "y2": y2,
            }
        )

        print(
            f"[{index}/{len(records)}] 已保存 {filename}："
            f"x={x1}, y={y1}, width={x2 - x1}, height={y2 - y1}"
        )

    return saved_records


def save_output_info(
    records: list[dict[str, int | str]],
    output_dir: Path,
    suffix: str,
) -> None:
    csv_path = output_dir / f"roi_info_{suffix}.csv"
    fieldnames = [
        "number",
        "filename",
        "x",
        "y",
        "width",
        "height",
        "x2",
        "y2",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"裁剪记录已保存：{csv_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 roi_info.csv，在处理前原图上使用相同坐标裁剪，"
            "输出编号_raw.jpg。"
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        help="处理前原图路径",
    )
    parser.add_argument(
        "--roi-csv",
        type=str,
        help="manual_roi_cropper.py 生成的 roi_info.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="输出文件夹",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="raw",
        help="输出文件名后缀，默认 raw",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_text = args.image or input("请输入处理前原图路径：")
    image_path = Path(strip_quotes(image_text)).expanduser()

    csv_text = args.roi_csv or input("请输入 roi_info.csv 路径：")
    csv_path = Path(strip_quotes(csv_text)).expanduser()

    if args.out:
        output_dir = Path(strip_quotes(args.out)).expanduser()
    else:
        default_dir = image_path.parent / f"{image_path.stem}_raw_rois"
        raw_out = input(
            f"请输入输出文件夹，直接回车使用 [{default_dir}]："
        ).strip()
        output_dir = (
            Path(strip_quotes(raw_out)).expanduser()
            if raw_out
            else default_dir
        )

    suffix = args.suffix.strip().strip("_")
    if not suffix:
        suffix = "raw"

    try:
        if not image_path.is_file():
            raise FileNotFoundError(f"找不到处理前原图：{image_path}")

        image = read_image_unicode(image_path)
        records = load_roi_records(csv_path)

        saved_records = crop_by_roi_csv(
            image=image,
            records=records,
            output_dir=output_dir,
            suffix=suffix,
        )

        save_output_info(
            records=saved_records,
            output_dir=output_dir,
            suffix=suffix,
        )

        print("\n处理完成：")
        print(f"  未处理原图：{image_path.resolve()}")
        print(f"  ROI 数据：{csv_path.resolve()}")
        print(f"  输出文件夹：{output_dir.resolve()}")
        print(f"  共保存：{len(saved_records)} 张")
        print(f"  命名格式：编号_{suffix}.jpg")

    except KeyboardInterrupt:
        print("\n用户中断，程序已退出。")
    except Exception as exc:
        print(f"\n运行失败：{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
