from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


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


def write_image_unicode(path: Path, image: np.ndarray, jpg_quality: int = 100) -> None:
    """兼容 Windows 中文路径保存图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    elif suffix == ".png":
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        raise ValueError(f"不支持的输出格式：{suffix}")

    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def strip_quotes(text: str) -> str:
    """移除从资源管理器复制路径时可能带上的引号。"""
    return text.strip().strip('"').strip("'")


def prompt_int(prompt: str, minimum: int = 0) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"请输入不小于 {minimum} 的整数。")


def auto_display_scale(
    width: int,
    height: int,
    max_width: int = 1500,
    max_height: int = 850,
) -> float:
    """图片过大时仅缩小显示，最终 ROI 坐标仍对应原图。"""
    return min(1.0, max_width / width, max_height / height)


def draw_saved_rois(
    image: np.ndarray,
    records: list[dict[str, int | str]],
    scale: float,
) -> None:
    """在框选界面中显示之前已经框选的区域。"""
    for item in records:
        x1 = int(round(int(item["x"]) * scale))
        y1 = int(round(int(item["y"]) * scale))
        x2 = int(round(int(item["x2"]) * scale))
        y2 = int(round(int(item["y2"]) * scale))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            str(item["number"]),
            (x1 + 4, max(20, y1 + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def collect_rois(
    image: np.ndarray,
    count: int,
    start_number: int,
) -> list[dict[str, int | str]]:
    original_h, original_w = image.shape[:2]
    scale = auto_display_scale(original_w, original_h)

    if scale < 1.0:
        display_base = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        print(f"原图较大，框选窗口按 {scale:.3f} 倍显示；保存坐标仍为原图坐标。")
    else:
        display_base = image.copy()

    records: list[dict[str, int | str]] = []

    print("\n框选操作：")
    print("  鼠标拖出矩形。")
    print("  按 Enter 或 Space 确认当前 ROI。")
    print("  按 C 取消当前 ROI，之后可重试或结束。\n")

    index = 0
    while index < count:
        number = start_number + index
        display = display_base.copy()
        draw_saved_rois(display, records, scale)

        window_name = f"ROI {index + 1}/{count} -> {number}.jpg"
        x_d, y_d, w_d, h_d = map(
            int,
            cv2.selectROI(
                window_name,
                display,
                showCrosshair=True,
                fromCenter=False,
            ),
        )
        cv2.destroyWindow(window_name)

        if w_d <= 0 or h_d <= 0:
            choice = input(
                f"第 {index + 1} 个 ROI 未有效框选。"
                "输入 r 重试，输入 q 提前结束："
            ).strip().lower()
            if choice == "q":
                break
            continue

        # 把显示图上的坐标映射回原图坐标。
        x1 = int(round(x_d / scale))
        y1 = int(round(y_d / scale))
        x2 = int(round((x_d + w_d) / scale))
        y2 = int(round((y_d + h_d) / scale))

        x1 = max(0, min(x1, original_w - 1))
        y1 = max(0, min(y1, original_h - 1))
        x2 = max(x1 + 1, min(x2, original_w))
        y2 = max(y1 + 1, min(y2, original_h))

        width = x2 - x1
        height = y2 - y1

        records.append(
            {
                "number": number,
                "filename": f"{number}.jpg",
                "x": x1,
                "y": y1,
                "width": width,
                "height": height,
                "x2": x2,
                "y2": y2,
            }
        )

        print(
            f"[{index + 1}/{count}] {number}.jpg："
            f"x={x1}, y={y1}, width={width}, height={height}"
        )
        index += 1

    cv2.destroyAllWindows()
    return records


def save_results(
    image: np.ndarray,
    records: list[dict[str, int | str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存所有 ROI 小图。
    for item in records:
        x1 = int(item["x"])
        y1 = int(item["y"])
        x2 = int(item["x2"])
        y2 = int(item["y2"])

        crop = image[y1:y2, x1:x2]
        output_path = output_dir / str(item["filename"])
        write_image_unicode(output_path, crop, jpg_quality=100)

    # 保存坐标及尺寸记录。UTF-8 BOM 便于 Excel 打开。
    csv_path = output_dir / "roi_info.csv"
    fieldnames = ["number", "filename", "x", "y", "width", "height", "x2", "y2"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # 保存带编号矩形框的原图预览。
    preview = image.copy()
    for item in records:
        x1 = int(item["x"])
        y1 = int(item["y"])
        x2 = int(item["x2"])
        y2 = int(item["y2"])
        number = int(item["number"])

        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            preview,
            str(number),
            (x1 + 6, max(30, y1 + 32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    preview_path = output_dir / "selection_preview.jpg"
    write_image_unicode(preview_path, preview, jpg_quality=95)

    print("\n处理完成：")
    print(f"  ROI 小图：{output_dir.resolve()}")
    print(f"  坐标记录：{csv_path.resolve()}")
    print(f"  框选预览：{preview_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="手动依次框选多个矩形 ROI，并按递增编号保存为 JPG。"
    )
    parser.add_argument("--image", type=str, help="原始图片路径")
    parser.add_argument("--count", type=int, help="需要框选的 ROI 数量")
    parser.add_argument("--start", type=int, help="起始编号，例如 1")
    parser.add_argument("--out", type=str, help="输出文件夹路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_text = args.image
    if not image_text:
        image_text = input("请输入原始图片路径：")
    image_path = Path(strip_quotes(image_text)).expanduser()

    if not image_path.is_file():
        print(f"错误：找不到图片文件：{image_path}")
        sys.exit(1)

    count = args.count
    if count is None:
        count = prompt_int("请输入需要框选的 ROI 个数：", minimum=1)
    if count < 1:
        print("错误：ROI 个数必须大于 0。")
        sys.exit(1)

    start_number = args.start
    if start_number is None:
        start_number = prompt_int("请输入起始编号，例如 1：", minimum=0)
    if start_number < 0:
        print("错误：起始编号不能小于 0。")
        sys.exit(1)

    output_text = args.out
    if not output_text:
        default_dir = image_path.parent / f"{image_path.stem}_rois"
        raw = input(f"请输入输出文件夹，直接回车使用 [{default_dir}]：").strip()
        output_dir = Path(strip_quotes(raw)).expanduser() if raw else default_dir
    else:
        output_dir = Path(strip_quotes(output_text)).expanduser()

    try:
        image = read_image_unicode(image_path)
        records = collect_rois(image, count, start_number)

        if not records:
            print("没有有效 ROI，未生成输出文件。")
            return

        save_results(image, records, output_dir)

        if len(records) < count:
            print(f"注意：计划框选 {count} 个，实际保存 {len(records)} 个。")

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n用户中断，程序已退出。")
    except Exception as exc:
        cv2.destroyAllWindows()
        print(f"\n运行失败：{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
