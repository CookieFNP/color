from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray, jpg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    else:
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 2]

    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise OSError(f"图片编码失败：{path}")
    encoded.tofile(str(path))


def load_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"名称文件不存在：{path}")

    names: list[str] = []

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                parts = [item.strip() for item in row if item.strip()]
                if parts:
                    names.append("".join(parts))
    else:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                value = line.strip()
                if value:
                    names.append(value)

    if not names:
        raise ValueError("名称列表为空。")

    return names


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.strip().rstrip(". ")
    return name or "未命名"


def resize_for_display(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)

    if scale >= 1.0:
        return image.copy(), 1.0

    display_width = max(1, int(round(width * scale)))
    display_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def draw_index_hint(image: np.ndarray, index: int, total: int) -> np.ndarray:
    result = image.copy()
    overlay = result.copy()
    cv2.rectangle(overlay, (0, 0), (result.shape[1], 50), (0, 0, 0), -1)
    result = cv2.addWeighted(overlay, 0.58, result, 0.42, 0)

    text = f"ROI {index}/{total} | Drag | ENTER/SPACE confirm | C cancel | ESC stop"
    cv2.putText(
        result,
        text,
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def select_rois_in_order(
    image: np.ndarray,
    names: list[str],
    max_width: int,
    max_height: int,
) -> list[tuple[int, int, int, int]]:
    display_base, scale = resize_for_display(image, max_width, max_height)
    rois: list[tuple[int, int, int, int]] = []
    window_name = "Select ROIs In Name Order"

    print("\n========== 按顺序框选 ROI ==========")
    print("鼠标拖动：框选当前胶块")
    print("Enter / Space：确认当前框")
    print("C：取消当前框并重画")
    print("Esc：提前结束")
    print("当前中文名称会显示在命令行。")

    for index, name in enumerate(names, start=1):
        print(f"\n[{index}/{len(names)}] 当前待框选：{name}")

        display = draw_index_hint(display_base, index, len(names))
        roi = cv2.selectROI(
            window_name,
            display,
            showCrosshair=True,
            fromCenter=False,
        )

        x, y, w, h = map(int, roi)

        if w <= 0 or h <= 0:
            print(f"未确认第 {index} 个 ROI，结束框选。")
            break

        x1 = int(round(x / scale))
        y1 = int(round(y / scale))
        x2 = int(round((x + w) / scale))
        y2 = int(round((y + h) / scale))

        x1 = int(np.clip(x1, 0, image.shape[1]))
        y1 = int(np.clip(y1, 0, image.shape[0]))
        x2 = int(np.clip(x2, 0, image.shape[1]))
        y2 = int(np.clip(y2, 0, image.shape[0]))

        if x2 <= x1 or y2 <= y1:
            print("ROI 映射后无效，停止。")
            break

        rois.append((x1, y1, x2, y2))
        print(
            f"已记录：{name} -> "
            f"({x1}, {y1}) - ({x2}, {y2})，"
            f"尺寸 {x2 - x1}×{y2 - y1}"
        )

    cv2.destroyAllWindows()
    return rois


def save_results(
    image: np.ndarray,
    names: list[str],
    rois: list[tuple[int, int, int, int]],
    out_dir: Path,
    pad: int,
    add_index: bool,
    output_ext: str,
) -> None:
    if not rois:
        raise RuntimeError("没有确认任何 ROI。")

    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / "mapping.csv"

    rows: list[list[object]] = []
    used_names: dict[str, int] = {}

    for order, ((x1, y1, x2, y2), name) in enumerate(zip(rois, names), start=1):
        crop_x1 = min(max(x1 + pad, 0), image.shape[1])
        crop_y1 = min(max(y1 + pad, 0), image.shape[0])
        crop_x2 = min(max(x2 - pad, 0), image.shape[1])
        crop_y2 = min(max(y2 - pad, 0), image.shape[0])

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise ValueError(f"第 {order} 个 ROI 应用 pad 后为空，请减小 --pad。")

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        safe_name = sanitize_filename(name)
        used_names[safe_name] = used_names.get(safe_name, 0) + 1
        duplicate_no = used_names[safe_name]

        stem = f"{order:03d}_{safe_name}" if add_index else safe_name
        if duplicate_no > 1:
            stem = f"{stem}_{duplicate_no}"

        filename = f"{stem}{output_ext}"
        imwrite_unicode(out_dir / filename, crop)

        rows.append(
            [
                order,
                name,
                filename,
                crop_x1,
                crop_y1,
                crop_x2,
                crop_y2,
                crop_x2 - crop_x1,
                crop_y2 - crop_y1,
            ]
        )

    with mapping_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["顺序", "名称", "文件名", "x1", "y1", "x2", "y2", "宽度", "高度"]
        )
        writer.writerows(rows)

    print(f"\n保存完成：{len(rois)} 张")
    print(f"输出目录：{out_dir.resolve()}")
    print(f"映射表：{mapping_path.resolve()}")

    if len(rois) < len(names):
        print(
            f"注意：名称共有 {len(names)} 个，"
            f"本次只保存了前 {len(rois)} 个。"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按名称顺序逐个手动框选 ROI，并批量裁剪命名。"
    )
    parser.add_argument("--image", required=True, type=Path, help="待处理的大图")
    parser.add_argument(
        "--names",
        required=True,
        type=Path,
        help="名称列表：txt 每行一个；csv 每行字段自动拼接",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("roi_output"),
        help="输出目录，默认 roi_output",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=0,
        help="每个 ROI 四周向内缩的像素，默认 0",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="文件名前不添加 001_、002_ 顺序编号",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="操作窗口最大宽度，默认 1600",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="操作窗口最大高度，默认 900",
    )
    parser.add_argument(
        "--ext",
        choices=[".jpg", ".png"],
        default=".jpg",
        help="输出格式，默认 .jpg",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        image = imread_unicode(args.image)
        names = load_names(args.names)

        rois = select_rois_in_order(
            image=image,
            names=names,
            max_width=args.max_width,
            max_height=args.max_height,
        )

        save_results(
            image=image,
            names=names,
            rois=rois,
            out_dir=args.out,
            pad=max(0, args.pad),
            add_index=not args.no_index,
            output_ext=args.ext,
        )
        return 0

    except Exception as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
