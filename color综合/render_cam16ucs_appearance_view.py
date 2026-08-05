# -*- coding: utf-8 -*-
"""
用途：
    生成“CAM16-UCS 色貌学肉眼视图”。

它和之前的 Lab display 脚本区别是：
    之前调：
        L / a / b / chroma

    这个脚本调：
        CAM16-UCS 的 J' / a' / b'
    更接近“人眼看起来的明暗、浓淡、偏色”。

典型用途：
    你已经通过 board_photo_match_v6_roi_crop.py 得到了：
        board_match_combined/02_board_corrected.png

    然后希望在这张校正图上，重新点四个点圈出整块板材/页面区域，
    只对该区域做“肉眼视图”渲染，并输出完整大图。

安装依赖：
    pip install colour-science

典型命令：
    python render_cam16ucs_appearance_view.py ^
      --image board_match_combined/02_board_corrected.png ^
      --out board_match_combined/09_cam16ucs_appearance_view.png ^
      --j-offset -5 ^
      --chroma-scale 0.95 ^
      --ap-offset -1 ^
      --bp-offset -1 ^
      --force-select-region

参数含义：
    --j-offset
        CAM16-UCS J' 偏移。
        负数更暗，正数更亮。

    --chroma-scale
        CAM16-UCS 色度缩放。
        >1 更浓/更饱和，<1 更灰/更淡。

    --ap-offset
        CAM16-UCS a' 偏移。
        正数偏红，负数偏绿。

    --bp-offset
        CAM16-UCS b' 偏移。
        正数偏黄，负数偏蓝。

    --hue-offset
        色相角旋转，单位度。
        一般先不用，默认 0。

输出：
    09_cam16ucs_appearance_view.png
        完整肉眼视图图

    09_cam16ucs_appearance_view_region_outline.png
        四点区域标注图

    09_cam16ucs_appearance_view_region_crop_before.png
        区域调整前截图

    09_cam16ucs_appearance_view_region_crop_after.png
        区域调整后截图

    09_cam16ucs_appearance_view_region_compare.png
        区域前后对比图
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# colour-science
# ============================================================

def import_colour():
    try:
        import colour
        return colour
    except Exception as exc:
        raise RuntimeError(
            "缺少依赖 colour-science。请先运行：pip install colour-science"
        ) from exc


# ============================================================
# Unicode 图像 IO
# ============================================================

def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    return img


def imwrite_unicode(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"无法编码图像：{path}")
    buf.tofile(str(path))


# ============================================================
# sRGB <-> XYZ，范围 0~1
# ============================================================

D65_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)


def srgb_to_linear(rgb_01: np.ndarray) -> np.ndarray:
    rgb_01 = np.clip(np.asarray(rgb_01, dtype=np.float64), 0.0, 1.0)
    return np.where(
        rgb_01 <= 0.04045,
        rgb_01 / 12.92,
        ((rgb_01 + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    return np.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * (lin ** (1 / 2.4)) - 0.055,
    )


def rgb01_to_xyz(rgb_01: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(rgb_01)
    return lin @ SRGB_TO_XYZ.T


def xyz_to_rgb01(xyz: np.ndarray) -> np.ndarray:
    lin = xyz @ XYZ_TO_SRGB.T
    return linear_to_srgb(lin)


# ============================================================
# 交互选四点
# ============================================================

def select_four_points(image_bgr: np.ndarray, title: str, max_w: int = 1300, max_h: int = 850) -> list[tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    shown = cv2.resize(
        image_bgr,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )

    points: list[tuple[int, int]] = []

    def redraw():
        canvas = shown.copy()

        if len(points) >= 2:
            pts = np.array(
                [[int(x * scale), int(y * scale)] for x, y in points],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], isClosed=(len(points) == 4), color=(0, 255, 255), thickness=2)

        for i, (x, y) in enumerate(points):
            xs = int(round(x * scale))
            ys = int(round(y * scale))
            cv2.circle(canvas, (xs, ys), 6, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                str(i + 1),
                (xs + 8, ys - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            canvas,
            "Click 4 points: TL, TR, BR, BL | Enter confirm | R reset | Esc cancel",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )
        cv2.imshow(title, canvas)

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(round(x / scale)), int(round(y / scale))))
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, shown.shape[1], shown.shape[0])
    cv2.moveWindow(title, 30, 30)
    cv2.setMouseCallback(title, on_mouse)

    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in [13, 10] and len(points) == 4:
            break

        if key in [ord("r"), ord("R")]:
            points = []
            redraw()

        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("用户取消了四点区域选择。")

    cv2.destroyWindow(title)
    return points


# ============================================================
# mask / crop / compare
# ============================================================

def polygon_mask(shape_hw: tuple[int, int], points: list[tuple[int, int]], feather: int = 9) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)
    pts = np.asarray(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1.0)

    if feather > 0:
        k = int(feather) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = np.clip(mask, 0, 1)

    return mask


def draw_region_outline(image_bgr: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    canvas = image_bgr.copy()
    pts = np.asarray(points, dtype=np.int32)
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 255), thickness=4)

    for i, (x, y) in enumerate(points):
        cv2.circle(canvas, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            str(i + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

    return canvas


def crop_bbox(image_bgr: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(w, max(xs))
    y2 = min(h, max(ys))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("四点区域 bbox 无效。")
    return image_bgr[y1:y2, x1:x2].copy()


def side_by_side(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    target_h = min(img.shape[0] for img in images)
    resized = []

    for img, label in zip(images, labels):
        scale = target_h / img.shape[0]
        new_w = max(1, int(img.shape[1] * scale))
        r = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

        cv2.rectangle(r, (0, 0), (r.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(
            r,
            label,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        resized.append(r)

    return np.concatenate(resized, axis=1)


# ============================================================
# CAM16-UCS 渲染核心
# ============================================================

def rotate_ab(ap: np.ndarray, bp: np.ndarray, hue_offset_deg: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(hue_offset_deg) < 1e-9:
        return ap, bp

    theta = np.deg2rad(float(hue_offset_deg))
    c = np.cos(theta)
    s = np.sin(theta)
    ap2 = ap * c - bp * s
    bp2 = ap * s + bp * c
    return ap2, bp2


def apply_cam16ucs_to_rgb_chunk(
    rgb_01: np.ndarray,
    colour,
    *,
    j_offset: float,
    chroma_scale: float,
    chroma_offset: float,
    ap_offset: float,
    bp_offset: float,
    hue_offset: float,
    L_A: float,
    Y_b: float,
) -> np.ndarray:
    """
    rgb_01: (N, 3), sRGB 0~1
    return: (N, 3), sRGB 0~1
    """
    xyz = rgb01_to_xyz(rgb_01)

    # colour.XYZ_to_CAM16UCS / CAM16UCS_to_XYZ 的 XYZ 使用 0~1 范围
    Jpapbp = colour.XYZ_to_CAM16UCS(
        xyz,
        XYZ_w=D65_XYZ,
        L_A=float(L_A),
        Y_b=float(Y_b),
    )

    Jp = Jpapbp[:, 0]
    ap = Jpapbp[:, 1]
    bp = Jpapbp[:, 2]

    # 明度
    Jp2 = np.clip(Jp + float(j_offset), 0, 100)

    # 色度缩放
    C = np.sqrt(ap * ap + bp * bp)
    C2 = C * float(chroma_scale) + float(chroma_offset)

    ratio = np.ones_like(C)
    valid = C > 1e-8
    ratio[valid] = C2[valid] / C[valid]

    ap2 = ap * ratio
    bp2 = bp * ratio

    # 色相旋转
    ap2, bp2 = rotate_ab(ap2, bp2, float(hue_offset))

    # 红绿 / 黄蓝偏移
    ap2 = ap2 + float(ap_offset)
    bp2 = bp2 + float(bp_offset)

    Jpapbp2 = np.stack([Jp2, ap2, bp2], axis=-1)

    xyz2 = colour.CAM16UCS_to_XYZ(
        Jpapbp2,
        XYZ_w=D65_XYZ,
        L_A=float(L_A),
        Y_b=float(Y_b),
    )

    rgb2 = xyz_to_rgb01(xyz2)
    return np.clip(rgb2, 0, 1)


def apply_cam16ucs_view_to_region(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    j_offset: float,
    chroma_scale: float,
    chroma_offset: float,
    ap_offset: float,
    bp_offset: float,
    hue_offset: float,
    L_A: float,
    Y_b: float,
    chunk_size: int,
) -> np.ndarray:
    colour = import_colour()

    rgb = image_bgr[:, :, ::-1].astype(np.float64) / 255.0
    h, w = rgb.shape[:2]

    flat = rgb.reshape(-1, 3)
    flat_out = np.empty_like(flat)

    total = len(flat)
    for start in range(0, total, int(chunk_size)):
        end = min(total, start + int(chunk_size))
        flat_out[start:end] = apply_cam16ucs_to_rgb_chunk(
            flat[start:end],
            colour,
            j_offset=j_offset,
            chroma_scale=chroma_scale,
            chroma_offset=chroma_offset,
            ap_offset=ap_offset,
            bp_offset=bp_offset,
            hue_offset=hue_offset,
            L_A=L_A,
            Y_b=Y_b,
        )

    rgb2 = flat_out.reshape(h, w, 3)

    alpha = mask[..., None].astype(np.float64)
    mixed = rgb * (1 - alpha) + rgb2 * alpha

    out_bgr = np.clip(mixed[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    return out_bgr


# ============================================================
# main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="生成 CAM16-UCS 肉眼视图整图。")

    parser.add_argument("--image", required=True, help="输入整图，推荐 02_board_corrected.png")
    parser.add_argument("--out", required=True, help="输出肉眼视图整图路径")

    parser.add_argument("--points-file", default=None, help="四点区域 JSON。默认放在 out 同目录。")
    parser.add_argument("--force-select-region", action="store_true", help="强制重新选择四点区域。")
    parser.add_argument("--feather", type=int, default=9, help="区域边缘羽化，默认 9。")

    parser.add_argument("--j-offset", type=float, default=0.0, help="J' 偏移。负数更暗，正数更亮。")
    parser.add_argument("--chroma-scale", type=float, default=1.0, help="色度倍率。>1 更浓，<1 更淡。")
    parser.add_argument("--chroma-offset", type=float, default=0.0, help="固定增加色度，默认 0。")
    parser.add_argument("--ap-offset", type=float, default=0.0, help="a' 偏移。正数偏红，负数偏绿。")
    parser.add_argument("--bp-offset", type=float, default=0.0, help="b' 偏移。正数偏黄，负数偏蓝。")
    parser.add_argument("--hue-offset", type=float, default=0.0, help="色相旋转角度，默认 0。")

    parser.add_argument("--L-A", dest="L_A", type=float, default=64.0, help="CAM16 适应亮度，默认 64。")
    parser.add_argument("--Y-b", dest="Y_b", type=float, default=20.0, help="CAM16 背景亮度，默认 20。")

    parser.add_argument("--chunk-size", type=int, default=200000, help="分块处理像素数，图大时可调小。")

    args = parser.parse_args()

    image_path = Path(args.image)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    points_file = Path(args.points_file) if args.points_file else (out_path.parent / "cam16ucs_view_region_points.json")

    img = imread_unicode(image_path)

    if points_file.exists() and not args.force_select_region:
        points = json.loads(points_file.read_text(encoding="utf-8"))
        points = [tuple(map(int, p)) for p in points]
        print("已加载四点区域：", points_file)
    else:
        print("\n请点击要做 CAM16-UCS 肉眼视图调整的四个点：建议左上、右上、右下、左下。")
        points = select_four_points(img, "Select CAM16-UCS appearance region")
        points_file.write_text(json.dumps(points, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已保存四点区域：", points_file)

    mask = polygon_mask(img.shape[:2], points, feather=args.feather)

    display_img = apply_cam16ucs_view_to_region(
        img,
        mask,
        j_offset=args.j_offset,
        chroma_scale=args.chroma_scale,
        chroma_offset=args.chroma_offset,
        ap_offset=args.ap_offset,
        bp_offset=args.bp_offset,
        hue_offset=args.hue_offset,
        L_A=args.L_A,
        Y_b=args.Y_b,
        chunk_size=args.chunk_size,
    )

    imwrite_unicode(out_path, display_img)

    outline = draw_region_outline(img, points)
    outline_path = out_path.with_name(out_path.stem + "_region_outline.png")
    imwrite_unicode(outline_path, outline)

    crop_before = crop_bbox(img, points)
    crop_after = crop_bbox(display_img, points)

    crop_before_path = out_path.with_name(out_path.stem + "_region_crop_before.png")
    crop_after_path = out_path.with_name(out_path.stem + "_region_crop_after.png")
    compare_path = out_path.with_name(out_path.stem + "_region_compare.png")

    imwrite_unicode(crop_before_path, crop_before)
    imwrite_unicode(crop_after_path, crop_after)
    imwrite_unicode(compare_path, side_by_side([crop_before, crop_after], ["before", "CAM16-UCS appearance view"]))

    summary = {
        "image": str(image_path),
        "out": str(out_path),
        "points_file": str(points_file),
        "points": points,
        "params": {
            "j_offset": args.j_offset,
            "chroma_scale": args.chroma_scale,
            "chroma_offset": args.chroma_offset,
            "ap_offset": args.ap_offset,
            "bp_offset": args.bp_offset,
            "hue_offset": args.hue_offset,
            "L_A": args.L_A,
            "Y_b": args.Y_b,
            "feather": args.feather,
            "chunk_size": args.chunk_size,
        },
        "outputs": {
            "appearance_view": str(out_path),
            "region_outline": str(outline_path),
            "region_crop_before": str(crop_before_path),
            "region_crop_after": str(crop_after_path),
            "region_compare": str(compare_path),
        },
    }

    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== CAM16-UCS 肉眼视图生成完成 ====")
    print("完整肉眼视图：", out_path)
    print("区域标注图：", outline_path)
    print("区域前后对比：", compare_path)
    print("四点文件：", points_file)
    print("summary：", summary_path)


if __name__ == "__main__":
    main()
