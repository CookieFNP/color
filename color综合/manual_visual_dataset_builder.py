# -*- coding: utf-8 -*-
"""
用途：
    manual_visual_dataset_builder.py

    为“真实拍照 corrected_lab -> 肉眼 visual_lab”建立训练样本库。

核心目标：
    不是自动识别 TopK，也不是自动判断置信度。
    而是把每张“胶块合照 + 24 色卡”稳定转成一批训练样本：

        corrected_L/a/b  ->  visual_display_L/a/b

适用场景：
    1. 有标准 CSV 的胶块大板：
        - CSV 里有 编号/名称/LAB
        - 输出 corrected_lab、visual_lab、standard_lab
        - 可同时作为正式 glue_visual_library 的来源

    2. 没有标准 CSV 的胶块大板：
        - 只需要传 --count N
        - 自动生成 sample_001 ... sample_N
        - 输出 corrected_lab、visual_lab
        - standard_lab 留空
        - 可作为 visual_mapping_T 的训练样本

典型两阶段流程：

    第一阶段：第一次处理一张图，手动选色卡四角 + 手动画所有胶块圆
    python manual_visual_dataset_builder.py ^
      --photo data/yiheng/pic_yiheng.jpg ^
      --standard standard_chart.png ^
      --data data/yiheng/data_yiheng.csv ^
      --out output_yiheng_dataset ^
      --force-select-chart ^
      --force-select-circles

    这一步输出：
        output_yiheng_dataset/02_corrected.png
        output_yiheng_dataset/chart_corners.json
        output_yiheng_dataset/visual_circles_manual.json
        output_yiheng_dataset/corrected_samples.csv

    然后你用原来的视觉渲染/分组规则脚本，人工调出肉眼一致的最终图，例如：
        output_yiheng_dataset/final_preview.png

    第二阶段：把最终肉眼图采样成 visual_lab，形成训练样本
    python manual_visual_dataset_builder.py ^
      --photo data/yiheng/pic_yiheng.jpg ^
      --standard standard_chart.png ^
      --data data/yiheng/data_yiheng.csv ^
      --out output_yiheng_dataset ^
      --final-preview output_yiheng_dataset/final_preview.png

    这一步输出：
        output_yiheng_dataset/visual_training_samples.csv
        output_yiheng_dataset/glue_visual_library_from_dataset.csv

如果没有标准 CSV：
    python manual_visual_dataset_builder.py ^
      --photo data/no_std/board.jpg ^
      --standard standard_chart.png ^
      --count 80 ^
      --out output_no_std_dataset ^
      --force-select-chart ^
      --force-select-circles

再采样最终图：
    python manual_visual_dataset_builder.py ^
      --photo data/no_std/board.jpg ^
      --standard standard_chart.png ^
      --count 80 ^
      --out output_no_std_dataset ^
      --final-preview output_no_std_dataset/final_preview.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 路径读写：兼容中文路径
# ============================================================

def imread_unicode(path: str | Path):
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return img


def imwrite_unicode(path: str | Path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    buf.tofile(str(path))


# ============================================================
# 色彩基础：sRGB / Linear RGB / XYZ / Lab
# ============================================================

D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)


def srgb_to_linear(rgb_255: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_255, dtype=np.float64) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    linear_rgb = np.clip(np.asarray(linear_rgb, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(
        linear_rgb <= 0.0031308,
        linear_rgb * 12.92,
        1.055 * np.power(linear_rgb, 1 / 2.4) - 0.055,
    )
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def rgb_to_lab(rgb_255: np.ndarray) -> np.ndarray:
    rgb_linear = srgb_to_linear(rgb_255)
    xyz = rgb_linear @ SRGB_TO_XYZ.T
    xyz_scaled = xyz / D65_WHITE

    eps = 216 / 24389
    kap = 24389 / 27
    f = np.where(xyz_scaled > eps, np.cbrt(xyz_scaled), (kap * xyz_scaled + 16) / 116)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def delta_e_76(lab1, lab2):
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    return np.linalg.norm(lab1 - lab2, axis=-1)


# ============================================================
# ColorChecker 提取和颜色校正
# ============================================================

def warp_chart_from_photo(photo_bgr: np.ndarray, corners: np.ndarray, output_size=(600, 400)):
    dst_w, dst_h = output_size
    src = np.asarray(corners, dtype=np.float32)
    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(photo_bgr, matrix, (dst_w, dst_h))
    return warped, matrix


def extract_chart_means(chart_bgr: np.ndarray, rows=4, cols=6, center_ratio=0.50) -> np.ndarray:
    h, w = chart_bgr.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    margin = (1.0 - center_ratio) / 2.0

    means_rgb = []
    for r in range(rows):
        for c in range(cols):
            x1 = int((c + margin) * cell_w)
            x2 = int((c + 1 - margin) * cell_w)
            y1 = int((r + margin) * cell_h)
            y2 = int((r + 1 - margin) * cell_h)

            patch = chart_bgr[y1:y2, x1:x2]
            if patch.size == 0:
                raise RuntimeError(f"ColorChecker patch 提取失败：row={r+1}, col={c+1}")
            mean_bgr = patch.reshape(-1, 3).mean(axis=0)
            means_rgb.append(mean_bgr[::-1])  # RGB

    return np.asarray(means_rgb, dtype=np.float64)


def build_features(linear_rgb: np.ndarray, model_type: str) -> np.ndarray:
    arr = np.asarray(linear_rgb, dtype=np.float64)
    flat = arr.reshape(-1, 3)
    R = flat[:, 0:1]
    G = flat[:, 1:2]
    B = flat[:, 2:3]
    ones = np.ones_like(R)

    if model_type == "linear_bias":
        return np.concatenate([R, G, B, ones], axis=1)

    if model_type == "poly2":
        return np.concatenate([R, G, B, R * R, G * G, B * B, R * G, R * B, G * B, ones], axis=1)

    if model_type == "root_poly2":
        # root polynomial 常用形式：一次项 + sqrt(二次交叉项) + bias
        # 这里加入二次项的 root 版本，避免高饱和色过度发散。
        eps = 1e-12
        return np.concatenate(
            [
                R, G, B,
                np.sqrt(np.maximum(R * G, eps)),
                np.sqrt(np.maximum(R * B, eps)),
                np.sqrt(np.maximum(G * B, eps)),
                ones,
            ],
            axis=1,
        )

    if model_type == "root_poly2_nobias":
        eps = 1e-12
        return np.concatenate(
            [
                R, G, B,
                np.sqrt(np.maximum(R * G, eps)),
                np.sqrt(np.maximum(R * B, eps)),
                np.sqrt(np.maximum(G * B, eps)),
            ],
            axis=1,
        )

    raise ValueError(f"未知 model_type：{model_type}")


def fit_correction_model(captured_rgb: np.ndarray, reference_rgb: np.ndarray, model_type="root_poly2_nobias", ridge_alpha=1e-6):
    captured_linear = srgb_to_linear(captured_rgb)
    reference_linear = srgb_to_linear(reference_rgb)

    X = build_features(captured_linear, model_type)
    Y = reference_linear.reshape(-1, 3)

    if ridge_alpha > 0:
        reg = ridge_alpha * np.eye(X.shape[1], dtype=np.float64)
        W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    else:
        W, *_ = np.linalg.lstsq(X, Y, rcond=None)

    return W


def apply_correction_to_image(photo_bgr: np.ndarray, W: np.ndarray, model_type="root_poly2_nobias") -> np.ndarray:
    rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    rgb_linear = srgb_to_linear(rgb)
    h, w = rgb_linear.shape[:2]

    X = build_features(rgb_linear, model_type)
    fixed_linear = (X @ W).reshape(h, w, 3)
    fixed_rgb = linear_to_srgb(fixed_linear)

    return cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2BGR)


# ============================================================
# 交互：选色卡四角 / 画圆
# ============================================================

def resize_for_display(img_bgr: np.ndarray, max_w=1200, max_h=800):
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return resized, scale


def select_four_points(img_bgr: np.ndarray, title="select chart corners"):
    display, scale = resize_for_display(img_bgr)
    temp = display.copy()
    points = []
    instructions = ["top-left", "top-right", "bottom-right", "bottom-left"]

    def redraw():
        nonlocal temp
        temp = display.copy()
        idx = min(len(points), 3)
        text = f"click ColorChecker {instructions[idx]} | Enter confirm | R reset | Esc cancel"
        cv2.putText(temp, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        for i, p in enumerate(points):
            px = int(p[0] * scale)
            py = int(p[1] * scale)
            cv2.circle(temp, (px, py), 7, (0, 0, 255), -1)
            cv2.putText(temp, str(i + 1), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x / scale, y / scale])
            redraw()

    redraw()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse)

    print("\n请依次点击 24 色卡四角：左上、右上、右下、左下。")
    print("点完按 Enter 确认，R 重选，Esc 退出。")

    while True:
        cv2.imshow(title, temp)
        key = cv2.waitKey(20) & 0xFF

        if key in [13, 10] and len(points) == 4:
            break
        if key in [ord("r"), ord("R")]:
            points.clear()
            redraw()
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("用户取消色卡四角选择。")

    cv2.destroyWindow(title)
    return np.asarray(points, dtype=np.float64)


def select_circles(img_bgr: np.ndarray, specs: list[dict], old_circles: list[dict] | None = None):
    display, scale = resize_for_display(img_bgr)
    circles = []

    old_circles = old_circles or []

    print("\n开始手动画胶块圆。")
    print("鼠标左键按下为圆心，拖动决定半径，松开后按 Enter 确认。")
    print("R 重画当前胶块；Esc 若有旧圆则沿用旧圆，否则跳过。")

    for i, spec in enumerate(specs):
        code = spec.get("code", f"sample_{i+1:03d}")
        name = spec.get("name", "")
        old = old_circles[i] if i < len(old_circles) else None

        current = None
        drawing = False
        start = None
        temp = display.copy()

        def draw_base():
            canvas = display.copy()
            title = f"{i+1}/{len(specs)}  {code} {name} | drag circle | Enter ok | R reset | Esc keep old/skip"
            cv2.putText(canvas, title[:95], (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2)

            # 画已确认的历史圆，便于参考
            for c in circles:
                if c is None:
                    continue
                cx = int(c["cx"] * scale)
                cy = int(c["cy"] * scale)
                rr = int(c["r"] * scale)
                cv2.circle(canvas, (cx, cy), rr, (80, 180, 80), 1)

            if old:
                cx = int(float(old["cx"]) * scale)
                cy = int(float(old["cy"]) * scale)
                rr = int(float(old["r"]) * scale)
                cv2.circle(canvas, (cx, cy), rr, (255, 0, 255), 2)
                cv2.putText(canvas, "old", (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            return canvas

        temp = draw_base()

        def mouse(event, x, y, flags, param):
            nonlocal drawing, start, current, temp

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start = (x, y)
                current = None

            elif event == cv2.EVENT_MOUSEMOVE and drawing and start is not None:
                temp = draw_base()
                r = math.hypot(x - start[0], y - start[1])
                cv2.circle(temp, start, int(r), (0, 0, 255), 2)

            elif event == cv2.EVENT_LBUTTONUP and drawing and start is not None:
                drawing = False
                r = math.hypot(x - start[0], y - start[1])
                current = {
                    "code": code,
                    "name": name,
                    "cx": float(start[0] / scale),
                    "cy": float(start[1] / scale),
                    "r": float(r / scale),
                }
                temp = draw_base()
                cv2.circle(temp, start, int(r), (0, 0, 255), 2)

        win = "select glue circles"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, mouse)

        while True:
            cv2.imshow(win, temp)
            key = cv2.waitKey(20) & 0xFF

            if key in [13, 10]:
                if current is not None:
                    circles.append(current)
                    break
                elif old is not None:
                    # 没重新画，Enter 也沿用旧圆
                    copied = dict(old)
                    copied["code"] = code
                    copied["name"] = name
                    circles.append(copied)
                    break

            if key in [ord("r"), ord("R")]:
                current = None
                temp = draw_base()

            if key == 27:
                if old is not None:
                    copied = dict(old)
                    copied["code"] = code
                    copied["name"] = name
                    circles.append(copied)
                    break
                else:
                    circles.append(None)
                    break

    cv2.destroyWindow("select glue circles")
    return circles


# ============================================================
# 数据读取
# ============================================================

def parse_lab_text(text):
    if pd.isna(text):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+", str(text))
    if len(nums) < 3:
        return None
    return tuple(float(x) for x in nums[:3])


def load_specs(data_csv: Path | None, count: int | None, code_prefix: str):
    specs = []

    if data_csv is not None:
        if not data_csv.exists():
            raise FileNotFoundError(f"找不到 data csv：{data_csv}")

        df = pd.read_csv(data_csv, encoding="utf-8-sig")

        code_col = "code" if "code" in df.columns else ("编号" if "编号" in df.columns else None)
        name_col = "name" if "name" in df.columns else ("名称" if "名称" in df.columns else None)

        if code_col is None:
            raise RuntimeError("data csv 里找不到 code/编号 列。")

        for i, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip() if name_col else ""

            std_lab = None
            if "LAB" in df.columns:
                std_lab = parse_lab_text(row["LAB"])
            elif all(c in df.columns for c in ["standard_L", "standard_a", "standard_b"]):
                std_lab = (float(row["standard_L"]), float(row["standard_a"]), float(row["standard_b"]))
            elif all(c in df.columns for c in ["L", "a", "b"]):
                std_lab = (float(row["L"]), float(row["a"]), float(row["b"]))

            specs.append(
                {
                    "index": i + 1,
                    "code": code,
                    "name": name,
                    "standard_lab": std_lab,
                }
            )

        return specs

    if count is None or count <= 0:
        raise RuntimeError("没有 data csv 时，必须提供 --count N。")

    for i in range(1, count + 1):
        specs.append(
            {
                "index": i,
                "code": f"{code_prefix}_{i:03d}",
                "name": "",
                "standard_lab": None,
            }
        )

    return specs


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# 采样
# ============================================================

def circle_mask_for_image(h, w, cx, cy, r):
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return dist <= r


def sample_circle_lab(img_bgr: np.ndarray, circle: dict):
    if circle is None:
        return None, None

    h, w = img_bgr.shape[:2]
    cx = float(circle["cx"])
    cy = float(circle["cy"])
    r = float(circle["r"])

    x1 = max(0, int(math.floor(cx - r)))
    y1 = max(0, int(math.floor(cy - r)))
    x2 = min(w, int(math.ceil(cx + r)))
    y2 = min(h, int(math.ceil(cy + r)))

    if x2 <= x1 or y2 <= y1:
        return None, None

    patch = img_bgr[y1:y2, x1:x2]
    local_cx = cx - x1
    local_cy = cy - y1
    mask = circle_mask_for_image(y2 - y1, x2 - x1, local_cx, local_cy, r)

    if mask.sum() < 20:
        return None, None

    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    pixels = rgb[mask]
    mean_rgb = pixels.reshape(-1, 3).mean(axis=0)
    lab = rgb_to_lab(mean_rgb.reshape(1, 1, 3))[0, 0]

    return mean_rgb, lab


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="手动建立 corrected_lab -> visual_lab 训练样本库。")

    parser.add_argument("--photo", required=True, type=Path, help="胶块合照，包含 24 色卡。")
    parser.add_argument("--standard", required=True, type=Path, help="标准 24 色卡图片 standard_chart.png。")
    parser.add_argument("--data", default=None, type=Path, help="可选，标准 CSV：编号/名称/LAB。")
    parser.add_argument("--count", default=None, type=int, help="没有 --data 时，手动胶块数量。")
    parser.add_argument("--code-prefix", default="sample", help="没有 --data 时自动生成 code 的前缀。")
    parser.add_argument("--out", required=True, type=Path, help="输出项目目录。")

    parser.add_argument("--model-type", default="root_poly2_nobias", choices=["linear_bias", "poly2", "root_poly2", "root_poly2_nobias"])
    parser.add_argument("--ridge-alpha", default=1e-6, type=float)

    parser.add_argument("--force-select-chart", action="store_true", help="强制重新选择色卡四角。")
    parser.add_argument("--force-select-circles", action="store_true", help="强制重新手动画胶块圆。")

    parser.add_argument("--final-preview", default=None, type=Path, help="人工调好的最终肉眼图；提供后会采样 visual_display_lab。")

    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    photo_bgr = imread_unicode(args.photo)
    standard_bgr = imread_unicode(args.standard)

    imwrite_unicode(out / "01_original.png", photo_bgr)

    specs = load_specs(args.data, args.count, args.code_prefix)
    (out / "target_specs.json").write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")

    # 1. 色卡四角
    chart_json = out / "chart_corners.json"
    if args.force_select_chart or not chart_json.exists():
        corners = select_four_points(photo_bgr)
        chart_json.write_text(json.dumps(corners.tolist(), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        corners = np.asarray(load_json(chart_json), dtype=np.float64)

    # 2. ColorChecker 校正
    captured_chart, _ = warp_chart_from_photo(photo_bgr, corners, output_size=(600, 400))
    reference_chart = cv2.resize(standard_bgr, (600, 400), interpolation=cv2.INTER_AREA)

    captured_rgb = extract_chart_means(captured_chart)
    reference_rgb = extract_chart_means(reference_chart)

    W = fit_correction_model(
        captured_rgb=captured_rgb,
        reference_rgb=reference_rgb,
        model_type=args.model_type,
        ridge_alpha=args.ridge_alpha,
    )

    corrected_bgr = apply_correction_to_image(photo_bgr, W, model_type=args.model_type)
    imwrite_unicode(out / "02_corrected.png", corrected_bgr)
    imwrite_unicode(out / "02a_captured_chart.png", captured_chart)
    imwrite_unicode(out / "02b_reference_chart.png", reference_chart)

    chart_before_lab = rgb_to_lab(captured_rgb)
    chart_after_rgb = cv2.cvtColor(
        apply_correction_to_image(cv2.cvtColor(captured_rgb.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2BGR), W, args.model_type),
        cv2.COLOR_BGR2RGB,
    ).reshape(-1, 3)
    chart_after_lab = rgb_to_lab(chart_after_rgb)
    chart_ref_lab = rgb_to_lab(reference_rgb)
    chart_report = {
        "model_type": args.model_type,
        "ridge_alpha": args.ridge_alpha,
        "mean_deltaE76_before": float(delta_e_76(chart_before_lab, chart_ref_lab).mean()),
        "mean_deltaE76_after": float(delta_e_76(chart_after_lab, chart_ref_lab).mean()),
        "max_deltaE76_before": float(delta_e_76(chart_before_lab, chart_ref_lab).max()),
        "max_deltaE76_after": float(delta_e_76(chart_after_lab, chart_ref_lab).max()),
        "matrix_shape": list(W.shape),
    }
    (out / "colorchecker_report.json").write_text(json.dumps(chart_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3. 手动画 circles
    circle_json = out / "visual_circles_manual.json"
    old_circles = load_json(circle_json) if circle_json.exists() else None
    if args.force_select_circles or not circle_json.exists():
        circles = select_circles(photo_bgr, specs, old_circles=old_circles)
        circle_json.write_text(json.dumps(circles, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        circles = load_json(circle_json)

    # 4. 采样 corrected_lab
    final_bgr = None
    if args.final_preview is not None:
        if not args.final_preview.exists():
            raise FileNotFoundError(f"找不到 final preview：{args.final_preview}")
        final_bgr = imread_unicode(args.final_preview)

    rows = []
    for i, spec in enumerate(specs):
        circle = circles[i] if i < len(circles) else None
        corrected_rgb, corrected_lab = sample_circle_lab(corrected_bgr, circle)

        visual_rgb = None
        visual_lab = None
        if final_bgr is not None:
            visual_rgb, visual_lab = sample_circle_lab(final_bgr, circle)

        std_lab = spec.get("standard_lab")

        row = {
            "source_project": out.name,
            "source_photo": str(args.photo),
            "index": spec["index"],
            "code": spec["code"],
            "name": spec.get("name", ""),
            "has_standard_lab": std_lab is not None,
            "standard_L": std_lab[0] if std_lab else "",
            "standard_a": std_lab[1] if std_lab else "",
            "standard_b": std_lab[2] if std_lab else "",
            "circle_cx": circle.get("cx", "") if circle else "",
            "circle_cy": circle.get("cy", "") if circle else "",
            "circle_r": circle.get("r", "") if circle else "",
            "corrected_R": corrected_rgb[0] if corrected_rgb is not None else "",
            "corrected_G": corrected_rgb[1] if corrected_rgb is not None else "",
            "corrected_B": corrected_rgb[2] if corrected_rgb is not None else "",
            "corrected_L": corrected_lab[0] if corrected_lab is not None else "",
            "corrected_a": corrected_lab[1] if corrected_lab is not None else "",
            "corrected_b": corrected_lab[2] if corrected_lab is not None else "",
            "visual_display_R": visual_rgb[0] if visual_rgb is not None else "",
            "visual_display_G": visual_rgb[1] if visual_rgb is not None else "",
            "visual_display_B": visual_rgb[2] if visual_rgb is not None else "",
            "visual_display_L": visual_lab[0] if visual_lab is not None else "",
            "visual_display_a": visual_lab[1] if visual_lab is not None else "",
            "visual_display_b": visual_lab[2] if visual_lab is not None else "",
            "final_preview": str(args.final_preview) if args.final_preview else "",
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    corrected_csv = out / "corrected_samples.csv"
    df.to_csv(corrected_csv, index=False, encoding="utf-8-sig")

    if final_bgr is not None:
        # 训练 mapping_T 用这个
        train_df = df[
            (df["corrected_L"] != "")
            & (df["corrected_a"] != "")
            & (df["corrected_b"] != "")
            & (df["visual_display_L"] != "")
            & (df["visual_display_a"] != "")
            & (df["visual_display_b"] != "")
        ].copy()
        train_csv = out / "visual_training_samples.csv"
        train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")

        # 有标准值时，也输出一份可并入正式胶块视觉库的 CSV
        lib_df = train_df[train_df["has_standard_lab"] == True].copy()
        if len(lib_df) > 0:
            lib_csv = out / "glue_visual_library_from_dataset.csv"
            lib_df.to_csv(lib_csv, index=False, encoding="utf-8-sig")
        else:
            lib_csv = ""

    else:
        train_csv = ""
        lib_csv = ""

    summary = {
        "photo": str(args.photo),
        "standard": str(args.standard),
        "data": str(args.data) if args.data else "",
        "count": len(specs),
        "out": str(out),
        "corrected_csv": str(corrected_csv),
        "visual_training_samples_csv": str(train_csv) if train_csv else "",
        "glue_visual_library_from_dataset_csv": str(lib_csv) if lib_csv else "",
        "chart_report": chart_report,
        "note": "如果没有 final-preview，本次只输出 corrected_samples.csv；提供 final-preview 后才会输出 visual_training_samples.csv。",
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print("输出目录：", out)
    print("校正图：", out / "02_corrected.png")
    print("circle：", circle_json)
    print("corrected samples：", corrected_csv)
    if train_csv:
        print("visual training samples：", train_csv)
    if lib_csv:
        print("glue visual library from dataset：", lib_csv)
    print("ColorChecker ΔE76 before/after:",
          f"{chart_report['mean_deltaE76_before']:.3f} -> {chart_report['mean_deltaE76_after']:.3f}")


if __name__ == "__main__":
    main()
