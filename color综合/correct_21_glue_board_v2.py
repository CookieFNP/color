# -*- coding: utf-8 -*-
"""
用途：
    针对“新板材上有 21 个胶块”的图片，只做：
    1）ColorChecker 基础校正
    2）背景大量保留原图（避免整图发白、发雾、发淡）
    3）手动画 21 个圆形胶块 ROI
    4）仅在圆形胶块区域内使用校正后的颜色
    5）输出最终图，方便你后续看哪些胶块还需要定点微调

核心思路：
    final = mostly_original_background + corrected_glue_circles

即：
    - 背景：原图占大权重，校正图占小权重
    - 胶块：圆形 ROI 内使用校正图
    - 不做“暖化”处理
    - 不做 128 库匹配

典型运行：
    python correct_21_glue_board_v2.py --photo board21.jpg --standard standard_chart.png --out output_21_v2

如果想强制重新点色卡、重新画圆：
    python correct_21_glue_board_v2.py --photo board21.jpg --standard standard_chart.png --out output_21_v2 --force-select-chart --force-select-circles

如果想让背景更接近原图：
    python correct_21_glue_board_v2.py --photo board21.jpg --standard standard_chart.png --out output_21_v2 --background-corrected-weight 0.25

说明：
    --background-corrected-weight = 0.35
    表示背景 = 35% corrected + 65% original
    数值越小，背景越接近原图。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# IO
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
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"无法写出图像：{path}")
    buf.tofile(str(path))


# ============================================================
# 色彩数学
# ============================================================

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    x = x / 255.0 if x.max(initial=0) > 1.0 else x
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


# D65
D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
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


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    return srgb_to_linear(rgb) @ SRGB_TO_XYZ.T


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    t = xyz / D65_WHITE

    eps = 216 / 24389
    kappa = 24389 / 27

    f = np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return xyz_to_lab(rgb_to_xyz(rgb))


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    C_bar = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt((C_bar ** 7) / (C_bar ** 7 + 25 ** 7 + 1e-30)))

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(C1p * C2p == 0, 0, dhp)
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)

    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    Lp_bar = (L1 + L2) / 2
    Cp_bar = (C1p + C2p) / 2

    hp_sum = h1p + h2p
    hp_diff = np.abs(h1p - h2p)

    hp_bar = np.where(
        C1p * C2p == 0,
        hp_sum,
        np.where(
            hp_diff <= 180,
            hp_sum / 2,
            np.where(hp_sum < 360, (hp_sum + 360) / 2, (hp_sum - 360) / 2),
        ),
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(hp_bar - 30))
        + 0.24 * np.cos(np.radians(2 * hp_bar))
        + 0.32 * np.cos(np.radians(3 * hp_bar + 6))
        - 0.20 * np.cos(np.radians(4 * hp_bar - 63))
    )

    dtheta = 30 * np.exp(-(((hp_bar - 275) / 25) ** 2))
    Rc = 2 * np.sqrt((Cp_bar ** 7) / (Cp_bar ** 7 + 25 ** 7 + 1e-30))

    Sl = 1 + (0.015 * ((Lp_bar - 50) ** 2)) / np.sqrt(20 + ((Lp_bar - 50) ** 2))
    Sc = 1 + 0.045 * Cp_bar
    Sh = 1 + 0.015 * Cp_bar * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc

    return np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


# ============================================================
# 交互：点色卡四角
# ============================================================

def resize_for_display(img_bgr: np.ndarray, max_w: int = 1400, max_h: int = 900) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    shown = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return shown, scale


def select_four_points(image_bgr: np.ndarray, title: str = "Select ColorChecker corners") -> list[tuple[int, int]]:
    shown, scale = resize_for_display(image_bgr)
    points: list[tuple[int, int]] = []

    def redraw():
        canvas = shown.copy()
        msg = "Click ColorChecker: TL, TR, BR, BL | Enter confirm | R reset | Esc cancel"
        cv2.putText(canvas, msg, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        for i, (x, y) in enumerate(points):
            xs = int(round(x * scale))
            ys = int(round(y * scale))
            cv2.circle(canvas, (xs, ys), 6, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (xs + 8, ys - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(title, canvas)

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(round(x / scale)), int(round(y / scale))))
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse_cb)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in [13, 10] and len(points) == 4:
            break
        if key in [ord("r"), ord("R")]:
            points.clear()
            redraw()
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("用户取消选择色卡四角。")

    cv2.destroyWindow(title)
    return points


# ============================================================
# 交互：画圆形胶块 ROI
# ============================================================

def select_circles(image_bgr: np.ndarray, count: int, title: str = "Draw circles") -> list[dict]:
    """
    操作方式：
        - 左键按下：确定圆心
        - 拖动
        - 左键松开：确定半径
        - Enter 结束当前圆
        - R 重画全部
        - Esc 取消
    """
    shown, scale = resize_for_display(image_bgr)
    circles: list[dict] = []

    drawing = False
    center_disp = None
    radius_disp = 0

    def redraw():
        canvas = shown.copy()
        msg = f"Draw circle {len(circles)+1}/{count} | drag mouse | Enter finish current | R reset all | Esc cancel"
        cv2.putText(canvas, msg, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)

        for i, c in enumerate(circles, start=1):
            x = int(round(c["cx"] * scale))
            y = int(round(c["cy"] * scale))
            r = int(round(c["r"] * scale))
            cv2.circle(canvas, (x, y), r, (0, 0, 255), 2)
            cv2.putText(canvas, str(i), (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if drawing and center_disp is not None and radius_disp > 0:
            cv2.circle(canvas, center_disp, radius_disp, (255, 0, 0), 2)

        cv2.imshow(title, canvas)

    def mouse_cb(event, x, y, flags, param):
        nonlocal drawing, center_disp, radius_disp

        if len(circles) >= count:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            center_disp = (x, y)
            radius_disp = 1
            redraw()

        elif event == cv2.EVENT_MOUSEMOVE and drawing and center_disp is not None:
            radius_disp = int(round(np.sqrt((x - center_disp[0]) ** 2 + (y - center_disp[1]) ** 2)))
            redraw()

        elif event == cv2.EVENT_LBUTTONUP and drawing and center_disp is not None:
            drawing = False
            radius_disp = int(round(np.sqrt((x - center_disp[0]) ** 2 + (y - center_disp[1]) ** 2)))
            if radius_disp > 2:
                circles.append(
                    {
                        "cx": float(center_disp[0] / scale),
                        "cy": float(center_disp[1] / scale),
                        "r": float(radius_disp / scale),
                    }
                )
            center_disp = None
            radius_disp = 0
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse_cb)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in [13, 10] and len(circles) == count:
            break
        if key in [ord("r"), ord("R")]:
            circles.clear()
            drawing = False
            center_disp = None
            radius_disp = 0
            redraw()
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("用户取消选择圆形 ROI。")

    cv2.destroyWindow(title)
    return circles


# ============================================================
# ColorChecker warp / 取色 / 拟合
# ============================================================

def warp_chart(image_bgr: np.ndarray, corners: list[tuple[int, int]], output_size: tuple[int, int] = (600, 400)) -> np.ndarray:
    w, h = output_size
    src = np.asarray(corners, dtype=np.float32)
    dst = np.asarray(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, M, (w, h))


def extract_colorchecker_24_rgb(chart_bgr: np.ndarray, rows: int = 4, cols: int = 6, inner_ratio: float = 0.50) -> np.ndarray:
    h, w = chart_bgr.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    margin = (1.0 - inner_ratio) / 2.0

    rgbs = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(round((c + margin) * cell_w))
            x2 = int(round((c + 1 - margin) * cell_w))
            y1 = int(round((r + margin) * cell_h))
            y2 = int(round((r + 1 - margin) * cell_h))

            patch = chart_bgr[y1:y2, x1:x2]
            rgb = patch[:, :, ::-1].reshape(-1, 3).mean(axis=0)
            rgbs.append(rgb)

    return np.asarray(rgbs, dtype=np.float64)


def build_features(linear_rgb: np.ndarray, model_type: str) -> np.ndarray:
    x = np.asarray(linear_rgb, dtype=np.float64)
    one_dim = x.ndim == 1
    if one_dim:
        x = x.reshape(1, 3)

    R = x[:, 0]
    G = x[:, 1]
    B = x[:, 2]

    if model_type == "linear_bias":
        phi = np.stack([R, G, B, np.ones_like(R)], axis=1)

    elif model_type == "poly2":
        phi = np.stack(
            [R, G, B, R * R, G * G, B * B, R * G, R * B, G * B, np.ones_like(R)],
            axis=1,
        )

    elif model_type == "root_poly2":
        eps = 1e-12
        phi = np.stack(
            [R, G, B, np.sqrt(np.maximum(R * G, eps)), np.sqrt(np.maximum(R * B, eps)), np.sqrt(np.maximum(G * B, eps)), np.ones_like(R)],
            axis=1,
        )

    else:
        raise ValueError(f"未知 model_type: {model_type}")

    return phi[0] if one_dim else phi


def fit_color_correction(captured_rgb: np.ndarray, reference_rgb: np.ndarray, model_type: str = "root_poly2", ridge_alpha: float = 1e-6) -> np.ndarray:
    X = srgb_to_linear(captured_rgb)
    Y = srgb_to_linear(reference_rgb)

    phi = build_features(X, model_type)

    d = phi.shape[1]
    reg = np.eye(d, dtype=np.float64) * ridge_alpha
    reg[-1, -1] = 0.0

    A = phi.T @ phi + reg
    B = phi.T @ Y

    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ B

    return W


def apply_color_correction_image(image_bgr: np.ndarray, W: np.ndarray, model_type: str, correction_strength: float = 1.0) -> np.ndarray:
    rgb = image_bgr[:, :, ::-1].astype(np.float64)
    h, w = rgb.shape[:2]

    lin = srgb_to_linear(rgb.reshape(-1, 3))
    phi = build_features(lin, model_type)
    pred = phi @ W
    pred = np.clip(pred, 0.0, 1.0)

    if correction_strength < 1.0:
        pred = lin * (1 - correction_strength) + pred * correction_strength

    srgb = linear_to_srgb(pred).reshape(h, w, 3)
    return srgb[:, :, ::-1].copy()


# ============================================================
# 融合逻辑
# ============================================================

def blend_background(original_bgr: np.ndarray, corrected_bgr: np.ndarray, corrected_weight: float) -> np.ndarray:
    """
    背景 = corrected_weight * corrected + (1-corrected_weight) * original
    corrected_weight 越小，越接近原图背景
    """
    alpha = float(np.clip(corrected_weight, 0.0, 1.0))
    out = corrected_bgr.astype(np.float32) * alpha + original_bgr.astype(np.float32) * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_circle_mask(shape_hw: tuple[int, int], circles: list[dict], feather: int = 15) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)

    for c in circles:
        cx = int(round(c["cx"]))
        cy = int(round(c["cy"]))
        r = int(round(c["r"]))
        cv2.circle(mask, (cx, cy), r, 1.0, thickness=-1)

    if feather > 0:
        k = max(3, int(feather) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return np.clip(mask, 0.0, 1.0)


def compose_corrected_circles_on_background(original_bgr: np.ndarray, corrected_bgr: np.ndarray, circles: list[dict], background_corrected_weight: float = 0.35, feather: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """
    输出：
        1）背景大多保留原图 + 胶块圆内用 corrected
        2）circle mask 预览
    """
    bg = blend_background(original_bgr, corrected_bgr, corrected_weight=background_corrected_weight)

    mask = make_circle_mask(original_bgr.shape[:2], circles, feather=feather)
    mask3 = mask[:, :, None]

    out = bg.astype(np.float32) * (1 - mask3) + corrected_bgr.astype(np.float32) * mask3
    out = np.clip(out, 0, 255).astype(np.uint8)

    mask_vis = (mask * 255.0).astype(np.uint8)
    return out, mask_vis


def draw_circle_overlay(image_bgr: np.ndarray, circles: list[dict], out_path: Path) -> None:
    canvas = image_bgr.copy()
    for i, c in enumerate(circles, start=1):
        cx = int(round(c["cx"]))
        cy = int(round(c["cy"]))
        r = int(round(c["r"]))
        cv2.circle(canvas, (cx, cy), r, (0, 0, 255), 2)
        cv2.putText(canvas, str(i), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    imwrite_unicode(out_path, canvas)


def make_triptych(original_bgr: np.ndarray, corrected_bgr: np.ndarray, final_bgr: np.ndarray, out_path: Path) -> None:
    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 48), (0, 0, 0), -1)
        cv2.putText(out, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        return out

    imgs = [add_label(original_bgr, "original"), add_label(corrected_bgr, "colorchecker corrected"), add_label(final_bgr, "final: original bg + corrected circles")]

    h_min = min(img.shape[0] for img in imgs)
    resized = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=cv2.INTER_AREA) for img in imgs]
    canvas = np.concatenate(resized, axis=1)
    imwrite_unicode(out_path, canvas)


def stat_pack(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Correct 21 glue circles with mostly original background.")
    parser.add_argument("--photo", required=True, help="包含 ColorChecker 和 21 个胶块的新照片")
    parser.add_argument("--standard", default="standard_chart.png", help="标准 ColorChecker 图")
    parser.add_argument("--out", default="output_21_v2", help="输出目录")

    parser.add_argument("--model-type", choices=["linear_bias", "poly2", "root_poly2"], default="root_poly2")
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--correction-strength", type=float, default=1.0)

    parser.add_argument("--chart-corners-file", default=None, help="可选，复用色卡四角 JSON")
    parser.add_argument("--force-select-chart", action="store_true", help="强制重新点色卡四角")

    parser.add_argument("--circle-count", type=int, default=21, help="默认 21 个圆形胶块 ROI")
    parser.add_argument("--circles-file", default=None, help="可选，复用圆形 ROI JSON")
    parser.add_argument("--force-select-circles", action="store_true", help="强制重新画圆")

    parser.add_argument("--background-corrected-weight", type=float, default=0.35, help="背景中 corrected 的权重。越小越接近原图，默认 0.35")
    parser.add_argument("--circle-feather", type=int, default=15, help="圆形边缘羽化，默认 15")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    photo_path = Path(args.photo)
    standard_path = Path(args.standard)

    original_bgr = imread_unicode(photo_path)
    standard_bgr = imread_unicode(standard_path)

    # 1) 色卡四角
    chart_corners_file = Path(args.chart_corners_file) if args.chart_corners_file else out_dir / "chart_corners.json"
    if chart_corners_file.exists() and not args.force_select_chart:
        corners = json.loads(chart_corners_file.read_text(encoding="utf-8"))
        corners = [tuple(map(int, p)) for p in corners]
        print("已加载色卡四角：", chart_corners_file)
    else:
        print("\n请依次点击 ColorChecker 四角：左上、右上、右下、左下")
        corners = select_four_points(original_bgr)
        chart_corners_file.write_text(json.dumps(corners, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已保存色卡四角：", chart_corners_file)

    # 2) 基础校正
    chart_warp = warp_chart(original_bgr, corners, output_size=(600, 400))
    standard_chart = cv2.resize(standard_bgr, (600, 400), interpolation=cv2.INTER_AREA)

    imwrite_unicode(out_dir / "01_chart_warp.png", chart_warp)
    imwrite_unicode(out_dir / "01_standard_chart_resized.png", standard_chart)

    captured_rgb = extract_colorchecker_24_rgb(chart_warp)
    reference_rgb = extract_colorchecker_24_rgb(standard_chart)

    W = fit_color_correction(captured_rgb, reference_rgb, model_type=args.model_type, ridge_alpha=args.ridge_alpha)
    corrected_bgr = apply_color_correction_image(original_bgr, W=W, model_type=args.model_type, correction_strength=args.correction_strength)
    imwrite_unicode(out_dir / "02_colorchecker_corrected.png", corrected_bgr)

    # 3) 色卡 before/after ΔE
    corrected_chart = warp_chart(corrected_bgr, corners, output_size=(600, 400))
    corrected_rgb = extract_colorchecker_24_rgb(corrected_chart)

    ref_lab = rgb_to_lab(reference_rgb)
    cap_lab = rgb_to_lab(captured_rgb)
    fix_lab = rgb_to_lab(corrected_rgb)

    de_before = delta_e_2000(cap_lab, ref_lab)
    de_after = delta_e_2000(fix_lab, ref_lab)

    # 4) 画 21 个圆形 ROI
    circles_file = Path(args.circles_file) if args.circles_file else out_dir / "glue_circles.json"
    if circles_file.exists() and not args.force_select_circles:
        circles = json.loads(circles_file.read_text(encoding="utf-8"))
        print("已加载圆形 ROI：", circles_file)
    else:
        print(f"\n请依次画 {args.circle_count} 个圆形胶块 ROI。")
        circles = select_circles(corrected_bgr, count=args.circle_count)
        circles_file.write_text(json.dumps(circles, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已保存圆形 ROI：", circles_file)

    draw_circle_overlay(original_bgr, circles, out_dir / "03_circles_on_original.png")
    draw_circle_overlay(corrected_bgr, circles, out_dir / "03_circles_on_corrected.png")

    # 5) 背景偏原图，圆内用 corrected
    final_bgr, mask_vis = compose_corrected_circles_on_background(
        original_bgr=original_bgr,
        corrected_bgr=corrected_bgr,
        circles=circles,
        background_corrected_weight=args.background_corrected_weight,
        feather=args.circle_feather,
    )

    bg_blend_bgr = blend_background(original_bgr, corrected_bgr, corrected_weight=args.background_corrected_weight)

    imwrite_unicode(out_dir / "04_background_blend.png", bg_blend_bgr)
    imwrite_unicode(out_dir / "05_circle_mask.png", mask_vis)
    imwrite_unicode(out_dir / "06_final_corrected_circles_on_original_bg.png", final_bgr)
    make_triptych(original_bgr, corrected_bgr, final_bgr, out_dir / "07_triptych.png")

    report = {
        "input": {
            "photo": str(photo_path),
            "standard": str(standard_path),
        },
        "colorchecker": {
            "model_type": args.model_type,
            "ridge_alpha": args.ridge_alpha,
            "correction_strength": args.correction_strength,
            "deltaE_before": stat_pack(de_before),
            "deltaE_after": stat_pack(de_after),
        },
        "circle_roi": {
            "count": args.circle_count,
            "circles_file": str(circles_file),
            "feather": args.circle_feather,
        },
        "background_blend": {
            "background_corrected_weight": args.background_corrected_weight,
            "note": "final background = weight * corrected + (1-weight) * original",
        },
        "outputs": {
            "colorchecker_corrected": str(out_dir / "02_colorchecker_corrected.png"),
            "background_blend": str(out_dir / "04_background_blend.png"),
            "circle_mask": str(out_dir / "05_circle_mask.png"),
            "final": str(out_dir / "06_final_corrected_circles_on_original_bg.png"),
            "triptych": str(out_dir / "07_triptych.png"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== 处理完成 ====")
    print("输出目录：", out_dir)
    print("\nColorChecker ΔE2000:")
    print("  before mean = {:.3f}, p95 = {:.3f}, max = {:.3f}".format(
        report["colorchecker"]["deltaE_before"]["mean"],
        report["colorchecker"]["deltaE_before"]["p95"],
        report["colorchecker"]["deltaE_before"]["max"],
    ))
    print("  after  mean = {:.3f}, p95 = {:.3f}, max = {:.3f}".format(
        report["colorchecker"]["deltaE_after"]["mean"],
        report["colorchecker"]["deltaE_after"]["p95"],
        report["colorchecker"]["deltaE_after"]["max"],
    ))

    print("\n主要输出：")
    print("  纯基础校正图：", out_dir / "02_colorchecker_corrected.png")
    print("  背景混合图：", out_dir / "04_background_blend.png")
    print("  最终图（原图背景 + 校正胶块圆）：", out_dir / "06_final_corrected_circles_on_original_bg.png")
    print("  三联图：", out_dir / "07_triptych.png")
    print("  report：", out_dir / "report.json")


if __name__ == "__main__":
    main()
