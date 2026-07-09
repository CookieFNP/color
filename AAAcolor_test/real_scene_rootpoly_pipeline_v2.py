from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage import color


# =========================
# 基础色彩数学
# =========================

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    rgb = rgb / 255.0 if rgb.size and rgb.max() > 1.0 else rgb
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.asarray(linear, dtype=np.float64)
    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * (linear ** (1.0 / 2.4)) - 0.055)
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def bgr_to_lab_image(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    return color.rgb2lab(rgb)


def rgb_u8_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_u8, dtype=np.float64)
    if rgb.ndim == 1:
        rgb = rgb[None, :]
    rgb = np.clip(rgb / 255.0, 0, 1)
    return color.rgb2lab(rgb)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    return color.deltaE_ciede2000(lab1, lab2)


def lab_clip(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64).copy()
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    lab[..., 1] = np.clip(lab[..., 1], -128, 127)
    lab[..., 2] = np.clip(lab[..., 2], -128, 127)
    return lab


def lab_chroma(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


# =========================
# root_poly2 模型
# =========================

def build_root_poly2_features(linear_rgb: np.ndarray) -> np.ndarray:
    """
    root_poly2:
    [R, G, B, sqrt(RG), sqrt(RB), sqrt(GB), 1]
    """
    x = np.asarray(linear_rgb, dtype=np.float64)
    if x.shape[-1] != 3:
        raise ValueError("linear_rgb 最后一维必须是 3")

    R = x[..., 0]
    G = x[..., 1]
    B = x[..., 2]

    eps = 0.0
    feats = [
        R,
        G,
        B,
        np.sqrt(np.clip(R * G, eps, None)),
        np.sqrt(np.clip(R * B, eps, None)),
        np.sqrt(np.clip(G * B, eps, None)),
        np.ones_like(R),
    ]
    return np.stack(feats, axis=-1)


def fit_root_poly2(captured_rgb_u8: np.ndarray, reference_rgb_u8: np.ndarray, ridge_alpha: float = 1e-6) -> np.ndarray:
    captured_lin = srgb_to_linear(captured_rgb_u8)
    reference_lin = srgb_to_linear(reference_rgb_u8)

    X = build_root_poly2_features(captured_lin)
    Y = reference_lin

    reg = ridge_alpha * np.eye(X.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0  # bias 一般不做正则
    W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    return W


def apply_root_poly2_to_image(bgr: np.ndarray, W: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lin = srgb_to_linear(rgb)
    h, w = lin.shape[:2]

    X = build_root_poly2_features(lin.reshape(-1, 3))
    out_lin = X @ W
    out_rgb = linear_to_srgb(out_lin.reshape(h, w, 3))

    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


# =========================
# ColorChecker 处理
# =========================

def order_corners_tl_tr_br_bl(points: list[list[float]] | np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("ColorChecker 四角必须是 4x2")

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_chart(photo_bgr: np.ndarray, corners: np.ndarray, width: int = 600, height: int = 400) -> np.ndarray:
    src = order_corners_tl_tr_br_bl(corners)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(photo_bgr, M, (width, height))


def extract_chart_rgb_means(
    chart_bgr: np.ndarray,
    rows: int = 4,
    cols: int = 6,
    inner_ratio: float = 0.55,
) -> np.ndarray:
    """
    从已经对齐的 4x6 色卡图里提取每个色块中心区域 RGB 均值。
    返回顺序：从左到右，从上到下，共 24 个 RGB。
    """
    h, w = chart_bgr.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    means = []

    for r in range(rows):
        for c in range(cols):
            cx1 = c * cell_w
            cy1 = r * cell_h
            cx2 = (c + 1) * cell_w
            cy2 = (r + 1) * cell_h

            mx = (1.0 - inner_ratio) * cell_w / 2.0
            my = (1.0 - inner_ratio) * cell_h / 2.0

            x1 = int(round(cx1 + mx))
            x2 = int(round(cx2 - mx))
            y1 = int(round(cy1 + my))
            y2 = int(round(cy2 - my))

            patch = chart_bgr[y1:y2, x1:x2]
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            means.append(np.mean(patch_rgb.reshape(-1, 3), axis=0))

    return np.asarray(means, dtype=np.float64)


# =========================
# 真实场景二阶段：ColorChecker residual model
# =========================

def residual_features_lab(lab: np.ndarray) -> np.ndarray:
    """
    只用 ColorChecker 24 点拟合 residual，不用目标胶块标准值。

    输入 Lab，输出轻量特征：
    [1, L/100, a/128, b/128, C/128, sin(h), cos(h)]
    """
    lab = np.asarray(lab, dtype=np.float64)
    if lab.ndim == 1:
        lab = lab[None, :]

    L = lab[:, 0] / 100.0
    a = lab[:, 1] / 128.0
    b = lab[:, 2] / 128.0
    C = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
    Cn = C / 128.0
    h = np.arctan2(lab[:, 2], lab[:, 1])

    X = np.stack(
        [
            np.ones_like(L),
            L,
            a,
            b,
            Cn,
            np.sin(h),
            np.cos(h),
        ],
        axis=1,
    )
    return X


def fit_chart_residual_model(
    corrected_chart_lab: np.ndarray,
    reference_chart_lab: np.ndarray,
    ridge_alpha: float = 1e-2,
) -> np.ndarray:
    """
    学习 root_poly2 校正后，ColorChecker 在 Lab 空间还剩多少残差：
    input: corrected_chart_lab
    output: reference_chart_lab - corrected_chart_lab
    """
    X = residual_features_lab(corrected_chart_lab)
    Y = np.asarray(reference_chart_lab, dtype=np.float64) - np.asarray(corrected_chart_lab, dtype=np.float64)

    reg = ridge_alpha * np.eye(X.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    return W


def predict_chart_residual(lab: np.ndarray, W: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    shape = lab.shape
    X = residual_features_lab(lab.reshape(-1, 3))
    pred = X @ W
    return pred.reshape(shape)


# =========================
# 真实场景二阶段：局部背景校正
# =========================

def normalize_roi(roi: list[int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def shrink_roi(roi: tuple[int, int, int, int], ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nw = w * ratio
    nh = h * ratio
    return (
        int(round(cx - nw / 2)),
        int(round(cy - nh / 2)),
        int(round(cx + nw / 2)),
        int(round(cy + nh / 2)),
    )


def clamp_roi(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def robust_lab_from_roi(
    bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    inner_ratio: float = 0.65,
    keep_ratio: float = 0.75,
) -> np.ndarray:
    h, w = bgr.shape[:2]
    roi = clamp_roi(normalize_roi(roi), w, h)
    roi = shrink_roi(roi, inner_ratio)
    roi = clamp_roi(roi, w, h)

    x1, y1, x2, y2 = roi
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"ROI 无效: {roi}")

    lab = bgr_to_lab_image(crop).reshape(-1, 3)
    med = np.median(lab, axis=0)
    dist = np.linalg.norm(lab - med[None, :], axis=1)

    n_keep = max(10, int(len(lab) * keep_ratio))
    n_keep = min(n_keep, len(lab))
    idx = np.argsort(dist)[:n_keep]

    return np.mean(lab[idx], axis=0)


def create_target_mask(shape: tuple[int, int], rois: list[tuple[int, int, int, int]], pad: int = 2) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for roi in rois:
        x1, y1, x2, y2 = clamp_roi(normalize_roi(roi), w, h)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        mask[y1:y2, x1:x2] = 1
    return mask



def local_background_lab(
    corrected_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    all_target_mask: np.ndarray | None = None,
    margin: int = 36,
    min_pixels: int = 80,
    reference_bg_lab: np.ndarray | None = None,
    bright_percentile: float = 70.0,
    max_chroma: float = 18.0,
) -> np.ndarray | None:
    """
    在 ROI 周围取一圈 ring 作为局部背景。

    v2 改动：
    - 背景不是纯白时，可以传入已知背景标准 Lab，例如 84.71,-1.14,-3.64。
    - 优先取较亮、低色度、且接近已知背景色相/色度的像素。
    - 尽量排除胶块投影阴影和邻近彩色胶块污染。
    """
    h, w = corrected_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_roi(normalize_roi(roi), w, h)

    ex1 = max(0, x1 - margin)
    ey1 = max(0, y1 - margin)
    ex2 = min(w, x2 + margin)
    ey2 = min(h, y2 + margin)

    if ex2 <= ex1 or ey2 <= ey1:
        return None

    ring_mask = np.ones((ey2 - ey1, ex2 - ex1), dtype=bool)

    # 排除目标自身
    rx1 = x1 - ex1
    ry1 = y1 - ey1
    rx2 = x2 - ex1
    ry2 = y2 - ey1
    ring_mask[ry1:ry2, rx1:rx2] = False

    # 排除其它已选胶块区域
    if all_target_mask is not None:
        sub_mask = all_target_mask[ey1:ey2, ex1:ex2].astype(bool)
        ring_mask[sub_mask] = False

    patch = corrected_bgr[ey1:ey2, ex1:ex2]
    if patch.size == 0 or ring_mask.sum() < min_pixels:
        return None

    # 避免把过曝高光当背景
    rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    rgb_pixels = rgb_patch[ring_mask]
    not_saturated = np.max(rgb_pixels, axis=1) < 250

    lab = bgr_to_lab_image(patch)
    pixels = lab[ring_mask]
    pixels = pixels[not_saturated]

    if len(pixels) < min_pixels:
        return None

    L = pixels[:, 0]
    C = lab_chroma(pixels)

    # 1. 先取较亮区域，尽量避开投影阴影。
    #    不是取最亮 1%，因为可能是高光/反光；取 top 30% 左右更稳。
    L_th = np.percentile(L, bright_percentile)
    keep = L >= L_th

    # 2. 背景是低色度材料，因此排除明显彩色污染。
    keep &= C <= max_chroma

    # 3. 如果提供了背景标准 Lab，再选更接近背景 a/b 的像素。
    if reference_bg_lab is not None:
        ref = np.asarray(reference_bg_lab, dtype=np.float64)
        ab_dist = np.linalg.norm(pixels[:, 1:3] - ref[None, 1:3], axis=1)

        # 不能卡太死，否则某些阴影区没有点；取接近背景色相的 60%
        ab_th = np.percentile(ab_dist, 60)
        keep &= ab_dist <= ab_th

    if keep.sum() < min_pixels:
        # 放宽：只要求较亮 + 不过分彩色
        keep = (L >= np.percentile(L, 55)) & (C <= np.percentile(C, 75))

    if keep.sum() < min_pixels:
        # 再兜底：取 L 较高的一部分
        keep = L >= np.percentile(L, 65)

    kept = pixels[keep]
    if len(kept) < min_pixels:
        return None

    # 稳健均值：去掉离中位数远的点
    med = np.median(kept, axis=0)
    dist = np.linalg.norm(kept - med[None, :], axis=1)
    n_keep = max(min_pixels, int(len(kept) * 0.70))
    n_keep = min(n_keep, len(kept))
    idx = np.argsort(dist)[:n_keep]
    return np.mean(kept[idx], axis=0)


def clamp_shift(shift: np.ndarray, cap: tuple[float, float, float]) -> np.ndarray:
    cap_arr = np.asarray(cap, dtype=np.float64)
    return np.clip(shift, -cap_arr, cap_arr)


def limit_shift_by_color_family(base_lab: np.ndarray, shift: np.ndarray) -> tuple[np.ndarray, str]:
    """
    色系限制：不使用目标标准值，只根据当前观测 Lab 的区域来限制补偿方向和幅度。
    """
    L, a, b = [float(x) for x in base_lab]
    C = math.sqrt(a * a + b * b)
    s = np.asarray(shift, dtype=np.float64).copy()

    family = "normal"

    if C < 6:
        family = "gray"
        s[1] *= 0.25
        s[2] *= 0.25
        s = clamp_shift(s, (7, 2, 2))

    elif a > 8 and b > -5:
        family = "red_orange_brown"
        # 红橙棕里常见问题是 a/b 不足，但 L 不能被过度拉亮。
        if L > 50 and s[0] > 0:
            s[0] *= 0.45
        s = clamp_shift(s, (8, 12, 14))

    elif b > 18:
        family = "yellow"
        # 黄色主要允许补 b，a 不要乱动太大。
        s[1] *= 0.65
        s = clamp_shift(s, (9, 6, 16))

    elif L > 70:
        family = "light"
        s[1] *= 0.45
        s[2] *= 0.65
        s = clamp_shift(s, (7, 4, 8))

    elif L < 42:
        family = "dark"
        s[0] = np.clip(s[0], -5, 8)
        s[1] = np.clip(s[1], -7, 7)
        s[2] = np.clip(s[2], -8, 10)

    else:
        s = clamp_shift(s, (8, 8, 10))

    return s, family


# =========================
# 标准色读取与最近邻分类
# =========================

@dataclass
class StandardColor:
    code: str
    name: str
    lab: np.ndarray


def parse_lab_values(text: str) -> list[float]:
    text = text.strip().strip('"').strip("'")
    parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Lab 不是 3 个数: {text}")
    return [float(p) for p in parts]


def read_standards_csv(path: str | Path) -> list[StandardColor]:
    path = Path(path)
    standards: list[StandardColor] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue

            code = row[0].strip()
            name = row[1].strip()

            if code.lower() in {"code", "编号"}:
                continue

            # 兼容：
            # W126,栗棕色,"48.03, 1.71, 6.3"
            # W126,栗棕色,48.03,1.71,6.3
            lab_text = ",".join(row[2:])
            try:
                lab = np.asarray(parse_lab_values(lab_text), dtype=np.float64)
            except Exception:
                continue

            standards.append(StandardColor(code=code, name=name, lab=lab))

    if not standards:
        raise RuntimeError(f"没有从标准 CSV 读到颜色: {path}")

    return standards


def nearest_standards(lab: np.ndarray, standards: list[StandardColor], top_k: int = 3) -> list[dict[str, Any]]:
    labs = np.asarray([s.lab for s in standards], dtype=np.float64)
    de = delta_e_2000(np.asarray(lab, dtype=np.float64)[None, :], labs)
    order = np.argsort(de)[:top_k]

    out = []
    for i in order:
        s = standards[int(i)]
        out.append(
            {
                "code": s.code,
                "name": s.name,
                "lab": [float(x) for x in s.lab],
                "delta_e_2000": float(de[int(i)]),
            }
        )
    return out


def standard_by_code(standards: list[StandardColor]) -> dict[str, StandardColor]:
    return {s.code.upper(): s for s in standards}


# =========================
# 屏幕适配 ROI / 点选
# =========================

def resize_for_view(img: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(float(max_w) / max(w, 1), float(max_h) / max(h, 1), 1.0)
    if scale >= 0.999:
        return img.copy(), 1.0
    view = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return view, scale


def draw_text_panel(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    x, y = 14, 28
    line_h = 28

    # 画半透明背景
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], min(out.shape[0], 12 + line_h * len(lines))), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

    for i, line in enumerate(lines):
        cv2.putText(out, line, (x, y + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def select_chart_corners_scaled(img_bgr: np.ndarray, max_w: int = 1450, max_h: int = 820) -> np.ndarray:
    view, scale = resize_for_view(img_bgr, max_w, max_h)
    points: list[tuple[int, int]] = []
    win = "select ColorChecker corners: TL, TR, BR, BL"

    def redraw() -> np.ndarray:
        canvas = view.copy()
        for i, p in enumerate(points):
            cv2.circle(canvas, p, 6, (0, 255, 255), -1)
            cv2.putText(canvas, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], (0, 255, 255), 2)
        canvas = draw_text_panel(
            canvas,
            [
                "Click ColorChecker corners in order: TL, TR, BR, BL",
                "r: reset    Enter/Space: confirm after 4 points    Esc: cancel",
                f"view scale = {scale:.3f}",
            ],
        )
        return canvas

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, redraw())
        key = cv2.waitKey(20) & 0xFF

        if key in (27,):  # Esc
            cv2.destroyWindow(win)
            raise RuntimeError("已取消色卡四角选择")

        if key in (ord("r"), ord("R")):
            points = []

        if key in (13, 10, 32):  # Enter / Space
            if len(points) == 4:
                break

    cv2.destroyWindow(win)
    pts = np.asarray(points, dtype=np.float32) / float(scale)
    return order_corners_tl_tr_br_bl(pts)


def draw_existing_rois(view: np.ndarray, rois: list[dict[str, Any]], scale: float) -> np.ndarray:
    canvas = view.copy()
    for item in rois:
        x1, y1, x2, y2 = normalize_roi(item["roi"])
        p1 = (int(round(x1 * scale)), int(round(y1 * scale)))
        p2 = (int(round(x2 * scale)), int(round(y2 * scale)))
        cv2.rectangle(canvas, p1, p2, (0, 255, 255), 2)
        cv2.putText(
            canvas,
            str(item.get("code", "")),
            (p1[0] + 3, max(20, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def select_one_roi_scaled(
    img_bgr: np.ndarray,
    code: str,
    idx: int,
    total: int,
    existing: list[dict[str, Any]],
    max_w: int,
    max_h: int,
) -> tuple[int, int, int, int]:
    view, scale = resize_for_view(img_bgr, max_w, max_h)
    view = draw_existing_rois(view, existing, scale)
    view = draw_text_panel(
        view,
        [
            f"Select ROI for {idx}/{total}: {code}",
            "Drag rectangle, then press Enter/Space. Press c/Esc to cancel current.",
            f"view scale = {scale:.3f}; coordinates will be converted to original image.",
        ],
    )

    win = f"select ROI {idx:03d}_{code}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    rect = cv2.selectROI(win, view, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)

    x, y, w, h = [int(v) for v in rect]
    if w <= 0 or h <= 0:
        raise RuntimeError(f"ROI 选择无效: {code}")

    x1 = int(round(x / scale))
    y1 = int(round(y / scale))
    x2 = int(round((x + w) / scale))
    y2 = int(round((y + h) / scale))
    return normalize_roi((x1, y1, x2, y2))


# =========================
# 命令行与工具
# =========================

def parse_code_range(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    # auto:1-128
    if text.lower().startswith("auto:"):
        rng = text.split(":", 1)[1]
        a, b = rng.split("-", 1)
        return [f"W{i:03d}" for i in range(int(a), int(b) + 1)]

    # W001-W128
    if "-" in text and "," not in text:
        a, b = text.split("-", 1)
        a_num = int("".join(ch for ch in a if ch.isdigit()))
        b_num = int("".join(ch for ch in b if ch.isdigit()))
        return [f"W{i:03d}" for i in range(a_num, b_num + 1)]

    return [x.strip().upper() for x in text.replace("，", ",").split(",") if x.strip()]


def prompt_target_codes() -> list[str]:
    n = int(input("请输入要取的胶块数量，例如 112 或 128：").strip())
    print("接下来输入每个胶块编号。直接回车表示按 W001、W002... 自动递增。")
    codes = []
    for i in range(1, n + 1):
        default = f"W{i:03d}"
        s = input(f"[{i}/{n}] code [{default}]: ").strip().upper()
        codes.append(s if s else default)
    return codes


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.replace("，", ",").split(",") if x.strip()]


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stat_pack(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "max": None, "p95": None, "std": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
    }


def candidate_name(bg_strength: float, residual_strength: float, family_limit: bool) -> str:
    name = f"bg{bg_strength:.2f}_res{residual_strength:.2f}"
    name = name.replace(".", "p")
    if family_limit:
        name += "_family"
    if bg_strength == 0 and residual_strength == 0:
        name = "root_only"
    return name


def evaluate_candidate(
    *,
    candidate: dict[str, Any],
    target_infos: list[dict[str, Any]],
    standards: list[StandardColor],
    code_map: dict[str, StandardColor],
    global_bg_lab: np.ndarray,
    residual_W_lab: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bg_strength = float(candidate["bg_strength"])
    residual_strength = float(candidate["residual_strength"])
    family_limit = bool(candidate["family_limit"])

    bg_gains = np.asarray([args.bg_gain_L, args.bg_gain_a, args.bg_gain_b], dtype=np.float64)
    bg_cap = (args.bg_cap_L, args.bg_cap_a, args.bg_cap_b)
    residual_cap = (args.res_cap_L, args.res_cap_a, args.res_cap_b)

    rows = []
    de_before = []
    de_root = []
    de_final = []
    correct_flags = []
    harm_flags = []

    for info in target_infos:
        code = info["code"].upper()
        std = code_map.get(code)

        raw_lab = np.asarray(info["raw_lab"], dtype=np.float64)
        root_lab = np.asarray(info["root_lab"], dtype=np.float64)
        local_bg = np.asarray(info["local_bg_lab"], dtype=np.float64) if info.get("local_bg_lab") is not None else None

        bg_shift = np.zeros(3, dtype=np.float64)
        if local_bg is not None and bg_strength != 0:
            bg_shift = bg_strength * bg_gains * (global_bg_lab - local_bg)
            bg_shift = clamp_shift(bg_shift, bg_cap)

        lab_bg = lab_clip(root_lab + bg_shift)

        residual_shift = np.zeros(3, dtype=np.float64)
        if residual_strength != 0:
            pred = predict_chart_residual(lab_bg[None, :], residual_W_lab)[0]
            residual_shift = residual_strength * pred
            residual_shift = clamp_shift(residual_shift, residual_cap)

        total_shift = bg_shift + residual_shift

        family = "disabled"
        if family_limit and np.linalg.norm(total_shift) > 1e-9:
            total_shift, family = limit_shift_by_color_family(root_lab, total_shift)

        final_lab = lab_clip(root_lab + total_shift)

        nearest = nearest_standards(final_lab, standards, top_k=3)
        pred = nearest[0]

        if std is not None:
            before_de = float(delta_e_2000(raw_lab[None, :], std.lab[None, :])[0])
            root_de = float(delta_e_2000(root_lab[None, :], std.lab[None, :])[0])
            final_de = float(delta_e_2000(final_lab[None, :], std.lab[None, :])[0])
            correct = pred["code"].upper() == code
            harm = final_de > before_de
        else:
            before_de = np.nan
            root_de = np.nan
            final_de = np.nan
            correct = False
            harm = False

        de_before.append(before_de)
        de_root.append(root_de)
        de_final.append(final_de)
        correct_flags.append(bool(correct))
        harm_flags.append(bool(harm))

        top2_margin = None
        if len(nearest) >= 2:
            top2_margin = float(nearest[1]["delta_e_2000"] - nearest[0]["delta_e_2000"])

        confidence = "ok"
        if pred["delta_e_2000"] > args.low_conf_de or (top2_margin is not None and top2_margin < args.low_conf_margin):
            confidence = "low"

        rows.append(
            {
                "idx": info["idx"],
                "code": code,
                "name": std.name if std is not None else "",
                "roi": json.dumps(info["roi"], ensure_ascii=False),
                "raw_L": float(raw_lab[0]),
                "raw_a": float(raw_lab[1]),
                "raw_b": float(raw_lab[2]),
                "root_L": float(root_lab[0]),
                "root_a": float(root_lab[1]),
                "root_b": float(root_lab[2]),
                "local_bg_L": float(local_bg[0]) if local_bg is not None else "",
                "local_bg_a": float(local_bg[1]) if local_bg is not None else "",
                "local_bg_b": float(local_bg[2]) if local_bg is not None else "",
                "bg_shift_L": float(bg_shift[0]),
                "bg_shift_a": float(bg_shift[1]),
                "bg_shift_b": float(bg_shift[2]),
                "residual_shift_L": float(residual_shift[0]),
                "residual_shift_a": float(residual_shift[1]),
                "residual_shift_b": float(residual_shift[2]),
                "total_shift_L": float(total_shift[0]),
                "total_shift_a": float(total_shift[1]),
                "total_shift_b": float(total_shift[2]),
                "final_L": float(final_lab[0]),
                "final_a": float(final_lab[1]),
                "final_b": float(final_lab[2]),
                "before_deltaE": before_de,
                "root_deltaE": root_de,
                "final_deltaE": final_de,
                "root_improvement": before_de - root_de if np.isfinite(before_de) and np.isfinite(root_de) else "",
                "final_improvement": before_de - final_de if np.isfinite(before_de) and np.isfinite(final_de) else "",
                "pred_code": pred["code"],
                "pred_name": pred["name"],
                "pred_deltaE": pred["delta_e_2000"],
                "top2_code": nearest[1]["code"] if len(nearest) > 1 else "",
                "top2_name": nearest[1]["name"] if len(nearest) > 1 else "",
                "top2_deltaE": nearest[1]["delta_e_2000"] if len(nearest) > 1 else "",
                "top2_margin": top2_margin if top2_margin is not None else "",
                "correct": correct,
                "harm": harm,
                "confidence": confidence,
                "family": family,
            }
        )

    eval_count = min(args.eval_count if args.eval_count else len(rows), len(rows))
    rows_eval = rows[:eval_count]

    valid_before = [float(r["before_deltaE"]) for r in rows_eval if np.isfinite(float(r["before_deltaE"]))]
    valid_root = [float(r["root_deltaE"]) for r in rows_eval if np.isfinite(float(r["root_deltaE"]))]
    valid_final = [float(r["final_deltaE"]) for r in rows_eval if np.isfinite(float(r["final_deltaE"]))]

    summary = {
        "candidate": candidate_name(bg_strength, residual_strength, family_limit),
        "bg_strength": bg_strength,
        "residual_strength": residual_strength,
        "family_limit": family_limit,
        "eval_count": eval_count,
        "before_mean_deltaE": stat_pack(valid_before)["mean"],
        "before_median_deltaE": stat_pack(valid_before)["median"],
        "before_max_deltaE": stat_pack(valid_before)["max"],
        "root_mean_deltaE": stat_pack(valid_root)["mean"],
        "root_median_deltaE": stat_pack(valid_root)["median"],
        "root_max_deltaE": stat_pack(valid_root)["max"],
        "final_mean_deltaE": stat_pack(valid_final)["mean"],
        "final_median_deltaE": stat_pack(valid_final)["median"],
        "final_max_deltaE": stat_pack(valid_final)["max"],
        "final_p95_deltaE": stat_pack(valid_final)["p95"],
        "harm_count": int(sum(bool(r["harm"]) for r in rows_eval)),
        "harm_rate": float(sum(bool(r["harm"]) for r in rows_eval) / max(eval_count, 1)),
        "classification_acc": float(sum(bool(r["correct"]) for r in rows_eval) / max(eval_count, 1)),
        "low_conf_count": int(sum(r["confidence"] == "low" for r in rows_eval)),
    }

    return summary, rows


def draw_result_overlay(photo_bgr: np.ndarray, rows: list[dict[str, Any]], out_path: Path) -> None:
    canvas = photo_bgr.copy()
    for r in rows:
        roi = json.loads(r["roi"]) if isinstance(r["roi"], str) else r["roi"]
        x1, y1, x2, y2 = normalize_roi(roi)
        correct = bool(r["correct"])
        color_box = (0, 220, 0) if correct else (0, 0, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_box, 3)
        text = f'{r["code"]}->{r["pred_code"]} E={float(r["final_deltaE"]):.1f}'
        cv2.putText(canvas, text, (x1 + 2, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_box, 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="真实场景 root_poly2 + 已知背景/局部背景 + ColorChecker residual 校正，不用目标胶块标准值反推补偿。")

    parser.add_argument("--photo", required=True, help="待测整图，例如 pic_all.jpg")
    parser.add_argument("--standard-chart", required=True, help="标准 ColorChecker 图片，例如 standard_chart.png")
    parser.add_argument("--standards-csv", required=True, help='128 标准 Lab CSV，例如 data.csv，格式 W126,栗棕色,"48.03, 1.71, 6.3"')
    parser.add_argument("--out", default="output_real_scene", help="输出目录")

    parser.add_argument("--target-codes", default="", help='目标编号，支持 W001,W002 或 auto:1-112 或 W001-W112。不填则手动输入。')
    parser.add_argument("--eval-count", type=int, default=0, help="评估前 N 个目标；比如后16个瞎取时填 112。0 表示全部。")
    parser.add_argument("--roi-json", default="", help="复用 ROI JSON")
    parser.add_argument("--chart-corners-json", default="", help="复用色卡四角 JSON")

    parser.add_argument("--view-max-w", type=int, default=1450, help="取点/取框显示最大宽度，避免超出屏幕")
    parser.add_argument("--view-max-h", type=int, default=820, help="取点/取框显示最大高度，避免底部看不到")

    parser.add_argument("--chart-width", type=int, default=600)
    parser.add_argument("--chart-height", type=int, default=400)
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--residual-ridge", type=float, default=1e-2)

    parser.add_argument("--roi-inner-ratio", type=float, default=0.65, help="取胶块中心区域比例，避免边缘反光")
    parser.add_argument("--roi-keep-ratio", type=float, default=0.75, help="ROI 内保留最接近中位数的像素比例")
    parser.add_argument("--bg-margin", type=int, default=36, help="局部背景 ring 外扩像素")
    parser.add_argument("--background-lab", default="", help='已知背景标准 Lab，例如 "84.71,-1.14,-3.64"；不填则用局部背景中位数作参考')
    parser.add_argument("--bg-bright-percentile", type=float, default=70.0, help="局部背景取较亮像素的百分位阈值，默认 70，越高越排除阴影")
    parser.add_argument("--bg-max-chroma", type=float, default=18.0, help="背景候选最大色度，默认 18，越小越排除彩色污染")
    parser.add_argument("--target-mask-pad", type=int, default=4)

    parser.add_argument("--bg-strength-list", default="0,0.6,0.8", help="局部背景校正强度列表")
    parser.add_argument("--residual-strength-list", default="0,0.5,0.8", help="ColorChecker residual 强度列表")
    parser.add_argument("--disable-family-limit", action="store_true", help="关闭按色系限制补偿")
    parser.add_argument("--select-metric", default="final_mean_deltaE", choices=["final_mean_deltaE", "final_max_deltaE", "classification_acc"])

    parser.add_argument("--bg-gain-L", type=float, default=0.75)
    parser.add_argument("--bg-gain-a", type=float, default=0.35)
    parser.add_argument("--bg-gain-b", type=float, default=0.90)

    parser.add_argument("--bg-cap-L", type=float, default=8.0)
    parser.add_argument("--bg-cap-a", type=float, default=5.0)
    parser.add_argument("--bg-cap-b", type=float, default=10.0)

    parser.add_argument("--res-cap-L", type=float, default=5.0)
    parser.add_argument("--res-cap-a", type=float, default=8.0)
    parser.add_argument("--res-cap-b", type=float, default=12.0)

    parser.add_argument("--low-conf-de", type=float, default=5.0)
    parser.add_argument("--low-conf-margin", type=float, default=1.0)

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    photo_bgr = cv2.imread(args.photo, cv2.IMREAD_COLOR)
    if photo_bgr is None:
        raise RuntimeError(f"无法读取图片: {args.photo}")

    standard_chart_bgr = cv2.imread(args.standard_chart, cv2.IMREAD_COLOR)
    if standard_chart_bgr is None:
        raise RuntimeError(f"无法读取标准色卡图片: {args.standard_chart}")

    standards = read_standards_csv(args.standards_csv)
    code_map = standard_by_code(standards)

    reference_bg_lab = None
    if args.background_lab.strip():
        reference_bg_lab = np.asarray(parse_lab_values(args.background_lab), dtype=np.float64)

    target_codes = parse_code_range(args.target_codes) if args.target_codes else prompt_target_codes()
    if not target_codes:
        raise RuntimeError("没有目标编号")

    # 选/读 ColorChecker 四角
    if args.chart_corners_json:
        corners = np.asarray(load_json(args.chart_corners_json)["corners"], dtype=np.float32)
    else:
        corners = select_chart_corners_scaled(photo_bgr, args.view_max_w, args.view_max_h)

    save_json(out_dir / "chart_corners.json", {"corners": corners.tolist()})

    chart_bgr = warp_chart(photo_bgr, corners, width=args.chart_width, height=args.chart_height)
    cv2.imwrite(str(out_dir / "01_warped_chart_before.png"), chart_bgr)

    # 提取色卡 RGB
    captured_chart_rgb = extract_chart_rgb_means(chart_bgr)
    reference_chart_rgb = extract_chart_rgb_means(cv2.resize(standard_chart_bgr, (args.chart_width, args.chart_height), interpolation=cv2.INTER_AREA))

    # root_poly2
    W_root = fit_root_poly2(captured_chart_rgb, reference_chart_rgb, ridge_alpha=args.ridge_alpha)

    root_photo_bgr = apply_root_poly2_to_image(photo_bgr, W_root)
    root_chart_bgr = apply_root_poly2_to_image(chart_bgr, W_root)
    cv2.imwrite(str(out_dir / "02_rootpoly2_corrected_photo.png"), root_photo_bgr)
    cv2.imwrite(str(out_dir / "03_rootpoly2_corrected_chart.png"), root_chart_bgr)

    # chart ΔE
    ref_chart_lab = rgb_u8_to_lab(reference_chart_rgb)
    cap_chart_lab_before = rgb_u8_to_lab(captured_chart_rgb)
    corr_chart_rgb = extract_chart_rgb_means(root_chart_bgr)
    corr_chart_lab = rgb_u8_to_lab(corr_chart_rgb)

    chart_de_before = delta_e_2000(cap_chart_lab_before, ref_chart_lab)
    chart_de_after = delta_e_2000(corr_chart_lab, ref_chart_lab)

    # 只用色卡拟合 residual
    residual_W_lab = fit_chart_residual_model(corr_chart_lab, ref_chart_lab, ridge_alpha=args.residual_ridge)

    # 选/读目标 ROI
    if args.roi_json:
        rois_data = load_json(args.roi_json)
        if isinstance(rois_data, dict) and "targets" in rois_data:
            rois_list = rois_data["targets"]
        else:
            rois_list = rois_data

        if len(rois_list) < len(target_codes):
            raise RuntimeError(f"ROI 数量不足：{len(rois_list)} < target_codes {len(target_codes)}")

        selected = []
        for i, code in enumerate(target_codes):
            item = rois_list[i]
            selected.append({"idx": i + 1, "code": code, "roi": normalize_roi(item["roi"])})
    else:
        selected = []
        for i, code in enumerate(target_codes, start=1):
            roi = select_one_roi_scaled(photo_bgr, code, i, len(target_codes), selected, args.view_max_w, args.view_max_h)
            selected.append({"idx": i, "code": code, "roi": roi})

    save_json(out_dir / "selected_rois.json", selected)

    target_rois = [normalize_roi(item["roi"]) for item in selected]
    target_mask = create_target_mask(photo_bgr.shape[:2], target_rois, pad=args.target_mask_pad)

    # 提取每个目标的 raw/root/local_bg
    target_infos = []
    local_bg_labs = []

    for item in selected:
        roi = normalize_roi(item["roi"])
        raw_lab = robust_lab_from_roi(photo_bgr, roi, inner_ratio=args.roi_inner_ratio, keep_ratio=args.roi_keep_ratio)
        root_lab = robust_lab_from_roi(root_photo_bgr, roi, inner_ratio=args.roi_inner_ratio, keep_ratio=args.roi_keep_ratio)
        bg_lab = local_background_lab(
            root_photo_bgr,
            roi,
            all_target_mask=target_mask,
            margin=args.bg_margin,
            reference_bg_lab=reference_bg_lab,
            bright_percentile=args.bg_bright_percentile,
            max_chroma=args.bg_max_chroma,
        )

        if bg_lab is not None:
            local_bg_labs.append(bg_lab)

        target_infos.append(
            {
                "idx": item["idx"],
                "code": item["code"],
                "roi": list(roi),
                "raw_lab": [float(x) for x in raw_lab],
                "root_lab": [float(x) for x in root_lab],
                "local_bg_lab": [float(x) for x in bg_lab] if bg_lab is not None else None,
            }
        )

    if reference_bg_lab is not None:
        # v2: 如果背景材料有已知标准 Lab，就用它作为局部背景校正的目标。
        # 这不是用目标胶块答案，而是用场景中已知背景材料做光照参照。
        global_bg_lab = np.asarray(reference_bg_lab, dtype=np.float64)
    elif local_bg_labs:
        global_bg_lab = np.median(np.asarray(local_bg_labs, dtype=np.float64), axis=0)
    else:
        # 兜底：没有背景时不做局部背景补偿
        global_bg_lab = np.array([50.0, 0.0, 0.0], dtype=np.float64)

    # 构建候选。不用目标标准值拟合任何参数，只是固定强度组合。
    bg_strengths = parse_float_list(args.bg_strength_list)
    residual_strengths = parse_float_list(args.residual_strength_list)

    candidates = []
    seen = set()
    for bg_s in bg_strengths:
        for res_s in residual_strengths:
            # root_only 不需要 family
            fam = (not args.disable_family_limit) and (abs(bg_s) > 1e-12 or abs(res_s) > 1e-12)
            key = (bg_s, res_s, fam)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"bg_strength": bg_s, "residual_strength": res_s, "family_limit": fam})

    summaries = []
    candidate_rows_by_name = {}

    for cand in candidates:
        summary, rows = evaluate_candidate(
            candidate=cand,
            target_infos=target_infos,
            standards=standards,
            code_map=code_map,
            global_bg_lab=global_bg_lab,
            residual_W_lab=residual_W_lab,
            args=args,
        )

        cname = summary["candidate"]
        summaries.append(summary)
        candidate_rows_by_name[cname] = rows
        write_csv(out_dir / f"target_results_{cname}.csv", rows)

    # 选择一个 diagnostic best，注意：这只是评估，不参与前面的校正计算。
    if args.select_metric == "classification_acc":
        best_summary = sorted(summaries, key=lambda r: (-float(r["classification_acc"]), float(r["final_mean_deltaE"] or 1e9)))[0]
    elif args.select_metric == "final_max_deltaE":
        best_summary = sorted(summaries, key=lambda r: float(r["final_max_deltaE"] or 1e9))[0]
    else:
        best_summary = sorted(summaries, key=lambda r: float(r["final_mean_deltaE"] or 1e9))[0]

    best_name = best_summary["candidate"]
    best_rows = candidate_rows_by_name[best_name]

    write_csv(out_dir / "candidate_summary.csv", summaries)
    write_csv(out_dir / "best_target_results.csv", best_rows)
    draw_result_overlay(photo_bgr, best_rows, out_dir / "04_best_overlay.png")

    # 分组统计，默认每 16 个一组，便于分析 128 色板
    group_rows = []
    for start in range(0, len(best_rows), 16):
        group = best_rows[start:start + 16]
        if not group:
            continue
        vals = [float(r["final_deltaE"]) for r in group if np.isfinite(float(r["final_deltaE"]))]
        roots = [float(r["root_deltaE"]) for r in group if np.isfinite(float(r["root_deltaE"]))]
        befores = [float(r["before_deltaE"]) for r in group if np.isfinite(float(r["before_deltaE"]))]
        group_rows.append(
            {
                "group": f"W{start + 1:03d}-W{start + len(group):03d}",
                "count": len(group),
                "before_mean_deltaE": stat_pack(befores)["mean"],
                "root_mean_deltaE": stat_pack(roots)["mean"],
                "final_mean_deltaE": stat_pack(vals)["mean"],
                "final_max_deltaE": stat_pack(vals)["max"],
                "classification_acc": sum(bool(r["correct"]) for r in group) / len(group),
                "harm_rate": sum(bool(r["harm"]) for r in group) / len(group),
            }
        )
    write_csv(out_dir / "group_summary_best.csv", group_rows)

    report = {
        "input": {
            "photo": args.photo,
            "standard_chart": args.standard_chart,
            "standards_csv": args.standards_csv,
            "standards_count": len(standards),
            "target_count": len(target_codes),
            "eval_count": args.eval_count if args.eval_count else len(target_codes),
        },
        "algorithm_note": (
            "校正只使用 ColorChecker 标准和图像局部背景；"
            "不使用目标胶块标准 Lab 反推 shift。best 只用于离线评估候选参数。"
        ),
        "chart": {
            "corners": corners.tolist(),
            "before_mean_deltaE": float(np.mean(chart_de_before)),
            "before_max_deltaE": float(np.max(chart_de_before)),
            "root_after_mean_deltaE": float(np.mean(chart_de_after)),
            "root_after_max_deltaE": float(np.max(chart_de_after)),
        },
        "root_poly2": {
            "ridge_alpha": args.ridge_alpha,
            "W": W_root.tolist(),
        },
        "chart_residual_model": {
            "ridge_alpha": args.residual_ridge,
            "W_lab": residual_W_lab.tolist(),
            "mean_chart_residual_after_root": [
                float(x) for x in np.mean(ref_chart_lab - corr_chart_lab, axis=0)
            ],
        },
        "local_background": {
            "reference_bg_lab": [float(x) for x in reference_bg_lab] if reference_bg_lab is not None else None,
            "global_bg_lab": [float(x) for x in global_bg_lab],
            "available_bg_count": len(local_bg_labs),
            "bg_margin": args.bg_margin,
            "bg_bright_percentile": args.bg_bright_percentile,
            "bg_max_chroma": args.bg_max_chroma,
            "bg_gains": [args.bg_gain_L, args.bg_gain_a, args.bg_gain_b],
            "bg_caps": [args.bg_cap_L, args.bg_cap_a, args.bg_cap_b],
        },
        "best_candidate_for_diagnostic": best_summary,
        "outputs": {
            "root_corrected_photo": str(out_dir / "02_rootpoly2_corrected_photo.png"),
            "candidate_summary": str(out_dir / "candidate_summary.csv"),
            "best_target_results": str(out_dir / "best_target_results.csv"),
            "group_summary_best": str(out_dir / "group_summary_best.csv"),
            "selected_rois": str(out_dir / "selected_rois.json"),
            "chart_corners": str(out_dir / "chart_corners.json"),
            "overlay": str(out_dir / "04_best_overlay.png"),
        },
    }

    save_json(out_dir / "report.json", report)

    print("\n=== ColorChecker ===")
    print(f"chart before mean/max ΔE00: {np.mean(chart_de_before):.4f} / {np.max(chart_de_before):.4f}")
    print(f"chart root   mean/max ΔE00: {np.mean(chart_de_after):.4f} / {np.max(chart_de_after):.4f}")

    print("\n=== Candidate summary ===")
    for s in sorted(summaries, key=lambda r: float(r["final_mean_deltaE"] or 1e9)):
        print(
            f'{s["candidate"]:22s} | final mean={s["final_mean_deltaE"]:.4f} '
            f'| max={s["final_max_deltaE"]:.4f} | acc={s["classification_acc"]:.4f} '
            f'| harm={s["harm_rate"]:.4f}'
        )

    print("\n=== Diagnostic best ===")
    print(json.dumps(best_summary, ensure_ascii=False, indent=2))
    print("\n输出目录：", out_dir.resolve())


if __name__ == "__main__":
    main()
