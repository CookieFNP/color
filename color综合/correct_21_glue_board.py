# -*- coding: utf-8 -*-
"""
用途：
    新板材 / 新样板上有 21 个胶块时，只做“颜色校正后视觉图”输出，不做匹配、不需要 CSV。

    输入：
        1. 一张包含 ColorChecker 色卡 + 21 个胶块的新照片
        2. standard_chart.png 标准色卡图

    输出：
        1. ColorChecker 基础校正图
        2. 使用之前最终展示参数后的视觉图
        3. 原图 / 基础校正图 / 展示校正图 三联对比
        4. 可选：框选 21 个胶块 ROI，只对胶块区域做展示校正，方便后续定点微调

    当前默认沿用你之前确认效果较好的板材展示参数：
        L_offset      = 5.0
        chroma_scale  = 0.95
        chroma_offset = 0
        a_offset      = 0.3
        b_offset      = 2.5

    注意：
        这个脚本不做 128 胶块匹配。
        不需要 glue_visual_library.csv。
        不需要 visual_mapping_T.json。
        它只负责把新图做 ColorChecker 基础校正，并输出一张接近肉眼观感的校正后图。

典型运行：
    python correct_21_glue_board.py --photo board21.jpg --standard standard_chart.png --out output_21_corrected

如果只想看 ColorChecker 基础校正图，不要浅暖化展示：
    python correct_21_glue_board.py --photo board21.jpg --standard standard_chart.png --out output_21_corrected --no-display-adjust

如果想手动框 21 个胶块，只对 21 个胶块区域应用浅暖化展示：
    python correct_21_glue_board.py --photo board21.jpg --standard standard_chart.png --out output_21_corrected --block-count 21
"""

from __future__ import annotations

import argparse
import json
import math
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
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"无法写出图像：{path}")
    buf.tofile(str(path))


# ============================================================
# sRGB / Linear RGB / Lab
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

XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    x = x / 255.0 if x.max(initial=0) > 1.0 else x
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(linear, dtype=np.float64), 0.0, 1.0)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(rgb)
    return lin @ SRGB_TO_XYZ.T


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


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    eps = 216 / 24389
    kappa = 24389 / 27

    def inv_f(f):
        return np.where(f ** 3 > eps, f ** 3, (116 * f - 16) / kappa)

    x = inv_f(fx)
    y = inv_f(fy)
    z = inv_f(fz)

    return np.stack([x, y, z], axis=-1) * D65_WHITE


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    lin = np.asarray(xyz, dtype=np.float64) @ XYZ_TO_SRGB.T
    return linear_to_srgb(lin)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return xyz_to_rgb(lab_to_xyz(lab))


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

    de = np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )

    return de


# ============================================================
# ColorChecker 选择、透视、取色
# ============================================================

def resize_for_display(img_bgr: np.ndarray, max_w: int = 1400, max_h: int = 900) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    shown = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return shown, scale


def select_four_points(image_bgr: np.ndarray, title: str = "Select ColorChecker corners") -> list[tuple[int, int]]:
    """
    依次点击：
        左上、右上、右下、左下
    """
    shown, scale = resize_for_display(image_bgr)
    points: list[tuple[int, int]] = []

    def draw():
        canvas = shown.copy()

        for i, (x, y) in enumerate(points):
            xs = int(round(x * scale))
            ys = int(round(y * scale))
            cv2.circle(canvas, (xs, ys), 6, (0, 0, 255), -1)
            cv2.putText(canvas, str(i + 1), (xs + 8, ys - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        msg = "Click ColorChecker: TL, TR, BR, BL | Enter confirm | R reset | Esc cancel"
        cv2.putText(canvas, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.imshow(title, canvas)

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((int(round(x / scale)), int(round(y / scale))))
            draw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse_cb)
    draw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in [13, 10] and len(points) == 4:
            break
        if key in [ord("r"), ord("R")]:
            points.clear()
            draw()
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("用户取消了色卡四角选择。")

    cv2.destroyWindow(title)
    return points


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
    """
    从展开后的 4x6 ColorChecker 中提取 24 色 RGB 均值。
    顺序：从左到右，从上到下。
    """
    h, w = chart_bgr.shape[:2]
    cell_w = w / cols
    cell_h = h / rows

    rgbs = []
    margin = (1.0 - inner_ratio) / 2.0

    for r in range(rows):
        for c in range(cols):
            x1 = int(round((c + margin) * cell_w))
            x2 = int(round((c + 1 - margin) * cell_w))
            y1 = int(round((r + margin) * cell_h))
            y2 = int(round((r + 1 - margin) * cell_h))

            patch = chart_bgr[y1:y2, x1:x2]
            if patch.size == 0:
                raise RuntimeError(f"ColorChecker patch 提取失败：row={r}, col={c}")

            rgb = patch[:, :, ::-1].reshape(-1, 3).mean(axis=0)
            rgbs.append(rgb)

    return np.asarray(rgbs, dtype=np.float64)


# ============================================================
# Color correction model
# ============================================================

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
            [
                R,
                G,
                B,
                R * R,
                G * G,
                B * B,
                R * G,
                R * B,
                G * B,
                np.ones_like(R),
            ],
            axis=1,
        )

    elif model_type == "root_poly2":
        eps = 1e-12
        phi = np.stack(
            [
                R,
                G,
                B,
                np.sqrt(np.maximum(R * G, eps)),
                np.sqrt(np.maximum(R * B, eps)),
                np.sqrt(np.maximum(G * B, eps)),
                np.ones_like(R),
            ],
            axis=1,
        )

    else:
        raise ValueError(f"未知模型：{model_type}")

    return phi[0] if one_dim else phi


def fit_color_correction(
    captured_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    model_type: str = "root_poly2",
    ridge_alpha: float = 1e-6,
) -> np.ndarray:
    """
    在线性 RGB 空间拟合：
        captured RGB -> reference RGB
    """
    x = srgb_to_linear(captured_rgb)
    y = srgb_to_linear(reference_rgb)

    phi = build_features(x, model_type)

    d = phi.shape[1]
    reg = np.eye(d, dtype=np.float64) * ridge_alpha
    reg[-1, -1] = 0.0

    A = phi.T @ phi + reg
    B = phi.T @ y

    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ B

    return W


def apply_color_correction_image(
    image_bgr: np.ndarray,
    W: np.ndarray,
    model_type: str,
    correction_strength: float = 1.0,
) -> np.ndarray:
    rgb = image_bgr[:, :, ::-1].astype(np.float64)
    h, w = rgb.shape[:2]

    lin = srgb_to_linear(rgb.reshape(-1, 3))
    phi = build_features(lin, model_type)
    pred = phi @ W
    pred = np.clip(pred, 0.0, 1.0)

    if correction_strength < 1.0:
        pred = lin * (1.0 - correction_strength) + pred * correction_strength

    srgb = linear_to_srgb(pred).reshape(h, w, 3)
    return srgb[:, :, ::-1].copy()


# ============================================================
# 展示层浅暖化 Lab 调整
# ============================================================

def adjust_lab_array(
    lab: np.ndarray,
    l_offset: float,
    chroma_scale: float,
    chroma_offset: float,
    a_offset: float,
    b_offset: float,
    chroma_max: float,
) -> np.ndarray:
    out = np.asarray(lab, dtype=np.float64).copy()

    out[..., 0] = np.clip(out[..., 0] + l_offset, 0.0, 100.0)

    a = out[..., 1]
    b = out[..., 2]
    C = np.sqrt(a * a + b * b)

    C_new = C * chroma_scale + chroma_offset
    C_new = np.clip(C_new, 0.0, chroma_max)

    ratio = np.ones_like(C)
    valid = C > 1e-6
    ratio[valid] = C_new[valid] / C[valid]

    out[..., 1] = a * ratio + a_offset
    out[..., 2] = b * ratio + b_offset

    C2 = np.sqrt(out[..., 1] ** 2 + out[..., 2] ** 2)
    valid2 = C2 > chroma_max
    if np.any(valid2):
        ratio2 = chroma_max / np.maximum(C2, 1e-6)
        out[..., 1] = np.where(valid2, out[..., 1] * ratio2, out[..., 1])
        out[..., 2] = np.where(valid2, out[..., 2] * ratio2, out[..., 2])

    return out


def apply_display_adjust_to_image(
    corrected_bgr: np.ndarray,
    *,
    l_offset: float = 5.0,
    chroma_scale: float = 0.95,
    chroma_offset: float = 0.0,
    a_offset: float = 0.3,
    b_offset: float = 2.5,
    chroma_max: float = 80.0,
) -> np.ndarray:
    """
    对整图做展示层 Lab 调整。
    用于快速看“校正后 + 浅暖化”的整体效果。
    """
    rgb = corrected_bgr[:, :, ::-1].astype(np.float64)
    lab = rgb_to_lab(rgb)
    lab_adj = adjust_lab_array(
        lab,
        l_offset=l_offset,
        chroma_scale=chroma_scale,
        chroma_offset=chroma_offset,
        a_offset=a_offset,
        b_offset=b_offset,
        chroma_max=chroma_max,
    )
    rgb_adj = lab_to_rgb(lab_adj)
    return rgb_adj[:, :, ::-1].copy()


def select_rois(image_bgr: np.ndarray, count: int, out_path: Path | None = None) -> list[tuple[int, int, int, int]]:
    """
    连续框选 count 个 ROI。
    返回 x1,y1,x2,y2。
    """
    shown, scale = resize_for_display(image_bgr)
    rois = []

    win = "Select block ROI | Enter confirm each | Esc cancel"

    for i in range(count):
        canvas = shown.copy()
        msg = f"Select block {i + 1}/{count}, Enter confirm"
        cv2.putText(canvas, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        for j, (x1, y1, x2, y2) in enumerate(rois, start=1):
            cv2.rectangle(
                canvas,
                (int(x1 * scale), int(y1 * scale)),
                (int(x2 * scale), int(y2 * scale)),
                (0, 0, 255),
                2,
            )
            cv2.putText(canvas, str(j), (int(x1 * scale), int(y1 * scale) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, canvas)
        cv2.waitKey(1)

        roi = cv2.selectROI(win, canvas, showCrosshair=True, fromCenter=False)
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            cv2.destroyWindow(win)
            raise RuntimeError(f"第 {i + 1} 个 ROI 未选择。")

        x1 = int(round(x / scale))
        y1 = int(round(y / scale))
        x2 = int(round((x + w) / scale))
        y2 = int(round((y + h) / scale))

        rois.append((x1, y1, x2, y2))

    cv2.destroyWindow(win)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "count": count,
            "rois_xyxy": [list(map(int, r)) for r in rois],
            "note": "ROI order follows manual selection order.",
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return rois


def load_rois(path: Path) -> list[tuple[int, int, int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "rois_xyxy" in data:
        return [tuple(map(int, r)) for r in data["rois_xyxy"]]

    if isinstance(data, list):
        return [tuple(map(int, r)) for r in data]

    raise RuntimeError(f"无法解析 ROI 文件：{path}")


def feather_rect_mask(h: int, w: int, feather: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=np.float32)
    if feather > 0:
        k = max(3, int(feather) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)


def apply_display_adjust_to_rois(
    corrected_bgr: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    *,
    l_offset: float,
    chroma_scale: float,
    chroma_offset: float,
    a_offset: float,
    b_offset: float,
    chroma_max: float,
    feather: int = 9,
    crop_dir: Path | None = None,
) -> np.ndarray:
    """
    只对指定 ROI 做展示层 Lab 调整，背景保持 corrected。
    """
    out = corrected_bgr.copy()
    H, W = out.shape[:2]

    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)

    for idx, roi in enumerate(rois, start=1):
        x1, y1, x2, y2 = roi
        x1 = max(0, min(W - 1, int(x1)))
        x2 = max(1, min(W, int(x2)))
        y1 = max(0, min(H - 1, int(y1)))
        y2 = max(1, min(H, int(y2)))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = corrected_bgr[y1:y2, x1:x2].copy()
        crop_adj = apply_display_adjust_to_image(
            crop,
            l_offset=l_offset,
            chroma_scale=chroma_scale,
            chroma_offset=chroma_offset,
            a_offset=a_offset,
            b_offset=b_offset,
            chroma_max=chroma_max,
        )

        mask = feather_rect_mask(crop.shape[0], crop.shape[1], feather)[:, :, None]
        blended = crop.astype(np.float32) * (1 - mask) + crop_adj.astype(np.float32) * mask
        out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        if crop_dir is not None:
            imwrite_unicode(crop_dir / f"block_{idx:02d}_corrected.png", crop)
            imwrite_unicode(crop_dir / f"block_{idx:02d}_display.png", crop_adj)

    return out


# ============================================================
# 可视化
# ============================================================

def add_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_triptych(original_bgr: np.ndarray, corrected_bgr: np.ndarray, display_bgr: np.ndarray, out_path: Path) -> None:
    max_h = 900
    imgs = [original_bgr, corrected_bgr, display_bgr]
    labels = ["original", "colorchecker corrected", "display adjusted"]

    resized = []
    for img, label in zip(imgs, labels):
        h, w = img.shape[:2]
        scale = min(max_h / h, 1.0)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        resized.append(add_label(small, label))

    min_h = min(x.shape[0] for x in resized)
    resized = [cv2.resize(x, (int(x.shape[1] * min_h / x.shape[0]), min_h), interpolation=cv2.INTER_AREA) for x in resized]

    combined = np.concatenate(resized, axis=1)
    imwrite_unicode(out_path, combined)


def draw_rois(image_bgr: np.ndarray, rois: list[tuple[int, int, int, int]], out_path: Path) -> None:
    canvas = image_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(rois, start=1):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(canvas, str(i), (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
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
    parser = argparse.ArgumentParser(description="Correct a new board with 21 glue blocks using ColorChecker only.")

    parser.add_argument("--photo", required=True, help="新板材照片，包含 ColorChecker 和 21 个胶块")
    parser.add_argument("--standard", default="standard_chart.png", help="标准 ColorChecker 图")
    parser.add_argument("--out", default="output_21_corrected", help="输出目录")

    parser.add_argument(
        "--model-type",
        choices=["linear_bias", "poly2", "root_poly2"],
        default="root_poly2",
        help="ColorChecker 基础校正模型，默认 root_poly2。",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--correction-strength", type=float, default=1.0)

    parser.add_argument("--chart-corners-file", default=None, help="可选，复用色卡四角 JSON")
    parser.add_argument("--force-select-chart", action="store_true", help="强制重新点击色卡四角")

    parser.add_argument("--no-display-adjust", action="store_true", help="不输出浅暖化展示图，只看 ColorChecker corrected")
    parser.add_argument("--display-l-offset", type=float, default=5.0)
    parser.add_argument("--display-chroma-scale", type=float, default=0.95)
    parser.add_argument("--display-chroma-offset", type=float, default=0.0)
    parser.add_argument("--display-a-offset", type=float, default=0.3)
    parser.add_argument("--display-b-offset", type=float, default=2.5)
    parser.add_argument("--display-chroma-max", type=float, default=80.0)

    parser.add_argument(
        "--block-count",
        type=int,
        default=0,
        help="可选。若设为 21，则手动框选 21 个胶块，只对这些 ROI 做展示调整。",
    )
    parser.add_argument("--block-rois-file", default=None, help="可选，复用 21 个胶块 ROI JSON")
    parser.add_argument("--force-select-blocks", action="store_true", help="强制重新框选胶块 ROI")
    parser.add_argument("--roi-feather", type=int, default=9, help="ROI 边缘羽化，默认 9")

    args = parser.parse_args()

    photo_path = Path(args.photo)
    standard_path = Path(args.standard)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_bgr = imread_unicode(photo_path)
    standard_bgr = imread_unicode(standard_path)

    # 1. 色卡四角
    chart_corners_file = Path(args.chart_corners_file) if args.chart_corners_file else out_dir / "chart_corners.json"

    if chart_corners_file.exists() and not args.force_select_chart:
        corners = json.loads(chart_corners_file.read_text(encoding="utf-8"))
        corners = [tuple(map(int, p)) for p in corners]
        print("已加载色卡四角：", chart_corners_file)
    else:
        print("\n请依次点击 ColorChecker 四角：左上、右上、右下、左下。")
        corners = select_four_points(original_bgr)
        chart_corners_file.write_text(json.dumps(corners, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已保存色卡四角：", chart_corners_file)

    # 2. 展开色卡并取 24 色
    chart_warp = warp_chart(original_bgr, corners, output_size=(600, 400))
    standard_chart = cv2.resize(standard_bgr, (600, 400), interpolation=cv2.INTER_AREA)

    imwrite_unicode(out_dir / "01_chart_warp.png", chart_warp)
    imwrite_unicode(out_dir / "01_standard_chart_resized.png", standard_chart)

    captured_rgb = extract_colorchecker_24_rgb(chart_warp)
    reference_rgb = extract_colorchecker_24_rgb(standard_chart)

    # 3. 拟合并应用 ColorChecker 校正
    W = fit_color_correction(
        captured_rgb=captured_rgb,
        reference_rgb=reference_rgb,
        model_type=args.model_type,
        ridge_alpha=args.ridge_alpha,
    )

    corrected_bgr = apply_color_correction_image(
        original_bgr,
        W=W,
        model_type=args.model_type,
        correction_strength=args.correction_strength,
    )

    imwrite_unicode(out_dir / "02_colorchecker_corrected.png", corrected_bgr)

    # 4. 计算色卡校正前后 ΔE
    corrected_chart = warp_chart(corrected_bgr, corners, output_size=(600, 400))
    corrected_rgb = extract_colorchecker_24_rgb(corrected_chart)

    ref_lab = rgb_to_lab(reference_rgb)
    cap_lab = rgb_to_lab(captured_rgb)
    fix_lab = rgb_to_lab(corrected_rgb)

    de_before = delta_e_2000(cap_lab, ref_lab)
    de_after = delta_e_2000(fix_lab, ref_lab)

    # 5. 输出浅暖化展示图
    display_global_bgr = None

    if not args.no_display_adjust:
        display_global_bgr = apply_display_adjust_to_image(
            corrected_bgr,
            l_offset=args.display_l_offset,
            chroma_scale=args.display_chroma_scale,
            chroma_offset=args.display_chroma_offset,
            a_offset=args.display_a_offset,
            b_offset=args.display_b_offset,
            chroma_max=args.display_chroma_max,
        )
        imwrite_unicode(out_dir / "03_display_adjusted_global.png", display_global_bgr)
        make_triptych(original_bgr, corrected_bgr, display_global_bgr, out_dir / "04_compare_original_corrected_display.png")

    else:
        make_triptych(original_bgr, corrected_bgr, corrected_bgr, out_dir / "04_compare_original_corrected_display.png")

    # 6. 可选：只对 21 个胶块 ROI 应用展示调整
    rois = []
    blocks_only_path = None

    if args.block_count > 0 and not args.no_display_adjust:
        block_rois_file = Path(args.block_rois_file) if args.block_rois_file else out_dir / "block_rois.json"

        if block_rois_file.exists() and not args.force_select_blocks:
            rois = load_rois(block_rois_file)
            print("已加载胶块 ROI：", block_rois_file)
        else:
            print(f"\n请依次框选 {args.block_count} 个胶块 ROI。")
            rois = select_rois(corrected_bgr, args.block_count, out_path=block_rois_file)
            print("已保存胶块 ROI：", block_rois_file)

        draw_rois(corrected_bgr, rois, out_dir / "05_block_rois_on_corrected.png")

        blocks_only_bgr = apply_display_adjust_to_rois(
            corrected_bgr,
            rois,
            l_offset=args.display_l_offset,
            chroma_scale=args.display_chroma_scale,
            chroma_offset=args.display_chroma_offset,
            a_offset=args.display_a_offset,
            b_offset=args.display_b_offset,
            chroma_max=args.display_chroma_max,
            feather=args.roi_feather,
            crop_dir=out_dir / "block_crops",
        )

        blocks_only_path = out_dir / "06_display_adjusted_blocks_only.png"
        imwrite_unicode(blocks_only_path, blocks_only_bgr)

    # 7. report
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
        "display_adjust": None
        if args.no_display_adjust
        else {
            "l_offset": args.display_l_offset,
            "chroma_scale": args.display_chroma_scale,
            "chroma_offset": args.display_chroma_offset,
            "a_offset": args.display_a_offset,
            "b_offset": args.display_b_offset,
            "chroma_max": args.display_chroma_max,
        },
        "block_rois": [list(map(int, r)) for r in rois],
        "outputs": {
            "chart_warp": str(out_dir / "01_chart_warp.png"),
            "colorchecker_corrected": str(out_dir / "02_colorchecker_corrected.png"),
            "display_adjusted_global": None if display_global_bgr is None else str(out_dir / "03_display_adjusted_global.png"),
            "compare": str(out_dir / "04_compare_original_corrected_display.png"),
            "display_adjusted_blocks_only": None if blocks_only_path is None else str(blocks_only_path),
        },
    }

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== 新板材胶块校正完成 ====")
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
    print("  ColorChecker 基础校正图：", out_dir / "02_colorchecker_corrected.png")
    if not args.no_display_adjust:
        print("  全图浅暖化展示图：", out_dir / "03_display_adjusted_global.png")
    print("  三联对比图：", out_dir / "04_compare_original_corrected_display.png")
    if blocks_only_path:
        print("  只调 21 胶块区域图：", blocks_only_path)
    print("  report：", out_dir / "report.json")


if __name__ == "__main__":
    main()
