# -*- coding: utf-8 -*-
"""
用途：
    新板材上有 21 个胶块时，按照“之前 128 胶块视觉库学到的模板”做视觉校正，
    只输出校正后视觉图，不做匹配、不需要输入 CSV。

    这个版本修正了两个重点：
        1. 胶块 ROI 使用“手动画圆”，不是方框。
        2. 圆内不是纯 ColorChecker 基础校正，而是：
              ColorChecker corrected_lab
              ↓
              使用 128 胶块视觉库训练出的 T 模板
              ↓
              visual_lab
              ↓
              将 residual 叠加到圆形胶块区域

核心流程：
    新图 + ColorChecker
    ↓
    ColorChecker 基础校正
    ↓
    读取 visual_mapping_T_poly2/visual_mapping_T.json
       或者从 glue_visual_library.csv 现场训练 T
    ↓
    手动画 21 个圆形胶块 ROI
    ↓
    每个圆：
        corrected_lab = 圆内校正后代表色
        visual_lab = T(corrected_lab)
        residual = visual_lab - corrected_lab
        圆内像素 Lab += residual
    ↓
    背景大量保留原图
    ↓
    输出：
        final = mostly_original_background + T视觉校正后的圆形胶块

注意：
    不做 128 胶块匹配。
    不需要新图有胶块编号。
    不做板材展示暖化参数。
    这里用的是 128 胶块视觉库学到的 T 模板。

典型运行：
    python correct_21_glue_board_v3.py --photo board21.jpg --standard standard_chart.png --out output_21_v3

如果你的 T 模型路径不是默认位置：
    python correct_21_glue_board_v3.py --photo board21.jpg --standard standard_chart.png --out output_21_v3 --mapping output_128/glue_visual_library/visual_mapping_T_poly2/visual_mapping_T.json

如果想强制从 glue_visual_library.csv 重新训练 T：
    python correct_21_glue_board_v3.py --photo board21.jpg --standard standard_chart.png --out output_21_v3 --fit-from-library --library output_128/glue_visual_library/glue_visual_library.csv

如果背景还太雾、太淡：
    python correct_21_glue_board_v3.py --photo board21.jpg --standard standard_chart.png --out output_21_v3 --background-corrected-weight 0.15

如果同一张图想重画 21 个圆：
    python correct_21_glue_board_v3.py --photo board21.jpg --standard standard_chart.png --out output_21_v3 --force-select-circles
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

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


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# sRGB / Linear RGB / Lab / DeltaE
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

    return np.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


def stat_pack(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


# ============================================================
# OpenCV 显示和交互
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

    def redraw():
        canvas = shown.copy()
        msg = "Click ColorChecker: TL, TR, BR, BL | Enter confirm | R reset | Esc cancel"
        cv2.putText(canvas, msg, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)

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
            raise RuntimeError("用户取消了色卡四角选择。")

    cv2.destroyWindow(title)
    return points


def select_circles(image_bgr: np.ndarray, count: int, title: str = "Draw circular glue ROIs") -> list[dict]:
    """
    真正的圆形框选，不调用 cv2.selectROI，所以不会是方框。

    操作：
        左键按下：圆心
        拖动：半径
        左键松开：保存一个圆
        U：撤销上一个圆
        R：清空重画
        Enter：当数量达到 count 后确认
        Esc：取消
    """
    shown, scale = resize_for_display(image_bgr)
    circles: list[dict] = []

    drawing = False
    center_disp: tuple[int, int] | None = None
    radius_disp = 0

    def redraw():
        canvas = shown.copy()

        msg = f"Draw circle {len(circles)+1}/{count} | drag mouse | U undo | R reset | Enter confirm"
        cv2.putText(canvas, msg, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

        for i, c in enumerate(circles, start=1):
            x = int(round(c["cx"] * scale))
            y = int(round(c["cy"] * scale))
            r = int(round(c["r"] * scale))
            cv2.circle(canvas, (x, y), r, (0, 0, 255), 2)
            cv2.putText(canvas, str(i), (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        if drawing and center_disp is not None and radius_disp > 1:
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
            radius_disp = int(round(math.hypot(x - center_disp[0], y - center_disp[1])))
            redraw()

        elif event == cv2.EVENT_LBUTTONUP and drawing and center_disp is not None:
            drawing = False
            radius_disp = int(round(math.hypot(x - center_disp[0], y - center_disp[1])))

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

        if key in [ord("u"), ord("U")]:
            if circles:
                circles.pop()
            redraw()

        if key in [ord("r"), ord("R")]:
            circles.clear()
            drawing = False
            center_disp = None
            radius_disp = 0
            redraw()

        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("用户取消圆形 ROI 选择。")

    cv2.destroyWindow(title)
    return circles


# ============================================================
# ColorChecker 校正
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
            if patch.size == 0:
                raise RuntimeError(f"ColorChecker patch 提取失败：row={r + 1}, col={c + 1}")

            rgb = patch[:, :, ::-1].reshape(-1, 3).mean(axis=0)
            rgbs.append(rgb)

    return np.asarray(rgbs, dtype=np.float64)


def build_color_features(linear_rgb: np.ndarray, model_type: str) -> np.ndarray:
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
        raise ValueError(f"未知 model_type: {model_type}")

    return phi[0] if one_dim else phi


def fit_color_correction(
    captured_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    model_type: str = "root_poly2",
    ridge_alpha: float = 1e-6,
) -> np.ndarray:
    x = srgb_to_linear(captured_rgb)
    y = srgb_to_linear(reference_rgb)

    phi = build_color_features(x, model_type)

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
    phi = build_color_features(lin, model_type)
    pred = phi @ W
    pred = np.clip(pred, 0.0, 1.0)

    if correction_strength < 1.0:
        pred = lin * (1 - correction_strength) + pred * correction_strength

    srgb = linear_to_srgb(pred).reshape(h, w, 3)
    return srgb[:, :, ::-1].copy()


# ============================================================
# T 模板：corrected_lab -> visual_lab
# ============================================================

def build_T_features(x_lab: np.ndarray, feature_mode: str) -> np.ndarray:
    x = np.asarray(x_lab, dtype=np.float64)
    one_dim = x.ndim == 1
    if one_dim:
        x = x.reshape(1, 3)

    L = x[:, 0]
    a = x[:, 1]
    b = x[:, 2]

    if feature_mode == "linear":
        phi = np.stack([np.ones_like(L), L, a, b], axis=1)

    elif feature_mode == "poly2":
        phi = np.stack(
            [
                np.ones_like(L),
                L,
                a,
                b,
                L * L,
                a * a,
                b * b,
                L * a,
                L * b,
                a * b,
            ],
            axis=1,
        )

    else:
        raise ValueError(f"未知 T feature_mode：{feature_mode}")

    return phi[0] if one_dim else phi


def apply_visual_mapping_T(corrected_lab: np.ndarray, mapping: dict) -> np.ndarray:
    x = np.asarray(corrected_lab, dtype=np.float64).reshape(1, 3)

    feature_mode = mapping["feature_mode"]
    l_mode = mapping["L_mode"]

    phi = build_T_features(x, feature_mode)

    out = np.zeros((1, 3), dtype=np.float64)

    if l_mode == "identity":
        out[:, 0] = x[:, 0]
    elif l_mode == "linear":
        out[:, 0] = phi @ np.asarray(mapping["coefficients"]["L"], dtype=np.float64)
    else:
        raise ValueError(f"未知 T L_mode：{l_mode}")

    out[:, 1] = phi @ np.asarray(mapping["coefficients"]["a"], dtype=np.float64)
    out[:, 2] = phi @ np.asarray(mapping["coefficients"]["b"], dtype=np.float64)

    return out[0]


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except Exception:
        return default


def train_visual_mapping_from_library(
    library_csv: Path,
    feature_mode: str = "poly2",
    l_mode: str = "identity",
    ridge_alpha: float = 1e-6,
) -> dict:
    """
    从 glue_visual_library.csv 现场训练：
        corrected_lab -> visual_display_lab
    """
    if not library_csv.exists():
        raise FileNotFoundError(f"找不到 glue_visual_library.csv：{library_csv}")

    with library_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("glue_visual_library.csv 为空。")

    required = [
        "corrected_L",
        "corrected_a",
        "corrected_b",
        "visual_display_L",
        "visual_display_a",
        "visual_display_b",
    ]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise RuntimeError("glue_visual_library.csv 缺少字段：" + ", ".join(missing))

    X = []
    Y = []

    for row in rows:
        x = [
            to_float(row.get("corrected_L")),
            to_float(row.get("corrected_a")),
            to_float(row.get("corrected_b")),
        ]
        y = [
            to_float(row.get("visual_display_L")),
            to_float(row.get("visual_display_a")),
            to_float(row.get("visual_display_b")),
        ]

        if any(v is None for v in x + y):
            continue

        X.append(x)
        Y.append(y)

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] < 10:
        raise RuntimeError("可用于训练 T 的样本太少。")

    Phi = build_T_features(X, feature_mode)

    d = Phi.shape[1]
    reg = np.eye(d, dtype=np.float64) * ridge_alpha
    reg[0, 0] = 0.0

    coeffs = {}

    if l_mode == "identity":
        coeffs["L"] = None
    elif l_mode == "linear":
        coeffs["L"] = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y[:, 0]).tolist()
    else:
        raise ValueError(f"未知 l_mode：{l_mode}")

    coeffs["a"] = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y[:, 1]).tolist()
    coeffs["b"] = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y[:, 2]).tolist()

    pred = np.zeros_like(Y)
    if l_mode == "identity":
        pred[:, 0] = X[:, 0]
    else:
        pred[:, 0] = Phi @ np.asarray(coeffs["L"], dtype=np.float64)
    pred[:, 1] = Phi @ np.asarray(coeffs["a"], dtype=np.float64)
    pred[:, 2] = Phi @ np.asarray(coeffs["b"], dtype=np.float64)

    fit_de = delta_e_2000(pred, Y)

    return {
        "version": "trained_on_the_fly_from_glue_visual_library",
        "feature_mode": feature_mode,
        "L_mode": l_mode,
        "ridge_alpha": ridge_alpha,
        "coefficients": coeffs,
        "training": {
            "library_csv": str(library_csv),
            "sample_count": int(X.shape[0]),
            "fit_deltaE2000": stat_pack(fit_de),
        },
    }


def load_or_train_mapping(args: argparse.Namespace, out_dir: Path) -> dict:
    mapping_path = Path(args.mapping) if args.mapping else None
    library_path = Path(args.library) if args.library else None

    if args.fit_from_library:
        if library_path is None:
            raise RuntimeError("--fit-from-library 需要提供 --library")
        mapping = train_visual_mapping_from_library(
            library_path,
            feature_mode=args.T_feature_mode,
            l_mode=args.T_l_mode,
            ridge_alpha=args.T_ridge_alpha,
        )
        write_json(out_dir / "visual_mapping_T_trained_from_library.json", mapping)
        print("已从 glue_visual_library.csv 现场训练 T：", library_path)
        return mapping

    if mapping_path and mapping_path.exists():
        print("已加载 visual_mapping_T：", mapping_path)
        return json.loads(mapping_path.read_text(encoding="utf-8"))

    if library_path and library_path.exists():
        mapping = train_visual_mapping_from_library(
            library_path,
            feature_mode=args.T_feature_mode,
            l_mode=args.T_l_mode,
            ridge_alpha=args.T_ridge_alpha,
        )
        write_json(out_dir / "visual_mapping_T_trained_from_library.json", mapping)
        print("未找到 mapping，已从 glue_visual_library.csv 现场训练 T：", library_path)
        return mapping

    raise FileNotFoundError(
        "找不到 visual_mapping_T.json，也找不到 glue_visual_library.csv。\n"
        "请检查 --mapping 或 --library 路径。"
    )


# ============================================================
# 圆形 ROI 视觉修正
# ============================================================

def make_one_circle_mask(shape_hw: tuple[int, int], circle: dict, feather: int = 15) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)

    cx = int(round(circle["cx"]))
    cy = int(round(circle["cy"]))
    r = int(round(circle["r"]))

    cv2.circle(mask, (cx, cy), max(1, r), 1.0, thickness=-1)

    if feather > 0:
        k = max(3, int(feather) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return np.clip(mask, 0.0, 1.0)


def make_all_circle_mask(shape_hw: tuple[int, int], circles: list[dict], feather: int = 15) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.float32)

    for c in circles:
        cx = int(round(c["cx"]))
        cy = int(round(c["cy"]))
        r = int(round(c["r"]))
        cv2.circle(mask, (cx, cy), max(1, r), 1.0, thickness=-1)

    if feather > 0:
        k = max(3, int(feather) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return np.clip(mask, 0.0, 1.0)


def representative_lab_from_circle(
    image_bgr: np.ndarray,
    circle: dict,
    sample_radius_scale: float = 0.75,
    trim_percent: float = 10.0,
) -> np.ndarray:
    """
    从圆心附近取代表色，避免边缘和背景影响。
    """
    h, w = image_bgr.shape[:2]

    cx = int(round(circle["cx"]))
    cy = int(round(circle["cy"]))
    r = int(round(circle["r"] * sample_radius_scale))

    x1 = max(0, cx - r)
    x2 = min(w, cx + r + 1)
    y1 = max(0, cy - r)
    y2 = min(h, cy + r + 1)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError("圆形 ROI 为空。")

    yy, xx = np.mgrid[y1:y2, x1:x2]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)

    pixels_rgb = crop[:, :, ::-1][mask].reshape(-1, 3).astype(np.float64)

    if pixels_rgb.shape[0] < 10:
        pixels_rgb = crop[:, :, ::-1].reshape(-1, 3).astype(np.float64)

    if trim_percent > 0 and pixels_rgb.shape[0] >= 30:
        lo = np.percentile(pixels_rgb, trim_percent, axis=0)
        hi = np.percentile(pixels_rgb, 100.0 - trim_percent, axis=0)
        keep = np.all((pixels_rgb >= lo) & (pixels_rgb <= hi), axis=1)
        if keep.sum() >= max(20, pixels_rgb.shape[0] * 0.2):
            pixels_rgb = pixels_rgb[keep]

    mean_rgb = pixels_rgb.mean(axis=0)
    return rgb_to_lab(mean_rgb.reshape(1, 3))[0]


def apply_T_residual_to_circles(
    corrected_bgr: np.ndarray,
    circles: list[dict],
    mapping: dict,
    *,
    circle_strength: float = 1.0,
    circle_feather: int = 15,
    sample_radius_scale: float = 0.75,
    trim_percent: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    对 corrected 图中的圆形胶块区域应用 T residual。

    每个圆：
        corrected_lab_mean -> T -> visual_lab_target
        residual = visual_lab_target - corrected_lab_mean
        圆内像素 Lab += residual * circle_strength
    """
    corrected_rgb = corrected_bgr[:, :, ::-1].astype(np.float64)
    corrected_lab_img = rgb_to_lab(corrected_rgb)

    out_lab = corrected_lab_img.copy()

    H, W = corrected_bgr.shape[:2]
    total_mask = np.zeros((H, W), dtype=np.float32)
    rows = []

    for idx, circle in enumerate(circles, start=1):
        mean_corrected_lab = representative_lab_from_circle(
            corrected_bgr,
            circle,
            sample_radius_scale=sample_radius_scale,
            trim_percent=trim_percent,
        )

        target_visual_lab = apply_visual_mapping_T(mean_corrected_lab, mapping)
        residual = (target_visual_lab - mean_corrected_lab) * float(circle_strength)

        mask = make_one_circle_mask((H, W), circle, feather=circle_feather)
        total_mask = np.maximum(total_mask, mask)

        out_lab[..., 0] = out_lab[..., 0] + mask * residual[0]
        out_lab[..., 1] = out_lab[..., 1] + mask * residual[1]
        out_lab[..., 2] = out_lab[..., 2] + mask * residual[2]

        rows.append(
            {
                "index": idx,
                "circle": {
                    "cx": float(circle["cx"]),
                    "cy": float(circle["cy"]),
                    "r": float(circle["r"]),
                },
                "corrected_lab": [float(x) for x in mean_corrected_lab],
                "T_visual_lab": [float(x) for x in target_visual_lab],
                "residual": [float(x) for x in residual],
            }
        )

    out_lab[..., 0] = np.clip(out_lab[..., 0], 0.0, 100.0)

    out_rgb = lab_to_rgb(out_lab)
    out_bgr = out_rgb[:, :, ::-1].copy()

    return out_bgr, total_mask, rows


def blend_background(original_bgr: np.ndarray, corrected_bgr: np.ndarray, corrected_weight: float) -> np.ndarray:
    alpha = float(np.clip(corrected_weight, 0.0, 1.0))
    bg = corrected_bgr.astype(np.float32) * alpha + original_bgr.astype(np.float32) * (1 - alpha)
    return np.clip(bg, 0, 255).astype(np.uint8)


def compose_final(
    *,
    original_bgr: np.ndarray,
    corrected_bgr: np.ndarray,
    T_circles_bgr: np.ndarray,
    circles: list[dict],
    background_corrected_weight: float,
    circle_feather: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    final:
        背景 = mostly original + little corrected
        圆内 = T_circles_bgr
    """
    bg = blend_background(original_bgr, corrected_bgr, corrected_weight=background_corrected_weight)
    mask = make_all_circle_mask(original_bgr.shape[:2], circles, feather=circle_feather)

    mask3 = mask[:, :, None]
    final = bg.astype(np.float32) * (1 - mask3) + T_circles_bgr.astype(np.float32) * mask3
    final = np.clip(final, 0, 255).astype(np.uint8)

    return final, bg, (mask * 255).astype(np.uint8)


# ============================================================
# 可视化
# ============================================================

def draw_circle_overlay(image_bgr: np.ndarray, circles: list[dict], out_path: Path) -> None:
    canvas = image_bgr.copy()
    for i, c in enumerate(circles, start=1):
        cx = int(round(c["cx"]))
        cy = int(round(c["cy"]))
        r = int(round(c["r"]))
        cv2.circle(canvas, (cx, cy), r, (0, 0, 255), 2)
        cv2.putText(canvas, str(i), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    imwrite_unicode(out_path, canvas)


def add_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(out, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
    return out


def make_triptych(original_bgr: np.ndarray, corrected_bgr: np.ndarray, final_bgr: np.ndarray, out_path: Path) -> None:
    imgs = [
        add_label(original_bgr, "original"),
        add_label(corrected_bgr, "ColorChecker corrected"),
        add_label(final_bgr, "final: original bg + T visual circles"),
    ]

    max_h = 900
    resized = []
    for img in imgs:
        h, w = img.shape[:2]
        scale = min(max_h / h, 1.0)
        resized.append(cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA))

    h_min = min(img.shape[0] for img in resized)
    resized = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=cv2.INTER_AREA) for img in resized]

    canvas = np.concatenate(resized, axis=1)
    imwrite_unicode(out_path, canvas)


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Apply 128-glue visual mapping T to 21 circular glue ROIs.")

    parser.add_argument("--photo", required=True, help="包含 ColorChecker + 21 个胶块的新图")
    parser.add_argument("--standard", default="standard_chart.png", help="标准 ColorChecker 图")
    parser.add_argument("--out", default="output_21_v3", help="输出目录")

    parser.add_argument("--model-type", choices=["linear_bias", "poly2", "root_poly2"], default="root_poly2")
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--correction-strength", type=float, default=1.0)

    parser.add_argument("--chart-corners-file", default=None, help="可选，复用色卡四角 JSON")
    parser.add_argument("--force-select-chart", action="store_true", help="强制重新点色卡四角")

    parser.add_argument("--circle-count", type=int, default=21)
    parser.add_argument("--circles-file", default=None, help="可选，复用圆形 ROI JSON")
    parser.add_argument("--force-select-circles", action="store_true", help="强制重新画圆形 ROI")
    parser.add_argument("--circle-feather", type=int, default=15)
    parser.add_argument("--circle-strength", type=float, default=1.0, help="T residual 应用强度，默认 1.0")
    parser.add_argument("--sample-radius-scale", type=float, default=0.75, help="取圆内代表色时使用半径比例，默认 0.75")
    parser.add_argument("--trim-percent", type=float, default=10.0)

    parser.add_argument(
        "--background-corrected-weight",
        type=float,
        default=0.25,
        help="背景中 corrected 的权重。越小越接近原图，默认 0.25。",
    )

    parser.add_argument(
        "--mapping",
        default="output_128/glue_visual_library/visual_mapping_T_poly2/visual_mapping_T.json",
        help="之前训练好的 visual_mapping_T.json",
    )
    parser.add_argument(
        "--library",
        default="output_128/glue_visual_library/glue_visual_library.csv",
        help="如果 mapping 不存在，可从该视觉库现场训练 T。",
    )
    parser.add_argument("--fit-from-library", action="store_true", help="强制从 glue_visual_library.csv 现场训练 T")
    parser.add_argument("--T-feature-mode", choices=["linear", "poly2"], default="poly2")
    parser.add_argument("--T-l-mode", choices=["identity", "linear"], default="identity")
    parser.add_argument("--T-ridge-alpha", type=float, default=1e-6)

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    photo_path = Path(args.photo)
    standard_path = Path(args.standard)

    original_bgr = imread_unicode(photo_path)
    standard_bgr = imread_unicode(standard_path)

    # 1. 加载/训练 T
    mapping = load_or_train_mapping(args, out_dir)

    # 2. 色卡四角
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

    # 3. ColorChecker 基础校正
    chart_warp = warp_chart(original_bgr, corners, output_size=(600, 400))
    standard_chart = cv2.resize(standard_bgr, (600, 400), interpolation=cv2.INTER_AREA)

    imwrite_unicode(out_dir / "01_chart_warp.png", chart_warp)
    imwrite_unicode(out_dir / "01_standard_chart_resized.png", standard_chart)

    captured_rgb = extract_colorchecker_24_rgb(chart_warp)
    reference_rgb = extract_colorchecker_24_rgb(standard_chart)

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

    corrected_chart = warp_chart(corrected_bgr, corners, output_size=(600, 400))
    corrected_rgb = extract_colorchecker_24_rgb(corrected_chart)

    ref_lab = rgb_to_lab(reference_rgb)
    cap_lab = rgb_to_lab(captured_rgb)
    fix_lab = rgb_to_lab(corrected_rgb)

    de_before = delta_e_2000(cap_lab, ref_lab)
    de_after = delta_e_2000(fix_lab, ref_lab)

    # 4. 画圆形 ROI
    circles_file = Path(args.circles_file) if args.circles_file else out_dir / "glue_circles.json"

    if circles_file.exists() and not args.force_select_circles:
        circles = json.loads(circles_file.read_text(encoding="utf-8"))
        print("已加载圆形 ROI：", circles_file)
    else:
        print(f"\n请依次拖拽画 {args.circle_count} 个圆形胶块 ROI。注意：这里是真圆，不是方框。")
        circles = select_circles(corrected_bgr, count=args.circle_count)
        circles_file.write_text(json.dumps(circles, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已保存圆形 ROI：", circles_file)

    draw_circle_overlay(original_bgr, circles, out_dir / "03_circles_on_original.png")
    draw_circle_overlay(corrected_bgr, circles, out_dir / "03_circles_on_corrected.png")

    # 5. 使用 T residual 校正圆内胶块
    T_circles_bgr, circle_mask, circle_rows = apply_T_residual_to_circles(
        corrected_bgr=corrected_bgr,
        circles=circles,
        mapping=mapping,
        circle_strength=args.circle_strength,
        circle_feather=args.circle_feather,
        sample_radius_scale=args.sample_radius_scale,
        trim_percent=args.trim_percent,
    )

    imwrite_unicode(out_dir / "04_T_visual_applied_full_corrected_basis.png", T_circles_bgr)

    # 6. 背景大权重原图，圆内用 T 视觉校正后的颜色
    final_bgr, background_bgr, final_mask = compose_final(
        original_bgr=original_bgr,
        corrected_bgr=corrected_bgr,
        T_circles_bgr=T_circles_bgr,
        circles=circles,
        background_corrected_weight=args.background_corrected_weight,
        circle_feather=args.circle_feather,
    )

    imwrite_unicode(out_dir / "05_background_mostly_original.png", background_bgr)
    imwrite_unicode(out_dir / "06_circle_mask_TRUE_CIRCLES.png", final_mask)
    imwrite_unicode(out_dir / "07_final_T_visual_circles_on_original_bg.png", final_bgr)
    make_triptych(original_bgr, corrected_bgr, final_bgr, out_dir / "08_triptych.png")

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
        "visual_mapping_T": {
            "source_mapping": args.mapping,
            "source_library": args.library,
            "fit_from_library": bool(args.fit_from_library),
            "feature_mode": mapping.get("feature_mode"),
            "L_mode": mapping.get("L_mode"),
            "ridge_alpha": mapping.get("ridge_alpha"),
            "training": mapping.get("training"),
        },
        "circle_roi": {
            "count": len(circles),
            "circles_file": str(circles_file),
            "circle_feather": args.circle_feather,
            "circle_strength": args.circle_strength,
            "sample_radius_scale": args.sample_radius_scale,
            "trim_percent": args.trim_percent,
            "rows": circle_rows,
        },
        "background": {
            "background_corrected_weight": args.background_corrected_weight,
            "background_original_weight": 1.0 - args.background_corrected_weight,
        },
        "outputs": {
            "colorchecker_corrected": str(out_dir / "02_colorchecker_corrected.png"),
            "circles_on_original": str(out_dir / "03_circles_on_original.png"),
            "T_visual_applied_full_corrected_basis": str(out_dir / "04_T_visual_applied_full_corrected_basis.png"),
            "background_mostly_original": str(out_dir / "05_background_mostly_original.png"),
            "circle_mask_true_circles": str(out_dir / "06_circle_mask_TRUE_CIRCLES.png"),
            "final": str(out_dir / "07_final_T_visual_circles_on_original_bg.png"),
            "triptych": str(out_dir / "08_triptych.png"),
        },
    }

    write_json(out_dir / "report.json", report)

    print("\n==== 21 胶块 T 视觉校正完成 ====")
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
    print("  基础校正图：", out_dir / "02_colorchecker_corrected.png")
    print("  真圆 mask：", out_dir / "06_circle_mask_TRUE_CIRCLES.png")
    print("  最终图：", out_dir / "07_final_T_visual_circles_on_original_bg.png")
    print("  三联图：", out_dir / "08_triptych.png")
    print("  report：", out_dir / "report.json")


if __name__ == "__main__":
    main()
