# -*- coding: utf-8 -*-
"""
用途：
    新板材上有 21 个胶块，现在你已经有其中 20 个胶块的 CSV 目标 Lab 值时，
    用这些已知值对对应圆形胶块做“定点 residual 校正”，只输出视觉校正图，不做匹配。

    适用场景：
        - 一张新图：包含 ColorChecker + 21 个胶块
        - 21 个胶块需要你手动画圆形 ROI
        - 你有第 2~21 个胶块的目标颜色值 CSV
        - 第 1 个胶块没有目标值，可以选择：
            1) 不处理，只保留 ColorChecker corrected
            2) 用之前 128 胶块视觉库训练出的 T 模板估计
            3) 后续再单独补值微调

核心流程：
    新图 + ColorChecker
    ↓
    ColorChecker 基础校正
    ↓
    手动画 21 个圆形 ROI
    ↓
    读取 CSV 中 20 个目标 Lab
    ↓
    默认把 CSV 第 1 行对应圆 2，CSV 第 2 行对应圆 3，...，CSV 第 20 行对应圆 21
    ↓
    对每个有目标值的圆：
        corrected_lab = 圆内校正后代表色
        target_lab = CSV 给出的目标 Lab
        residual = target_lab - corrected_lab
        圆内像素 Lab += residual * strength
    ↓
    背景大量保留原图
    ↓
    输出最终图

CSV 支持格式 1：
    index,L,a,b
    2,55.1,3.2,18.6
    3,60.2,4.1,20.3

CSV 支持格式 2：
    code,name,L,a,b
    W031,淡雅黄,77.97,-2.16,28.66

CSV 支持格式 3：
    编号,名称,LAB
    W031,淡雅黄,"77.97, -2.16, 28.66"

如果 CSV 只有 20 行：
    默认认为它们对应圆 2~21，也就是跳过第一个无数据胶块。

如果 CSV 有 21 行：
    默认认为它们对应圆 1~21。

典型运行：
    python correct_21_glue_board_with_csv.py --photo board21.jpg --standard standard_chart.png --target-csv data_20.csv --out output_21_csv

如果背景还雾：
    python correct_21_glue_board_with_csv.py --photo board21.jpg --standard standard_chart.png --target-csv data_20.csv --out output_21_csv --background-corrected-weight 0.15

如果定点修正太猛：
    python correct_21_glue_board_with_csv.py --photo board21.jpg --standard standard_chart.png --target-csv data_20.csv --out output_21_csv --target-strength 0.8

如果第一个无数据胶块想用 T 模板估计：
    python correct_21_glue_board_with_csv.py --photo board21.jpg --standard standard_chart.png --target-csv data_20.csv --out output_21_csv --first-mode T --mapping output_128/glue_visual_library/visual_mapping_T_poly2/visual_mapping_T.json
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


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = []
    seen = set()
    preferred = [
        "circle_index",
        "source",
        "code",
        "name",
        "target_L",
        "target_a",
        "target_b",
        "corrected_L",
        "corrected_a",
        "corrected_b",
        "residual_L",
        "residual_a",
        "residual_b",
        "applied_strength",
    ]

    for k in preferred:
        if any(k in r for r in rows) and k not in seen:
            keys.append(k)
            seen.add(k)

    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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
# 交互
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
            raise RuntimeError("用户取消色卡四角选择。")

    cv2.destroyWindow(title)
    return points


def select_circles(image_bgr: np.ndarray, count: int, title: str = "Draw circular glue ROIs") -> list[dict]:
    """
    真圆框选，不使用 cv2.selectROI。

    操作：
        左键按下：圆心
        拖动：半径
        左键松开：保存圆
        U：撤销上一个
        R：全部重画
        Enter：画满 count 个后确认
        Esc：取消
    """
    shown, scale = resize_for_display(image_bgr)
    circles: list[dict] = []

    drawing = False
    center_disp: tuple[int, int] | None = None
    radius_disp = 0

    def redraw():
        canvas = shown.copy()

        msg = f"Draw circle {len(circles)+1}/{count} | drag | U undo | R reset | Enter confirm"
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
# T 模板，可选用于第一个无数据胶块
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


# ============================================================
# CSV 目标 Lab 读取
# ============================================================

def normalize_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "")


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


def parse_lab_text(text: Any) -> list[float] | None:
    if text is None:
        return None
    raw = str(text).strip().strip('"').strip("'")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", raw)
    if len(nums) < 3:
        return None
    return [float(nums[0]), float(nums[1]), float(nums[2])]


def read_target_lab_csv(csv_path: Path) -> list[dict]:
    """
    读取目标 Lab CSV。
    兼容：
        L,a,b
        target_L,target_a,target_b
        visual_display_L,visual_display_a,visual_display_b
        LAB = "L,a,b"
        中文：编号,名称,LAB
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 target CSV：{csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("target CSV 为空。")

    field_map = {normalize_col(c): c for c in rows[0].keys()}

    def find_col(cands: list[str]) -> str | None:
        for cand in cands:
            key = normalize_col(cand)
            if key in field_map:
                return field_map[key]
        return None

    code_col = find_col(["code", "编号", "色号"])
    name_col = find_col(["name", "名称", "颜色名称"])
    index_col = find_col(["index", "idx", "circle_index", "block_index", "序号"])

    lab_col = find_col(["LAB", "lab", "Lab"])

    L_col = find_col(["L", "target_L", "targetL", "visual_display_L", "visualL", "standard_L", "standardL"])
    a_col = find_col(["a", "A", "target_a", "targetA", "visual_display_a", "visuala", "standard_a", "standarda"])
    b_col = find_col(["b", "B", "target_b", "targetB", "visual_display_b", "visualb", "standard_b", "standardb"])

    targets = []

    for row_idx, row in enumerate(rows, start=1):
        lab = None

        if lab_col:
            lab = parse_lab_text(row.get(lab_col))

        if lab is None and L_col and a_col and b_col:
            L = to_float(row.get(L_col))
            a = to_float(row.get(a_col))
            b = to_float(row.get(b_col))
            if L is not None and a is not None and b is not None:
                lab = [L, a, b]

        if lab is None:
            raise RuntimeError(f"CSV 第 {row_idx} 行无法解析 Lab：{row}")

        index_value = to_float(row.get(index_col), None) if index_col else None

        targets.append(
            {
                "row_index": row_idx,
                "index": None if index_value is None else int(index_value),
                "code": "" if not code_col else str(row.get(code_col, "")).strip(),
                "name": "" if not name_col else str(row.get(name_col, "")).strip(),
                "lab": [float(lab[0]), float(lab[1]), float(lab[2])],
                "raw": row,
            }
        )

    return targets


def assign_targets_to_circles(targets: list[dict], circle_count: int, skip_first_if_20: bool = True) -> dict[int, dict]:
    """
    返回：
        circle_index -> target
    circle_index 从 1 开始。

    规则：
        如果 CSV 行里有 index/circle_index/block_index，就按 index。
        否则：
            - 20 行且 circle_count=21，则默认对应 2~21
            - 21 行则对应 1~21
            - 其他数量则按 1 开始顺序对应
    """
    explicit = [t for t in targets if t.get("index") is not None]

    assigned: dict[int, dict] = {}

    if len(explicit) == len(targets):
        for t in targets:
            idx = int(t["index"])
            if idx < 1 or idx > circle_count:
                raise RuntimeError(f"CSV index 超出 1~{circle_count}：{idx}")
            assigned[idx] = t
        return assigned

    if skip_first_if_20 and len(targets) == circle_count - 1:
        start = 2
    else:
        start = 1

    for offset, t in enumerate(targets):
        idx = start + offset
        if idx > circle_count:
            break
        assigned[idx] = t

    return assigned


# ============================================================
# 圆形 ROI residual 校正
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


def apply_target_residual_to_circles(
    *,
    corrected_bgr: np.ndarray,
    circles: list[dict],
    assigned_targets: dict[int, dict],
    target_strength: float,
    first_mode: str,
    mapping: dict | None,
    first_strength: float,
    circle_feather: int,
    sample_radius_scale: float,
    trim_percent: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    有 CSV 目标 Lab 的圆：
        residual = target_lab - corrected_lab

    第一个无数据圆：
        none: 不改
        T: corrected_lab -> T_visual_lab，再 residual
    """
    corrected_rgb = corrected_bgr[:, :, ::-1].astype(np.float64)
    corrected_lab_img = rgb_to_lab(corrected_rgb)

    out_lab = corrected_lab_img.copy()
    H, W = corrected_bgr.shape[:2]

    total_mask = np.zeros((H, W), dtype=np.float32)
    rows = []

    for circle_index, circle in enumerate(circles, start=1):
        mean_corrected_lab = representative_lab_from_circle(
            corrected_bgr,
            circle,
            sample_radius_scale=sample_radius_scale,
            trim_percent=trim_percent,
        )

        target = assigned_targets.get(circle_index)

        if target is not None:
            target_lab = np.asarray(target["lab"], dtype=np.float64)
            residual = (target_lab - mean_corrected_lab) * float(target_strength)
            source = "csv_target"
            strength = float(target_strength)
            code = target.get("code", "")
            name = target.get("name", "")

        else:
            if first_mode == "T":
                if mapping is None:
                    raise RuntimeError("first-mode=T 需要提供 --mapping")
                target_lab = apply_visual_mapping_T(mean_corrected_lab, mapping)
                residual = (target_lab - mean_corrected_lab) * float(first_strength)
                source = "T_fallback"
                strength = float(first_strength)
                code = ""
                name = ""
            elif first_mode == "none":
                target_lab = mean_corrected_lab.copy()
                residual = np.zeros(3, dtype=np.float64)
                source = "unchanged_no_target"
                strength = 0.0
                code = ""
                name = ""
            else:
                raise ValueError(f"未知 first_mode：{first_mode}")

        mask = make_one_circle_mask((H, W), circle, feather=circle_feather)
        total_mask = np.maximum(total_mask, mask)

        out_lab[..., 0] += mask * residual[0]
        out_lab[..., 1] += mask * residual[1]
        out_lab[..., 2] += mask * residual[2]

        rows.append(
            {
                "circle_index": circle_index,
                "source": source,
                "code": code,
                "name": name,
                "target_L": float(target_lab[0]),
                "target_a": float(target_lab[1]),
                "target_b": float(target_lab[2]),
                "corrected_L": float(mean_corrected_lab[0]),
                "corrected_a": float(mean_corrected_lab[1]),
                "corrected_b": float(mean_corrected_lab[2]),
                "residual_L": float(residual[0]),
                "residual_a": float(residual[1]),
                "residual_b": float(residual[2]),
                "applied_strength": strength,
                "circle_cx": float(circle["cx"]),
                "circle_cy": float(circle["cy"]),
                "circle_r": float(circle["r"]),
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
    target_circles_bgr: np.ndarray,
    circles: list[dict],
    background_corrected_weight: float,
    circle_feather: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bg = blend_background(original_bgr, corrected_bgr, corrected_weight=background_corrected_weight)
    mask = make_all_circle_mask(original_bgr.shape[:2], circles, feather=circle_feather)

    mask3 = mask[:, :, None]
    final = bg.astype(np.float32) * (1 - mask3) + target_circles_bgr.astype(np.float32) * mask3
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
        add_label(final_bgr, "final: original bg + CSV target circles"),
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


def draw_target_info_image(image_bgr: np.ndarray, rows: list[dict], out_path: Path) -> None:
    canvas = image_bgr.copy()
    for row in rows:
        cx = int(round(row["circle_cx"]))
        cy = int(round(row["circle_cy"]))
        r = int(round(row["circle_r"]))
        idx = int(row["circle_index"])
        source = row["source"]

        color = (0, 0, 255) if source == "csv_target" else (255, 0, 0)
        cv2.circle(canvas, (cx, cy), r, color, 2)

        label = f"{idx}"
        if row.get("code"):
            label += f" {row['code']}"
        elif source == "T_fallback":
            label += " T"
        elif source == "unchanged_no_target":
            label += " none"

        cv2.putText(canvas, label, (cx - r, max(30, cy - r - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    imwrite_unicode(out_path, canvas)


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Correct 21 circular glue blocks with 20 target Lab values from CSV.")

    parser.add_argument("--photo", required=True, help="包含 ColorChecker + 21 胶块的新图")
    parser.add_argument("--standard", default="standard_chart.png", help="标准 ColorChecker 图")
    parser.add_argument("--target-csv", required=True, help="20 个或 21 个目标 Lab 的 CSV")
    parser.add_argument("--out", default="output_21_csv", help="输出目录")

    parser.add_argument("--model-type", choices=["linear_bias", "poly2", "root_poly2"], default="root_poly2")
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--correction-strength", type=float, default=1.0)

    parser.add_argument("--chart-corners-file", default=None)
    parser.add_argument("--force-select-chart", action="store_true")

    parser.add_argument("--circle-count", type=int, default=21)
    parser.add_argument("--circles-file", default=None)
    parser.add_argument("--force-select-circles", action="store_true")
    parser.add_argument("--circle-feather", type=int, default=15)
    parser.add_argument("--sample-radius-scale", type=float, default=0.75)
    parser.add_argument("--trim-percent", type=float, default=10.0)

    parser.add_argument("--target-strength", type=float, default=1.0, help="CSV target residual 强度，默认 1.0")
    parser.add_argument(
        "--first-mode",
        choices=["none", "T"],
        default="none",
        help="第一个无 CSV 数据胶块的处理方式：none=不处理；T=用 visual_mapping_T 估计。默认 none。",
    )
    parser.add_argument("--first-strength", type=float, default=1.0)
    parser.add_argument(
        "--mapping",
        default="output_128/glue_visual_library/visual_mapping_T_poly2/visual_mapping_T.json",
        help="first-mode=T 时使用的 visual_mapping_T.json",
    )

    parser.add_argument(
        "--background-corrected-weight",
        type=float,
        default=0.25,
        help="背景中 corrected 的权重。越小越接近原图，默认 0.25。",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    photo_path = Path(args.photo)
    standard_path = Path(args.standard)
    target_csv_path = Path(args.target_csv)

    original_bgr = imread_unicode(photo_path)
    standard_bgr = imread_unicode(standard_path)

    # 1. 读取目标 CSV，并分配到圆
    targets = read_target_lab_csv(target_csv_path)
    assigned_targets = assign_targets_to_circles(
        targets,
        circle_count=args.circle_count,
        skip_first_if_20=True,
    )

    print(f"已读取 target CSV：{target_csv_path}")
    print(f"CSV 行数：{len(targets)}")
    print("目标值对应圆序号：", sorted(assigned_targets.keys()))

    # 2. 可选加载 T，给第一个无数据胶块用
    mapping = None
    if args.first_mode == "T":
        mapping_path = Path(args.mapping)
        if not mapping_path.exists():
            raise FileNotFoundError(f"first-mode=T 但找不到 mapping：{mapping_path}")
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        print("已加载第一个胶块 T fallback：", mapping_path)

    # 3. 色卡四角
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

    # 4. ColorChecker 基础校正
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

    # 5. 真圆 ROI
    circles_file = Path(args.circles_file) if args.circles_file else out_dir / "glue_circles.json"

    if circles_file.exists() and not args.force_select_circles:
        circles = json.loads(circles_file.read_text(encoding="utf-8"))
        print("已加载圆形 ROI：", circles_file)
    else:
        print(f"\n请依次拖拽画 {args.circle_count} 个圆形胶块 ROI。")
        circles = select_circles(corrected_bgr, count=args.circle_count)
        circles_file.write_text(json.dumps(circles, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已保存圆形 ROI：", circles_file)

    draw_circle_overlay(original_bgr, circles, out_dir / "03_circles_on_original.png")
    draw_circle_overlay(corrected_bgr, circles, out_dir / "03_circles_on_corrected.png")

    # 6. CSV target residual 应用到圆内
    target_circles_bgr, total_mask, circle_rows = apply_target_residual_to_circles(
        corrected_bgr=corrected_bgr,
        circles=circles,
        assigned_targets=assigned_targets,
        target_strength=args.target_strength,
        first_mode=args.first_mode,
        mapping=mapping,
        first_strength=args.first_strength,
        circle_feather=args.circle_feather,
        sample_radius_scale=args.sample_radius_scale,
        trim_percent=args.trim_percent,
    )

    imwrite_unicode(out_dir / "04_csv_target_applied_full_corrected_basis.png", target_circles_bgr)

    # 7. 背景大量原图，圆内用 CSV target residual 后结果
    final_bgr, background_bgr, final_mask = compose_final(
        original_bgr=original_bgr,
        corrected_bgr=corrected_bgr,
        target_circles_bgr=target_circles_bgr,
        circles=circles,
        background_corrected_weight=args.background_corrected_weight,
        circle_feather=args.circle_feather,
    )

    imwrite_unicode(out_dir / "05_background_mostly_original.png", background_bgr)
    imwrite_unicode(out_dir / "06_circle_mask_TRUE_CIRCLES.png", final_mask)
    imwrite_unicode(out_dir / "07_final_csv_target_circles_on_original_bg.png", final_bgr)
    make_triptych(original_bgr, corrected_bgr, final_bgr, out_dir / "08_triptych.png")
    draw_target_info_image(final_bgr, circle_rows, out_dir / "09_final_with_circle_labels.png")

    write_csv(out_dir / "circle_residual_report.csv", circle_rows)

    report = {
        "input": {
            "photo": str(photo_path),
            "standard": str(standard_path),
            "target_csv": str(target_csv_path),
        },
        "target_csv": {
            "row_count": len(targets),
            "assigned_circle_indices": sorted(assigned_targets.keys()),
            "rule": "If CSV has 20 rows and circle_count=21, rows are assigned to circles 2~21 by order unless an index column is provided.",
        },
        "colorchecker": {
            "model_type": args.model_type,
            "ridge_alpha": args.ridge_alpha,
            "correction_strength": args.correction_strength,
            "deltaE_before": stat_pack(de_before),
            "deltaE_after": stat_pack(de_after),
        },
        "circle_roi": {
            "count": len(circles),
            "circles_file": str(circles_file),
            "circle_feather": args.circle_feather,
            "sample_radius_scale": args.sample_radius_scale,
            "trim_percent": args.trim_percent,
        },
        "residual": {
            "target_strength": args.target_strength,
            "first_mode": args.first_mode,
            "first_strength": args.first_strength,
            "rows": circle_rows,
        },
        "background": {
            "background_corrected_weight": args.background_corrected_weight,
            "background_original_weight": 1.0 - args.background_corrected_weight,
        },
        "outputs": {
            "colorchecker_corrected": str(out_dir / "02_colorchecker_corrected.png"),
            "circles_on_original": str(out_dir / "03_circles_on_original.png"),
            "csv_target_applied_full_corrected_basis": str(out_dir / "04_csv_target_applied_full_corrected_basis.png"),
            "background_mostly_original": str(out_dir / "05_background_mostly_original.png"),
            "circle_mask_true_circles": str(out_dir / "06_circle_mask_TRUE_CIRCLES.png"),
            "final": str(out_dir / "07_final_csv_target_circles_on_original_bg.png"),
            "triptych": str(out_dir / "08_triptych.png"),
            "final_with_labels": str(out_dir / "09_final_with_circle_labels.png"),
            "circle_residual_report_csv": str(out_dir / "circle_residual_report.csv"),
        },
    }

    write_json(out_dir / "report.json", report)

    print("\n==== 21 胶块 CSV 定点校正完成 ====")
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
    print("  最终图：", out_dir / "07_final_csv_target_circles_on_original_bg.png")
    print("  带编号最终图：", out_dir / "09_final_with_circle_labels.png")
    print("  每个圆 residual 报告：", out_dir / "circle_residual_report.csv")
    print("  report：", out_dir / "report.json")


if __name__ == "__main__":
    main()
