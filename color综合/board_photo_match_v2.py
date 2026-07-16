# -*- coding: utf-8 -*-
"""
用途：
    板材端完整匹配流程脚本。

    输入一张“板材 + ColorChecker 色卡”的照片，脚本会：
        1. 手动点击 ColorChecker 四角
        2. 基于标准色卡图 standard_chart.png 做 ColorChecker 基础校正
        3. 手动框选板材 ROI
        4. 从校正图中提取 board_corrected_lab
        5. 使用 train_visual_mapping_T.py 训练得到的 visual_mapping_T.json：
              board_visual_lab = T(board_corrected_lab)
        6. 读取 glue_visual_library.csv
        7. 计算：
              ΔE(board_visual_lab, glue_visual_library.visual_display_lab)
        8. 输出 TopK 胶块匹配结果、JSON、CSV、推荐胶块拼图

    注意：
        板材是未知样本，不能使用 standard_lab - corrected_lab。
        板材端只允许用从胶块 v0.7 视觉库训练出来的 T。
        最终匹配发生在 visual Lab 域：
              board_visual_lab vs glue_visual_display_lab

典型运行：
    python board_photo_match_v2.py --photo board.jpg --standard standard_chart.png --library output_128/glue_visual_library/glue_visual_library.csv --mapping output_128/glue_visual_library/visual_mapping_T/visual_mapping_T.json --out board_match_output --top-k 10 --board-display-l-offset 1.5 --board-display-chroma-scale 1.04

说明：
    board_visual_lab 用于匹配，不做展示微调。
    board_display_lab 只用于输出图片/界面展示，可轻微提亮、提色度，不参与 TopK 排序。

如果你之前把 poly2 单独保存到 visual_mapping_T_poly2：
    python board_photo_match.py --photo board.jpg --standard standard_chart.png --library output_128/glue_visual_library/glue_visual_library.csv --mapping output_128/glue_visual_library/visual_mapping_T_poly2/visual_mapping_T.json --out board_match_output --top-k 10
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
# 基础 IO
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


def safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r'[\\/:*?"<>|\s]+', "_", text)
    return text.strip("_") or "unknown"


# ============================================================
# 色彩空间：sRGB / Linear RGB / Lab / CIEDE2000
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
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    x = np.asarray(linear, dtype=np.float64)
    x = np.clip(x, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1 / 2.4)) - 0.055)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


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
    """
    CIEDE2000，输入 shape (..., 3)，输出 shape (...,)。
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    kL = kC = kH = 1.0

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
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )

    return de


# ============================================================
# ColorChecker 提取与颜色校正
# ============================================================

def select_four_points(image_bgr: np.ndarray, title: str) -> list[tuple[int, int]]:
    """
    鼠标依次点击四点：
        左上、右上、右下、左下
    """
    points: list[tuple[int, int]] = []

    display = image_bgr.copy()
    max_w = 1400
    max_h = 900

    h, w = display.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    shown = cv2.resize(display, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def redraw():
        canvas = shown.copy()
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

        instruction = "Click ColorChecker corners: TL, TR, BR, BL | R reset | Enter confirm"
        cv2.putText(
            canvas,
            instruction,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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
    cv2.setMouseCallback(title, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in [13, 10]:  # Enter
            if len(points) == 4:
                break

        elif key in [ord("r"), ord("R")]:
            points = []
            redraw()

        elif key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("用户取消了 ColorChecker 四角选择。")

    cv2.destroyWindow(title)
    return points


def warp_chart(image_bgr: np.ndarray, corners: list[tuple[int, int]], output_size: tuple[int, int] = (600, 400)) -> np.ndarray:
    """
    透视变换色卡。
    output_size = (width, height)，默认 6:4。
    """
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
    从矫正后的 4x6 ColorChecker 图中提取 24 个中心区域 RGB 均值。
    顺序：从左到右、从上到下。
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
                raise RuntimeError("ColorChecker patch 提取失败。")

            rgb = patch[:, :, ::-1].reshape(-1, 3).mean(axis=0)
            rgbs.append(rgb)

    return np.asarray(rgbs, dtype=np.float64)


def build_features(linear_rgb: np.ndarray, model_type: str) -> np.ndarray:
    """
    颜色校正模型特征。

    linear_bias:
        [R, G, B, 1]

    poly2:
        [R, G, B, R², G², B², RG, RB, GB, 1]

    root_poly2:
        [R, G, B, sqrt(RG), sqrt(RB), sqrt(GB), 1]
    """
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
        raise ValueError(f"未知颜色校正模型：{model_type}")

    return phi[0] if one_dim else phi


def fit_color_correction(
    captured_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    model_type: str = "root_poly2",
    ridge_alpha: float = 1e-6,
) -> np.ndarray:
    """
    拟合 captured RGB -> reference RGB 的映射。
    在 Linear RGB 中拟合。
    """
    x = srgb_to_linear(captured_rgb)
    y = srgb_to_linear(reference_rgb)

    phi = build_features(x, model_type)

    d = phi.shape[1]
    reg = np.eye(d, dtype=np.float64) * ridge_alpha
    reg[-1, -1] = 0.0  # bias 不正则化

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
    """
    对整图应用颜色校正。
    correction_strength:
        1.0 = 完整校正
        0.5 = 原图 linear RGB 与校正结果中间混合
    """
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


def colorchecker_delta_report(captured_rgb: np.ndarray, corrected_rgb: np.ndarray, reference_rgb: np.ndarray) -> dict:
    before_lab = rgb_to_lab(captured_rgb)
    after_lab = rgb_to_lab(corrected_rgb)
    ref_lab = rgb_to_lab(reference_rgb)

    before_de = delta_e_2000(before_lab, ref_lab)
    after_de = delta_e_2000(after_lab, ref_lab)

    def pack(x):
        x = np.asarray(x, dtype=np.float64)
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "max": float(np.max(x)),
            "p95": float(np.percentile(x, 95)),
        }

    return {
        "before": pack(before_de),
        "after": pack(after_de),
        "before_deltaE": before_de.tolist(),
        "after_deltaE": after_de.tolist(),
    }


# ============================================================
# ROI 取色
# ============================================================

def select_board_roi(image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """
    用 OpenCV selectROI 手动框选板材区域。
    返回 x1,y1,x2,y2。
    """
    h, w = image_bgr.shape[:2]

    max_w = 1400
    max_h = 900
    scale = min(max_w / w, max_h / h, 1.0)

    shown = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cv2.namedWindow("Select board ROI | Enter confirm | Esc cancel", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select board ROI | Enter confirm | Esc cancel", shown, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select board ROI | Enter confirm | Esc cancel")

    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        raise RuntimeError("未选择有效板材 ROI。")

    x1 = int(round(x / scale))
    y1 = int(round(y / scale))
    x2 = int(round((x + rw) / scale))
    y2 = int(round((y + rh) / scale))

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    return x1, y1, x2, y2


def load_roi_file(path: Path) -> tuple[int, int, int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "roi_xyxy" in data:
        return tuple(map(int, data["roi_xyxy"]))

    if isinstance(data, list) and len(data) == 4:
        return tuple(map(int, data))

    raise RuntimeError(f"无法解析 ROI 文件：{path}")


def save_roi_file(path: Path, roi: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "roi_xyxy": list(map(int, roi)),
        "note": "board ROI selected on original/corrected image coordinates",
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def representative_rgb_from_roi(
    image_bgr: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    trim_percent: float = 10.0,
) -> np.ndarray:
    x1, y1, x2, y2 = roi_xyxy
    crop = image_bgr[y1:y2, x1:x2]

    if crop.size == 0:
        raise RuntimeError("板材 ROI 为空。")

    pixels_rgb = crop[:, :, ::-1].reshape(-1, 3).astype(np.float64)

    if trim_percent > 0 and pixels_rgb.shape[0] >= 30:
        lo = np.percentile(pixels_rgb, trim_percent, axis=0)
        hi = np.percentile(pixels_rgb, 100.0 - trim_percent, axis=0)
        keep = np.all((pixels_rgb >= lo) & (pixels_rgb <= hi), axis=1)

        if keep.sum() >= max(20, pixels_rgb.shape[0] * 0.2):
            pixels_rgb = pixels_rgb[keep]

    return pixels_rgb.mean(axis=0)


# ============================================================
# 视觉映射 T
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
        phi = np.stack(
            [
                np.ones_like(L),
                L,
                a,
                b,
            ],
            axis=1,
        )

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


def adjust_board_display_lab(
    board_visual_lab: np.ndarray,
    l_offset: float = 0.0,
    chroma_scale: float = 1.0,
) -> np.ndarray:
    """
    用途：
        只对“板材展示色块/界面预览”做轻微视觉微调。

    注意：
        这个函数不参与 TopK 匹配。
        TopK 匹配仍然使用 board_visual_lab。

    参数：
        l_offset:
            L 通道显示提亮量。
            建议 1.0~2.0，默认 0。

        chroma_scale:
            a/b 色度显示倍率。
            建议 1.03~1.06，默认 1。
    """
    lab = np.asarray(board_visual_lab, dtype=np.float64).copy()

    lab[0] = np.clip(lab[0] + float(l_offset), 0.0, 100.0)
    lab[1] = lab[1] * float(chroma_scale)
    lab[2] = lab[2] * float(chroma_scale)

    return lab


# ============================================================
# 胶块视觉库匹配
# ============================================================

def read_glue_visual_library(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 glue_visual_library.csv：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = ["code", "name", "visual_display_L", "visual_display_a", "visual_display_b"]

    if not rows:
        raise RuntimeError("glue_visual_library.csv 为空。")

    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise RuntimeError("glue_visual_library.csv 缺少字段：" + ", ".join(missing))

    return rows


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


def match_topk(board_visual_lab: np.ndarray, library_rows: list[dict], top_k: int) -> list[dict]:
    board = np.asarray(board_visual_lab, dtype=np.float64).reshape(1, 3)

    results = []

    for row in library_rows:
        lab = np.asarray(
            [
                to_float(row.get("visual_display_L")),
                to_float(row.get("visual_display_a")),
                to_float(row.get("visual_display_b")),
            ],
            dtype=np.float64,
        )

        if np.any(~np.isfinite(lab)):
            continue

        de = float(delta_e_2000(board, lab.reshape(1, 3))[0])

        results.append(
            {
                "code": row.get("code"),
                "name": row.get("name"),
                "deltaE_match": de,
                "visual_display_L": float(lab[0]),
                "visual_display_a": float(lab[1]),
                "visual_display_b": float(lab[2]),
                "visual_crop_path": row.get("visual_crop_path"),
                "machine_L": row.get("machine_L"),
                "machine_a": row.get("machine_a"),
                "machine_b": row.get("machine_b"),
                "standard_L": row.get("standard_L"),
                "standard_a": row.get("standard_a"),
                "standard_b": row.get("standard_b"),
            }
        )

    results.sort(key=lambda x: x["deltaE_match"])

    for i, row in enumerate(results[:top_k], start=1):
        row["rank"] = i

    return results[:top_k]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "rank",
        "code",
        "name",
        "deltaE_match",
        "visual_display_L",
        "visual_display_a",
        "visual_display_b",
        "visual_crop_path",
    ]

    keys = []
    seen = set()

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
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# 可视化输出
# ============================================================

def draw_roi_preview(image_bgr: np.ndarray, roi: tuple[int, int, int, int], out_path: Path, text: str) -> None:
    x1, y1, x2, y2 = roi
    canvas = image_bgr.copy()

    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 4)
    cv2.putText(
        canvas,
        text,
        (x1, max(40, y1 - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
    )

    imwrite_unicode(out_path, canvas)


def color_swatch_bgr_from_lab(lab: np.ndarray, size: tuple[int, int] = (120, 120)) -> np.ndarray:
    rgb = lab_to_rgb(np.asarray(lab, dtype=np.float64).reshape(1, 3))[0]
    bgr = rgb[::-1]
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def make_contact_sheet(
    *,
    top_results: list[dict],
    board_visual_lab: np.ndarray,
    board_display_lab: np.ndarray,
    out_path: Path,
    base_dir: Path,
    tile_w: int = 220,
    tile_h: int = 270,
) -> None:
    """
    生成 TopK 胶块拼图。

    注意：
        TopK 排序使用 board_visual_lab。
        左侧展示色块使用 board_display_lab，这样可以做轻微提亮/提色度，
        但不影响匹配结果。
    """
    n = len(top_results) + 1
    cols = min(5, n)
    rows = int(math.ceil(n / cols))

    sheet = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # board swatch
    board_tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
    swatch = color_swatch_bgr_from_lab(board_display_lab, size=(150, 150))
    board_tile[45:195, 35:185] = swatch
    cv2.putText(board_tile, "BOARD display", (15, 28), font, 0.62, (0, 0, 0), 2)
    cv2.putText(
        board_tile,
        f"L={board_display_lab[0]:.1f}",
        (15, 220),
        font,
        0.55,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        board_tile,
        f"a={board_display_lab[1]:.1f} b={board_display_lab[2]:.1f}",
        (15, 245),
        font,
        0.55,
        (0, 0, 0),
        1,
    )
    sheet[0:tile_h, 0:tile_w] = board_tile

    for idx, row in enumerate(top_results, start=1):
        r = idx // cols
        c = idx % cols
        y0 = r * tile_h
        x0 = c * tile_w

        tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)

        crop_path_text = row.get("visual_crop_path") or ""
        crop_path = Path(crop_path_text)

        if not crop_path.exists():
            crop_path2 = base_dir / crop_path
            crop_path = crop_path2 if crop_path2.exists() else crop_path

        if crop_path.exists():
            crop = imread_unicode(crop_path, cv2.IMREAD_UNCHANGED)

            if crop.ndim == 3 and crop.shape[2] == 4:
                alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
                rgb = crop[:, :, :3]
                bg = np.full_like(rgb, 255)
                crop_bgr = (rgb.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)
            else:
                crop_bgr = crop[:, :, :3]

            ch, cw = crop_bgr.shape[:2]
            scale = min(150 / cw, 150 / ch, 1.0)
            crop_resized = cv2.resize(
                crop_bgr,
                (max(1, int(cw * scale)), max(1, int(ch * scale))),
                interpolation=cv2.INTER_AREA,
            )

            yy = 45 + (150 - crop_resized.shape[0]) // 2
            xx = 35 + (150 - crop_resized.shape[1]) // 2
            tile[yy:yy + crop_resized.shape[0], xx:xx + crop_resized.shape[1]] = crop_resized
        else:
            lab = np.asarray(
                [
                    float(row["visual_display_L"]),
                    float(row["visual_display_a"]),
                    float(row["visual_display_b"]),
                ],
                dtype=np.float64,
            )
            swatch = color_swatch_bgr_from_lab(lab, size=(150, 150))
            tile[45:195, 35:185] = swatch

        title = f"#{row['rank']} {row.get('code','')}"
        name = str(row.get("name") or "")
        de = float(row.get("deltaE_match", 0))

        cv2.putText(tile, title[:18], (12, 26), font, 0.65, (0, 0, 0), 2)
        cv2.putText(tile, name[:12], (12, 218), font, 0.55, (0, 0, 0), 1)
        cv2.putText(tile, f"dE={de:.2f}", (12, 245), font, 0.6, (0, 0, 0), 2)

        sheet[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    imwrite_unicode(out_path, sheet)


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Board photo -> ColorChecker corrected_lab -> visual_mapping_T -> glue visual library TopK."
    )

    parser.add_argument("--photo", required=True, help="板材 + ColorChecker 原图")
    parser.add_argument("--standard", required=True, help="标准 ColorChecker 图片，例如 standard_chart.png")
    parser.add_argument("--library", required=True, help="glue_visual_library.csv")
    parser.add_argument("--mapping", required=True, help="visual_mapping_T.json")
    parser.add_argument("--out", default="board_match_output", help="输出目录")
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument(
        "--model-type",
        choices=["linear_bias", "poly2", "root_poly2"],
        default="root_poly2",
        help="ColorChecker 基础校正模型，默认 root_poly2。",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--correction-strength", type=float, default=1.0)

    parser.add_argument("--chart-corners-file", default=None, help="可选，复用 ColorChecker 四角 JSON。")
    parser.add_argument("--force-select-chart", action="store_true", help="强制重新点击色卡四角。")

    parser.add_argument("--board-roi-file", default=None, help="可选，复用板材 ROI JSON。")
    parser.add_argument("--force-select-board", action="store_true", help="强制重新框选板材 ROI。")

    parser.add_argument(
        "--trim-percent",
        type=float,
        default=10.0,
        help="板材 ROI 取 trimmed mean 的百分比，默认 10。",
    )

    parser.add_argument(
        "--board-display-l-offset",
        type=float,
        default=0.0,
        help="只用于展示的板材 L 提亮量，不影响 TopK 匹配。建议 1.0~2.0，默认 0。",
    )

    parser.add_argument(
        "--board-display-chroma-scale",
        type=float,
        default=1.0,
        help="只用于展示的板材 a/b 色度倍率，不影响 TopK 匹配。建议 1.03~1.06，默认 1。",
    )

    args = parser.parse_args()

    photo_path = Path(args.photo)
    standard_path = Path(args.standard)
    library_path = Path(args.library)
    mapping_path = Path(args.mapping)
    out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    original_bgr = imread_unicode(photo_path)
    standard_bgr = imread_unicode(standard_path)

    # ---------- 1. 选择/读取 ColorChecker 四角 ----------
    chart_corners_file = Path(args.chart_corners_file) if args.chart_corners_file else (out_dir / "board_chart_corners.json")

    if chart_corners_file.exists() and not args.force_select_chart:
        corners = json.loads(chart_corners_file.read_text(encoding="utf-8"))
        corners = [tuple(map(int, p)) for p in corners]
        print("已加载 ColorChecker 四角：", chart_corners_file)
    else:
        print("\n请依次点击板材照片中色卡四角：左上、右上、右下、左下。")
        corners = select_four_points(original_bgr, "Select ColorChecker corners")
        chart_corners_file.write_text(json.dumps(corners, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已保存 ColorChecker 四角：", chart_corners_file)

    chart_warp = warp_chart(original_bgr, corners, output_size=(600, 400))
    standard_chart = cv2.resize(standard_bgr, (600, 400), interpolation=cv2.INTER_AREA)

    imwrite_unicode(out_dir / "01_chart_warp.png", chart_warp)
    imwrite_unicode(out_dir / "01_standard_chart_resized.png", standard_chart)

    captured_rgb = extract_colorchecker_24_rgb(chart_warp)
    reference_rgb = extract_colorchecker_24_rgb(standard_chart)

    # ---------- 2. 拟合并应用 ColorChecker 校正 ----------
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

    imwrite_unicode(out_dir / "02_board_corrected.png", corrected_bgr)

    corrected_chart = warp_chart(corrected_bgr, corners, output_size=(600, 400))
    corrected_chart_rgb = extract_colorchecker_24_rgb(corrected_chart)

    chart_report = colorchecker_delta_report(
        captured_rgb=captured_rgb,
        corrected_rgb=corrected_chart_rgb,
        reference_rgb=reference_rgb,
    )

    # ---------- 3. 选择/读取板材 ROI ----------
    board_roi_file = Path(args.board_roi_file) if args.board_roi_file else (out_dir / "board_roi.json")

    if board_roi_file.exists() and not args.force_select_board:
        board_roi = load_roi_file(board_roi_file)
        print("已加载板材 ROI：", board_roi_file)
    else:
        print("\n请框选板材 ROI。板材比较均匀的话，框中间一块即可。")
        board_roi = select_board_roi(corrected_bgr)
        save_roi_file(board_roi_file, board_roi)
        print("已保存板材 ROI：", board_roi_file)

    draw_roi_preview(original_bgr, board_roi, out_dir / "03_board_roi_on_original.png", "board ROI")
    draw_roi_preview(corrected_bgr, board_roi, out_dir / "03_board_roi_on_corrected.png", "board ROI")

    # ---------- 4. 提取 board_corrected_lab ----------
    board_corrected_rgb = representative_rgb_from_roi(
        corrected_bgr,
        board_roi,
        trim_percent=args.trim_percent,
    )
    board_corrected_lab = rgb_to_lab(board_corrected_rgb.reshape(1, 3))[0]

    # ---------- 5. T 映射：corrected Lab -> visual Lab ----------
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    board_visual_lab = apply_visual_mapping_T(board_corrected_lab, mapping)

    # board_visual_lab：用于 TopK 匹配，不要为了“看起来亮一点”而改它。
    # board_display_lab：只用于展示色块/拼图，可以轻微提亮、提色度。
    board_display_lab = adjust_board_display_lab(
        board_visual_lab,
        l_offset=args.board_display_l_offset,
        chroma_scale=args.board_display_chroma_scale,
    )

    board_visual_rgb = lab_to_rgb(board_visual_lab.reshape(1, 3))[0]
    board_display_rgb = lab_to_rgb(board_display_lab.reshape(1, 3))[0]
    board_display_bgr = board_display_rgb[::-1]

    swatch = np.zeros((160, 480, 3), dtype=np.uint8)
    swatch[:, :160] = board_corrected_rgb[::-1].astype(np.uint8)
    swatch[:, 160:320] = board_visual_rgb[::-1].astype(np.uint8)
    swatch[:, 320:] = board_display_bgr.astype(np.uint8)
    cv2.putText(swatch, "corrected", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2)
    cv2.putText(swatch, "visual T", (188, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2)
    cv2.putText(swatch, "display", (350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2)
    imwrite_unicode(out_dir / "04_board_corrected_vs_visual_display_swatch.png", swatch)

    # ---------- 6. 匹配胶块视觉库 ----------
    library_rows = read_glue_visual_library(library_path)

    top_results = match_topk(
        board_visual_lab=board_visual_lab,
        library_rows=library_rows,
        top_k=args.top_k,
    )

    write_csv(out_dir / "05_board_match_topk.csv", top_results)

    result = {
        "input": {
            "photo": str(photo_path),
            "standard": str(standard_path),
            "library": str(library_path),
            "mapping": str(mapping_path),
        },
        "colorchecker_model": {
            "model_type": args.model_type,
            "ridge_alpha": args.ridge_alpha,
            "correction_strength": args.correction_strength,
            "chart_deltaE": chart_report,
        },
        "board": {
            "roi_xyxy": list(map(int, board_roi)),
            "corrected_rgb": [float(x) for x in board_corrected_rgb],
            "corrected_lab": [float(x) for x in board_corrected_lab],

            # 用于匹配，不做显示微调
            "visual_lab": [float(x) for x in board_visual_lab],
            "visual_rgb": [float(x) for x in board_visual_rgb],

            # 只用于展示，不参与 TopK 排序
            "display_lab": [float(x) for x in board_display_lab],
            "display_rgb": [float(x) for x in board_display_rgb],
            "display_adjust": {
                "l_offset": float(args.board_display_l_offset),
                "chroma_scale": float(args.board_display_chroma_scale),
            },
        },
        "mapping_info": {
            "version": mapping.get("version"),
            "feature_mode": mapping.get("feature_mode"),
            "L_mode": mapping.get("L_mode"),
            "ridge_alpha": mapping.get("ridge_alpha"),
        },
        "topk": top_results,
        "note": "匹配误差 deltaE_match = ΔE(board_visual_lab, glue_visual_display_lab)。",
    }

    write_json(out_dir / "05_board_match_result.json", result)

    make_contact_sheet(
        top_results=top_results,
        board_visual_lab=board_visual_lab,
        board_display_lab=board_display_lab,
        out_path=out_dir / "06_board_match_contact_sheet.png",
        base_dir=library_path.parent,
    )

    # ---------- 7. 终端输出 ----------
    print("\n==== 板材匹配完成 ====")
    print("输出目录：", out_dir)

    print("\nColorChecker ΔE：")
    print("  before mean:", chart_report["before"]["mean"])
    print("  after  mean:", chart_report["after"]["mean"])

    print("\nBoard corrected Lab:")
    print("  L={:.3f}, a={:.3f}, b={:.3f}".format(*board_corrected_lab))

    print("\nBoard visual Lab 用于匹配:")
    print("  L={:.3f}, a={:.3f}, b={:.3f}".format(*board_visual_lab))

    print("\nBoard display Lab 仅用于展示:")
    print("  L={:.3f}, a={:.3f}, b={:.3f}".format(*board_display_lab))
    print("  display adjust: L_offset={}, chroma_scale={}".format(args.board_display_l_offset, args.board_display_chroma_scale))

    print("\nTopK:")
    for row in top_results:
        print(
            "#{rank:02d} {code} {name}  ΔE={de:.3f}".format(
                rank=row["rank"],
                code=row.get("code") or "",
                name=row.get("name") or "",
                de=row["deltaE_match"],
            )
        )

    print("\n主要输出：")
    print("  corrected 图：", out_dir / "02_board_corrected.png")
    print("  ROI 预览：", out_dir / "03_board_roi_on_corrected.png")
    print("  色块对比：", out_dir / "04_board_corrected_vs_visual_display_swatch.png")
    print("  TopK CSV：", out_dir / "05_board_match_topk.csv")
    print("  TopK JSON：", out_dir / "05_board_match_result.json")
    print("  推荐拼图：", out_dir / "06_board_match_contact_sheet.png")


if __name__ == "__main__":
    main()
