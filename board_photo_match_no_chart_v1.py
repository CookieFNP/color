# -*- coding: utf-8 -*-
"""
用途：
    板材端【无色卡快速匹配】脚本，方案 1：固定校正模型。

    这个脚本不再要求每张板材照片里放 ColorChecker 色卡。
    它会读取你之前某次带色卡流程生成的 report.json 中的 ColorChecker 校正模型：
        report["model"]["weights"]

    然后对新的无色卡板材照片直接套用这个固定模型：
        原图
        ↓
        固定 ColorChecker 模型校正
        ↓
        手动框选板材 ROI
        ↓
        corrected_lab
        ↓
        visual_mapping_T
        ↓
        和合并后的 glue_visual_library.csv 做 TopK 匹配

    当前脚本已经固化你最后确认的板材展示参数：
        board-display-l-offset      = -5
        board-display-chroma-scale  = 0.95
        board-display-chroma-offset = 0
        board-display-a-offset      = -1.5
        board-display-b-offset      = -1.5

    注意：
        1. 无色卡模式适合固定设备、固定光照、固定拍摄环境。
        2. 如果换手机、换光源、自动白平衡飘了，结果会比有色卡模式更不稳定。
        3. TopK 匹配仍然用 board_visual_lab；display 参数只影响展示图和 ROI 截图。

典型运行：
    python board_photo_match_no_chart_v1.py ^
      --photo data/test/P8316.jpg ^
      --calibration-report output_zhengwei2/report.json ^
      --library output_combined/output_combined/glue_visual_library.csv ^
      --mapping output_combined/visual_mapping_T_poly2/visual_mapping_T.json ^
      --out board_match_no_chart ^
      --top-k 10
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
    chroma_offset: float = 0.0,
    a_offset: float = 0.0,
    b_offset: float = 0.0,
    chroma_max: float = 80.0,
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
            如果不是发暗，只是发灰，可以设为 0。

        chroma_scale:
            a/b 色度倍率。
            只改变颜色浓淡，不改变整体冷暖方向。

        chroma_offset:
            固定色度增量。
            沿原 hue 方向增加或减少色度。

        a_offset:
            红绿轴偏移。
            a_offset > 0 表示略偏红，a_offset < 0 表示略偏绿。

        b_offset:
            黄蓝轴偏移。
            b_offset > 0 表示加暖/加黄，b_offset < 0 表示偏冷/偏蓝。
            如果肉眼觉得现场有暖色调而结果发灰，优先试 b_offset=1~3。

        chroma_max:
            色度上限，防止显示色过艳。
    """
    lab = np.asarray(board_visual_lab, dtype=np.float64).copy()

    lab[0] = np.clip(lab[0] + float(l_offset), 0.0, 100.0)

    a = float(lab[1])
    b = float(lab[2])
    C = float(np.sqrt(a * a + b * b))

    if C > 1e-6:
        C_new = C * float(chroma_scale) + float(chroma_offset)
        C_new = np.clip(C_new, 0.0, float(chroma_max))
        ratio = C_new / C
        lab[1] = a * ratio
        lab[2] = b * ratio
    else:
        # 几乎中性灰，没有可靠 hue，不强行染色。
        lab[1] = a * float(chroma_scale)
        lab[2] = b * float(chroma_scale)

    # 最后叠加“冷暖/红绿”的显示偏置。
    # 这一步是为了模拟肉眼现场暖色调，只用于 display，不用于匹配。
    lab[1] = lab[1] + float(a_offset)
    lab[2] = lab[2] + float(b_offset)

    # 再限制一次总色度。
    a2 = float(lab[1])
    b2 = float(lab[2])
    C2 = float(np.sqrt(a2 * a2 + b2 * b2))
    if C2 > float(chroma_max) and C2 > 1e-6:
        ratio2 = float(chroma_max) / C2
        lab[1] = a2 * ratio2
        lab[2] = b2 * ratio2

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


def adjust_lab_image_for_display(
    lab_img: np.ndarray,
    l_offset: float = 0.0,
    chroma_scale: float = 1.0,
    chroma_offset: float = 0.0,
    a_offset: float = 0.0,
    b_offset: float = 0.0,
    chroma_max: float = 80.0,
) -> np.ndarray:
    """
    用途：
        对一整块 ROI 图像的每个像素做和 board_display_lab 类似的显示层微调。

    注意：
        只用于输出 07_board_roi_crop_display_adjusted.png，
        不参与 TopK 匹配，不改变 board_visual_lab。
    """
    lab = np.asarray(lab_img, dtype=np.float64).copy()

    lab[..., 0] = np.clip(lab[..., 0] + float(l_offset), 0.0, 100.0)

    a = lab[..., 1]
    b = lab[..., 2]
    C = np.sqrt(a * a + b * b)

    ratio = np.ones_like(C, dtype=np.float64)
    mask = C > 1e-6
    C_new = C * float(chroma_scale) + float(chroma_offset)
    C_new = np.clip(C_new, 0.0, float(chroma_max))
    ratio[mask] = C_new[mask] / C[mask]

    lab[..., 1] = a * ratio + float(a_offset)
    lab[..., 2] = b * ratio + float(b_offset)

    # 限制总色度，避免个别像素过艳。
    a2 = lab[..., 1]
    b2 = lab[..., 2]
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    mask2 = C2 > float(chroma_max)
    if np.any(mask2):
        ratio2 = float(chroma_max) / np.maximum(C2, 1e-6)
        lab[..., 1] = np.where(mask2, a2 * ratio2, a2)
        lab[..., 2] = np.where(mask2, b2 * ratio2, b2)

    return lab


def render_and_save_board_roi_crops(
    *,
    original_bgr: np.ndarray,
    corrected_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    out_dir: Path,
    l_offset: float,
    chroma_scale: float,
    chroma_offset: float,
    a_offset: float,
    b_offset: float,
    chroma_max: float,
) -> None:
    """
    用途：
        把用户框选的板材 ROI 单独截出来，并额外输出一张应用 display 参数后的 ROI 图。

    输出：
        07_board_roi_crop_original.png
        07_board_roi_crop_corrected.png
        07_board_roi_crop_display_adjusted.png
        07_board_roi_crop_compare.png
    """
    x1, y1, x2, y2 = roi
    original_crop = original_bgr[y1:y2, x1:x2].copy()
    corrected_crop = corrected_bgr[y1:y2, x1:x2].copy()

    if corrected_crop.size == 0:
        return

    corrected_rgb = corrected_crop[..., ::-1].astype(np.float64)
    lab_img = rgb_to_lab(corrected_rgb.reshape(-1, 3)).reshape(corrected_rgb.shape)
    display_lab_img = adjust_lab_image_for_display(
        lab_img,
        l_offset=l_offset,
        chroma_scale=chroma_scale,
        chroma_offset=chroma_offset,
        a_offset=a_offset,
        b_offset=b_offset,
        chroma_max=chroma_max,
    )
    display_rgb = lab_to_rgb(display_lab_img.reshape(-1, 3)).reshape(corrected_rgb.shape)
    display_bgr = display_rgb[..., ::-1].astype(np.uint8)

    imwrite_unicode(out_dir / "07_board_roi_crop_original.png", original_crop)
    imwrite_unicode(out_dir / "07_board_roi_crop_corrected.png", corrected_crop)
    imwrite_unicode(out_dir / "07_board_roi_crop_display_adjusted.png", display_bgr)

    # 做一张横向对比图，方便一眼看。
    h = max(original_crop.shape[0], corrected_crop.shape[0], display_bgr.shape[0])
    w = original_crop.shape[1] + corrected_crop.shape[1] + display_bgr.shape[1]
    compare = np.full((h + 42, w, 3), 245, dtype=np.uint8)

    x = 0
    for title, img in [
        ("original ROI", original_crop),
        ("corrected ROI", corrected_crop),
        ("display adjusted ROI", display_bgr),
    ]:
        compare[42:42 + img.shape[0], x:x + img.shape[1]] = img
        cv2.putText(compare, title, (x + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        x += img.shape[1]

    imwrite_unicode(out_dir / "07_board_roi_crop_compare.png", compare)


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



def parse_float_list(text: str | None) -> list[float]:
    """
    解析命令行传入的浮点数列表，例如：
        "0,0.5,1.0"
        "0;0.5;1.0"
    """
    if text is None:
        return []

    parts = re.split(r"[,;，；\s]+", str(text).strip())
    out = []

    for p in parts:
        if not p:
            continue
        out.append(float(p))

    return out


def build_display_variants(
    board_visual_lab: np.ndarray,
    l_offsets: list[float],
    chroma_scales: list[float],
    chroma_offsets: list[float],
    a_offsets: list[float],
    b_offsets: list[float],
    chroma_max: float,
) -> list[dict]:
    """
    生成多组 board_display_lab 方案。

    注意：
        所有 variant 都只用于展示，不参与 TopK 匹配。
    """
    if not l_offsets:
        l_offsets = [0.0]
    if not chroma_scales:
        chroma_scales = [1.0]
    if not chroma_offsets:
        chroma_offsets = [0.0]
    if not a_offsets:
        a_offsets = [0.0]
    if not b_offsets:
        b_offsets = [0.0]

    variants = []

    for l_off in l_offsets:
        for c_scale in chroma_scales:
            for c_off in chroma_offsets:
                for a_off in a_offsets:
                    for b_off in b_offsets:
                        lab = adjust_board_display_lab(
                            board_visual_lab,
                            l_offset=l_off,
                            chroma_scale=c_scale,
                            chroma_offset=c_off,
                            a_offset=a_off,
                            b_offset=b_off,
                            chroma_max=chroma_max,
                        )
                        variants.append(
                            {
                                "name": f"L{l_off:+.1f}_C{c_scale:.2f}_O{c_off:+.1f}_a{a_off:+.1f}_b{b_off:+.1f}",
                                "l_offset": float(l_off),
                                "chroma_scale": float(c_scale),
                                "chroma_offset": float(c_off),
                                "a_offset": float(a_off),
                                "b_offset": float(b_off),
                                "lab": lab,
                            }
                        )

    return variants


def make_display_variants_sheet(
    *,
    variants: list[dict],
    out_path: Path,
    tile_w: int = 220,
    tile_h: int = 210,
) -> None:
    """
    输出多组板材展示色块，方便肉眼选择“淡一点/艳一点/亮一点”。
    """
    if not variants:
        return

    cols = min(5, len(variants))
    rows = int(math.ceil(len(variants) / cols))
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, item in enumerate(variants):
        r = idx // cols
        c = idx % cols
        y0 = r * tile_h
        x0 = c * tile_w

        tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)

        lab = np.asarray(item["lab"], dtype=np.float64)
        swatch = color_swatch_bgr_from_lab(lab, size=(150, 120))
        tile[45:165, 35:185] = swatch

        title = item["name"]
        cv2.putText(tile, title[:22], (12, 28), font, 0.55, (0, 0, 0), 2)
        cv2.putText(tile, f"L={lab[0]:.1f}", (12, 182), font, 0.50, (0, 0, 0), 1)
        cv2.putText(tile, f"a={lab[1]:.1f} b={lab[2]:.1f}", (12, 202), font, 0.50, (0, 0, 0), 1)

        sheet[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    imwrite_unicode(out_path, sheet)


def make_multi_variant_contact_sheet(
    *,
    top_results: list[dict],
    display_variants: list[dict],
    out_path: Path,
    base_dir: Path,
    tile_w: int = 220,
    tile_h: int = 270,
) -> None:
    """
    每一行一个 board display variant，后面跟同一组 TopK 胶块。
    用于比较“板材展示色块”与推荐胶块在不同显示参数下的观感。
    TopK 本身不随 display variant 改变。
    """
    if not display_variants:
        return

    show_top = top_results[: min(4, len(top_results))]
    cols = 1 + len(show_top)
    rows = len(display_variants)

    sheet = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for rr, variant in enumerate(display_variants):
        y0 = rr * tile_h

        # board tile
        tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
        lab = np.asarray(variant["lab"], dtype=np.float64)
        swatch = color_swatch_bgr_from_lab(lab, size=(150, 150))
        tile[45:195, 35:185] = swatch
        cv2.putText(tile, variant["name"][:20], (12, 26), font, 0.55, (0, 0, 0), 2)
        cv2.putText(tile, "BOARD display", (12, 218), font, 0.50, (0, 0, 0), 1)
        cv2.putText(tile, f"L={lab[0]:.1f} a={lab[1]:.1f} b={lab[2]:.1f}", (12, 245), font, 0.45, (0, 0, 0), 1)
        sheet[y0:y0 + tile_h, 0:tile_w] = tile

        # top glue tiles
        for cc, row in enumerate(show_top, start=1):
            x0 = cc * tile_w
            glue_tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)

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
                glue_tile[yy:yy + crop_resized.shape[0], xx:xx + crop_resized.shape[1]] = crop_resized
            else:
                lab_g = np.asarray(
                    [
                        float(row["visual_display_L"]),
                        float(row["visual_display_a"]),
                        float(row["visual_display_b"]),
                    ],
                    dtype=np.float64,
                )
                swatch_g = color_swatch_bgr_from_lab(lab_g, size=(150, 150))
                glue_tile[45:195, 35:185] = swatch_g

            cv2.putText(glue_tile, f"#{row['rank']} {row.get('code','')}"[:18], (12, 26), font, 0.65, (0, 0, 0), 2)
            cv2.putText(glue_tile, str(row.get("name") or "")[:12], (12, 218), font, 0.55, (0, 0, 0), 1)
            cv2.putText(glue_tile, f"dE={float(row.get('deltaE_match', 0)):.2f}", (12, 245), font, 0.6, (0, 0, 0), 2)

            sheet[y0:y0 + tile_h, x0:x0 + tile_w] = glue_tile

    imwrite_unicode(out_path, sheet)


# ============================================================

# ============================================================
# 固定校正模型读取
# ============================================================

def load_fixed_calibration_from_report(report_path: str | Path) -> dict:
    """
    从之前带色卡流程生成的 report.json 读取固定 ColorChecker 校正模型。

    支持的 report 结构：
        {
          "model": {
            "type": "root_poly2",
            "ridge_alpha": 1e-6,
            "correction_strength": 1.0,
            "weights": [...]
          }
        }

    也兼容少量扁平结构：
        {
          "type": "...",
          "weights": [...]
        }
    """
    report_path = Path(report_path)
    data = json.loads(report_path.read_text(encoding="utf-8"))

    model = data.get("model")
    if model is None:
        model = data

    weights = model.get("weights")
    if weights is None:
        raise RuntimeError(
            "这个 calibration report 里没有 model.weights，不能用于无色卡固定校正。\n"
            "请使用 main.py / 胶块建库流程生成的 report.json，里面应包含 model.weights。\n"
            "注意：board_photo_match_v6 输出的 05_board_match_result.json 通常不含 weights，不能直接当校正模型。"
        )

    model_type = model.get("type") or model.get("model_type") or data.get("model_type")
    if model_type is None:
        raise RuntimeError("calibration report 里找不到 model.type / model_type。")

    W = np.asarray(weights, dtype=np.float64)

    return {
        "source_report": str(report_path),
        "model_type": str(model_type),
        "ridge_alpha": model.get("ridge_alpha"),
        "correction_strength_from_report": model.get("correction_strength"),
        "weights": W,
        "source_chart_deltaE": data.get("chart_delta_e_2000") or data.get("chart_deltaE"),
    }

# ============================================================
# 主流程：无色卡固定模型
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-chart board photo -> fixed ColorChecker model -> visual_mapping_T -> glue visual library TopK."
    )

    parser.add_argument("--photo", required=True, help="无色卡板材原图")
    parser.add_argument(
        "--calibration-report",
        required=True,
        help="之前带色卡流程生成的 report.json，必须包含 model.weights。",
    )
    parser.add_argument("--library", required=True, help="合并后的 glue_visual_library.csv")
    parser.add_argument("--mapping", required=True, help="combined visual_mapping_T.json")
    parser.add_argument("--out", default="board_match_no_chart", help="输出目录")
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument(
        "--correction-strength",
        type=float,
        default=1.0,
        help="固定校正模型强度。1=完整套用，0.5=半强度。默认 1。",
    )

    parser.add_argument("--board-roi-file", default=None, help="可选，复用板材 ROI JSON。")
    parser.add_argument("--force-select-board", action="store_true", help="强制重新框选板材 ROI。")

    parser.add_argument(
        "--trim-percent",
        type=float,
        default=10.0,
        help="板材 ROI 取 trimmed mean 的百分比，默认 10。",
    )

    # 这里固化你最后确认的展示参数。
    parser.add_argument(
        "--board-display-l-offset",
        type=float,
        default=-3.0,
        help="只用于展示的板材 L 偏移，不影响 TopK。默认 -5。",
    )
    parser.add_argument(
        "--board-display-chroma-scale",
        type=float,
        default=0.95,
        help="只用于展示的板材 a/b 色度倍率，不影响 TopK。默认 0.95。",
    )
    parser.add_argument(
        "--board-display-chroma-offset",
        type=float,
        default=0.0,
        help="只用于展示的板材固定色度增量，不影响 TopK。默认 0。",
    )
    parser.add_argument(
        "--board-display-a-offset",
        type=float,
        default=-0.5,
        help="只用于展示的 a 轴偏移，不影响 TopK。默认 -1.5。",
    )
    parser.add_argument(
        "--board-display-b-offset",
        type=float,
        default=-0.5,
        help="只用于展示的 b 轴偏移，不影响 TopK。默认 -1.5。",
    )
    parser.add_argument(
        "--board-display-chroma-max",
        type=float,
        default=100.0,
        help="只用于展示的板材色度上限，防止过艳，默认 80。",
    )

    parser.add_argument(
        "--show-display-variants",
        action="store_true",
        help="输出多组板材 display 色块对比图，方便选择展示参数。",
    )
    parser.add_argument(
        "--display-l-offsets",
        default="-8,-5,-3,0",
        help='多版本展示用 L 偏移列表，例如 "-8,-5,-3,0"。',
    )
    parser.add_argument(
        "--display-chroma-scales",
        default="0.90,0.95,1.00",
        help='多版本展示用色度倍率列表。',
    )
    parser.add_argument(
        "--display-chroma-offsets",
        default="0",
        help='多版本展示用固定色度增量列表。',
    )
    parser.add_argument(
        "--display-a-offsets",
        default="-2,-1.5,-1,0",
        help='多版本展示用 a 轴偏移列表。',
    )
    parser.add_argument(
        "--display-b-offsets",
        default="-2,-1.5,-1,0",
        help='多版本展示用 b 轴偏移列表。',
    )

    args = parser.parse_args()

    photo_path = Path(args.photo)
    calibration_report_path = Path(args.calibration_report)
    library_path = Path(args.library)
    mapping_path = Path(args.mapping)
    out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    original_bgr = imread_unicode(photo_path)

    # ---------- 1. 读取固定校正模型 ----------
    fixed_model = load_fixed_calibration_from_report(calibration_report_path)
    W = fixed_model["weights"]
    model_type = fixed_model["model_type"]

    # 保存一份本次实际使用的固定模型信息，方便追溯。
    model_copy = {
        "source_report": fixed_model["source_report"],
        "model_type": model_type,
        "ridge_alpha": fixed_model.get("ridge_alpha"),
        "correction_strength_from_report": fixed_model.get("correction_strength_from_report"),
        "correction_strength_used": float(args.correction_strength),
        "source_chart_deltaE": fixed_model.get("source_chart_deltaE"),
        "note": "无色卡快速模式：本次图片没有现场 ColorChecker，直接套用 source_report 中的固定校正模型。",
    }
    write_json(out_dir / "00_fixed_calibration_model_used.json", model_copy)

    # ---------- 2. 应用固定 ColorChecker 校正 ----------
    corrected_bgr = apply_color_correction_image(
        original_bgr,
        W=W,
        model_type=model_type,
        correction_strength=args.correction_strength,
    )
    imwrite_unicode(out_dir / "02_board_corrected.png", corrected_bgr)

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

    # v6/v7：把框选的 ROI 单独截出来，并输出一张按 display 参数微调后的 ROI 图。
    render_and_save_board_roi_crops(
        original_bgr=original_bgr,
        corrected_bgr=corrected_bgr,
        roi=board_roi,
        out_dir=out_dir,
        l_offset=args.board_display_l_offset,
        chroma_scale=args.board_display_chroma_scale,
        chroma_offset=args.board_display_chroma_offset,
        a_offset=args.board_display_a_offset,
        b_offset=args.board_display_b_offset,
        chroma_max=args.board_display_chroma_max,
    )

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

    # board_visual_lab：用于 TopK 匹配，不要为了“看起来更像”而改它。
    # board_display_lab：只用于展示色块/拼图/ROI展示。
    board_display_lab = adjust_board_display_lab(
        board_visual_lab,
        l_offset=args.board_display_l_offset,
        chroma_scale=args.board_display_chroma_scale,
        chroma_offset=args.board_display_chroma_offset,
        a_offset=args.board_display_a_offset,
        b_offset=args.board_display_b_offset,
        chroma_max=args.board_display_chroma_max,
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

    display_variants = []
    if args.show_display_variants:
        display_variants = build_display_variants(
            board_visual_lab=board_visual_lab,
            l_offsets=parse_float_list(args.display_l_offsets),
            chroma_scales=parse_float_list(args.display_chroma_scales),
            chroma_offsets=parse_float_list(args.display_chroma_offsets),
            a_offsets=parse_float_list(args.display_a_offsets),
            b_offsets=parse_float_list(args.display_b_offsets),
            chroma_max=args.board_display_chroma_max,
        )
        make_display_variants_sheet(
            variants=display_variants,
            out_path=out_dir / "04b_board_display_variants.png",
        )
        make_multi_variant_contact_sheet(
            top_results=top_results,
            display_variants=display_variants,
            out_path=out_dir / "06b_board_match_display_variants_contact_sheet.png",
            base_dir=library_path.parent,
        )

    result = {
        "input": {
            "photo": str(photo_path),
            "calibration_report": str(calibration_report_path),
            "library": str(library_path),
            "mapping": str(mapping_path),
        },
        "fixed_color_calibration": {
            "source_report": fixed_model["source_report"],
            "model_type": model_type,
            "ridge_alpha": fixed_model.get("ridge_alpha"),
            "correction_strength_from_report": fixed_model.get("correction_strength_from_report"),
            "correction_strength_used": float(args.correction_strength),
            "source_chart_deltaE": fixed_model.get("source_chart_deltaE"),
            "note": "无色卡模式没有本张图现场 ColorChecker ΔE，只能追溯固定模型来源 report 的色卡指标。",
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
                "chroma_offset": float(args.board_display_chroma_offset),
                "a_offset": float(args.board_display_a_offset),
                "b_offset": float(args.board_display_b_offset),
                "chroma_max": float(args.board_display_chroma_max),
            },
        },
        "mapping_info": {
            "version": mapping.get("version"),
            "feature_mode": mapping.get("feature_mode"),
            "L_mode": mapping.get("L_mode"),
            "ridge_alpha": mapping.get("ridge_alpha"),
        },
        "topk": top_results,
        "display_variants": [
            {
                "name": v["name"],
                "l_offset": v["l_offset"],
                "chroma_scale": v["chroma_scale"],
                "chroma_offset": v["chroma_offset"],
                "a_offset": v.get("a_offset", 0.0),
                "b_offset": v.get("b_offset", 0.0),
                "lab": [float(x) for x in v["lab"]],
            }
            for v in display_variants
        ],
        "note": "匹配误差 deltaE_match = ΔE(board_visual_lab, glue_visual_display_lab)。display 参数只用于展示，不参与 TopK 排序。",
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
    print("\n==== 无色卡板材匹配完成 ====")
    print("输出目录：", out_dir)

    print("\n固定校正模型：")
    print("  source report:", fixed_model["source_report"])
    print("  model_type:", model_type)
    print("  correction_strength_used:", args.correction_strength)
    src_de = fixed_model.get("source_chart_deltaE")
    if isinstance(src_de, dict):
        # 兼容 main.py 的 chart_delta_e_2000 格式
        after = src_de.get("after")
        before = src_de.get("before")
        if isinstance(before, dict) and isinstance(after, dict):
            print("  source ColorChecker before mean:", before.get("mean"))
            print("  source ColorChecker after  mean:", after.get("mean"))

    print("\nBoard corrected Lab:")
    print("  L={:.3f}, a={:.3f}, b={:.3f}".format(*board_corrected_lab))

    print("\nBoard visual Lab 用于匹配:")
    print("  L={:.3f}, a={:.3f}, b={:.3f}".format(*board_visual_lab))

    print("\nBoard display Lab 仅用于展示:")
    print("  L={:.3f}, a={:.3f}, b={:.3f}".format(*board_display_lab))
    print("  display adjust: L_offset={}, chroma_scale={}, chroma_offset={}, a_offset={}, b_offset={}, chroma_max={}".format(
        args.board_display_l_offset,
        args.board_display_chroma_scale,
        args.board_display_chroma_offset,
        args.board_display_a_offset,
        args.board_display_b_offset,
        args.board_display_chroma_max,
    ))

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
    print("  固定模型记录：", out_dir / "00_fixed_calibration_model_used.json")
    print("  corrected 图：", out_dir / "02_board_corrected.png")
    print("  ROI 预览：", out_dir / "03_board_roi_on_corrected.png")
    print("  ROI 截图：", out_dir / "07_board_roi_crop_display_adjusted.png")
    print("  ROI 对比：", out_dir / "07_board_roi_crop_compare.png")
    print("  色块对比：", out_dir / "04_board_corrected_vs_visual_display_swatch.png")
    print("  TopK CSV：", out_dir / "05_board_match_topk.csv")
    print("  TopK JSON：", out_dir / "05_board_match_result.json")
    print("  推荐拼图：", out_dir / "06_board_match_contact_sheet.png")
    if args.show_display_variants:
        print("  多版本板材色块：", out_dir / "04b_board_display_variants.png")
        print("  多版本匹配拼图：", out_dir / "06b_board_match_display_variants_contact_sheet.png")


if __name__ == "__main__":
    main()
