# 颜色校正模型拟合

from __future__ import annotations

import cv2
import numpy as np

from .color_math import linear_to_srgb, srgb_to_linear


def build_features(linear_rgb: np.ndarray, model: str = "linear_bias") -> np.ndarray:
    # 线性 RGB 构建回归特征
    rgb = np.asarray(linear_rgb, dtype=np.float32)
    flat = rgb.reshape(-1, 3)

    R = flat[:, 0:1]
    G = flat[:, 1:2]
    B = flat[:, 2:3]
    ones = np.ones_like(R)

    if model == "linear_bias":
        return np.concatenate([R, G, B, ones], axis=1)

    if model == "poly2":
        return np.concatenate([R, G, B, R * R, G * G, B * B, R * G, R * B, G * B, ones], axis=1)

    raise ValueError(f"Unknown model: {model}")


def fit_correction_model(
    captured_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    model: str = "linear_bias",
    ridge_alpha: float = 1e-6,
) -> np.ndarray:
    # 在线性 RGB 空间中拟合实拍 RGB 到参考 RGB 的映射关系
    captured_linear = srgb_to_linear(captured_rgb)
    reference_linear = srgb_to_linear(reference_rgb)

    X = build_features(captured_linear, model=model)
    Y = reference_linear.reshape(-1, 3)

    if ridge_alpha > 0:
        reg = ridge_alpha * np.eye(X.shape[1], dtype=np.float32)
        reg[-1, -1] = 0.0
        W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    else:
        W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    return W.astype(np.float32)


def apply_correction_to_image(img_bgr: np.ndarray, W: np.ndarray, model: str = "linear_bias") -> np.ndarray:
    # 将颜色校正模型应用到整张 BGR 图像
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_linear = srgb_to_linear(img_rgb)

    h, w = img_linear.shape[:2]
    X = build_features(img_linear, model=model)
    corrected_linear = (X @ W).reshape(h, w, 3)

    corrected_rgb = linear_to_srgb(corrected_linear)
    return cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
