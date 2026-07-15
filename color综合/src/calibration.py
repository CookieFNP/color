from __future__ import annotations

import cv2
import numpy as np

from .color_math import linear_to_srgb, srgb_to_linear


def build_features(linear_rgb: np.ndarray, model: str = "root_poly2") -> np.ndarray:
    rgb = np.asarray(linear_rgb, dtype=np.float64)
    flat = rgb.reshape(-1, 3)
    R = flat[:, 0:1]
    G = flat[:, 1:2]
    B = flat[:, 2:3]
    ones = np.ones_like(R)

    if model == "linear_bias":
        return np.concatenate([R, G, B, ones], axis=1)

    if model == "poly2":
        return np.concatenate([R, G, B, R * R, G * G, B * B, R * G, R * B, G * B, ones], axis=1)

    if model == "root_poly2":
        eps = 1e-12
        return np.concatenate(
            [
                R,
                G,
                B,
                np.sqrt(np.maximum(R * G, eps)),
                np.sqrt(np.maximum(R * B, eps)),
                np.sqrt(np.maximum(G * B, eps)),
                ones,
            ],
            axis=1,
        )

    raise ValueError(f"未知模型：{model}")


def fit_correction_model(
    captured_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    model: str = "root_poly2",
    ridge_alpha: float = 1e-6,
) -> np.ndarray:
    captured_linear = srgb_to_linear(captured_rgb)
    reference_linear = srgb_to_linear(reference_rgb)

    X = build_features(captured_linear, model=model)
    Y = reference_linear.reshape(-1, 3)

    if ridge_alpha > 0:
        reg = ridge_alpha * np.eye(X.shape[1], dtype=np.float64)
        reg[-1, -1] = 0.0
        W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    else:
        W, *_ = np.linalg.lstsq(X, Y, rcond=None)

    return W.astype(np.float64)


def predict_rgb(rgb_255: np.ndarray, W: np.ndarray, model: str = "root_poly2") -> np.ndarray:
    linear = srgb_to_linear(rgb_255)
    X = build_features(linear, model=model)
    fixed_linear = np.clip(X @ W, 0.0, 1.0)
    return linear_to_srgb(fixed_linear.reshape(np.asarray(rgb_255).shape), as_u8=True)


def apply_correction_to_image(
    img_bgr: np.ndarray,
    W: np.ndarray,
    model: str = "root_poly2",
    correction_strength: float = 1.0,
) -> np.ndarray:
    strength = float(np.clip(correction_strength, 0.0, 1.0))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_linear = srgb_to_linear(img_rgb)

    h, w = img_rgb.shape[:2]
    X = build_features(img_linear.reshape(-1, 3), model=model)
    corrected_linear = np.clip(X @ W, 0.0, 1.0).reshape(h, w, 3)

    mixed_linear = (1.0 - strength) * img_linear + strength * corrected_linear
    corrected_rgb = linear_to_srgb(mixed_linear, as_u8=True)
    return cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
