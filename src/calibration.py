# 颜色校正模型拟合

from __future__ import annotations

import cv2
import numpy as np

from .color_math import linear_to_srgb, srgb_to_linear


SUPPORTED_MODELS = (
    "linear_bias",
    "ccm",
    "poly2",
    "poly3",
    "root_poly2",
    "root_poly3",
)


def _normalize_model_name(model: str) -> str:
    """Normalize aliases used from CLI / reports."""
    model = (model or "linear_bias").strip().lower().replace("-", "_")
    aliases = {
        "linear": "linear_bias",
        "degree1": "linear_bias",
        "ccm": "linear_bias",
        "degree2": "poly2",
        "degree3": "poly3",
        "root2": "root_poly2",
        "root_poly": "root_poly2",
        "rpcc": "root_poly2",
        "root3": "root_poly3",
    }
    return aliases.get(model, model)


def build_features(linear_rgb: np.ndarray, model: str = "linear_bias") -> np.ndarray:
    """
    Build regression features from linear RGB.

    Supported models:
        linear_bias / ccm:
            [R, G, B, 1]

        poly2:
            [R, G, B, R², G², B², RG, RB, GB, 1]

        poly3:
            [R, G, B, R², G², B², RG, RB, GB,
             R³, G³, B³, R²G, R²B, G²R, G²B, B²R, B²G, RGB, 1]

        root_poly2:
            [R, G, B, sqrt(RG), sqrt(RB), sqrt(GB), 1]

        root_poly3:
            root_poly2 + cubic-root homogeneous terms, e.g. cbrt(R²G), cbrt(RGB)
    """
    rgb = np.asarray(linear_rgb, dtype=np.float64)
    flat = rgb.reshape(-1, 3)

    R = flat[:, 0:1]
    G = flat[:, 1:2]
    B = flat[:, 2:3]
    ones = np.ones_like(R)

    model = _normalize_model_name(model)

    if model == "linear_bias":
        return np.concatenate([R, G, B, ones], axis=1)

    if model == "poly2":
        return np.concatenate(
            [R, G, B, R * R, G * G, B * B, R * G, R * B, G * B, ones],
            axis=1,
        )

    if model == "poly3":
        return np.concatenate(
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
                R ** 3,
                G ** 3,
                B ** 3,
                R * R * G,
                R * R * B,
                G * G * R,
                G * G * B,
                B * B * R,
                B * B * G,
                R * G * B,
                ones,
            ],
            axis=1,
        )

    if model == "root_poly2":
        # Root polynomial keeps non-linear channel interaction but is less sensitive to exposure scaling
        # than ordinary polynomial terms like R²/G²/B².
        return np.concatenate(
            [
                R,
                G,
                B,
                np.sqrt(R * G),
                np.sqrt(R * B),
                np.sqrt(G * B),
                ones,
            ],
            axis=1,
        )

    if model == "root_poly3":
        return np.concatenate(
            [
                R,
                G,
                B,
                np.sqrt(R * G),
                np.sqrt(R * B),
                np.sqrt(G * B),
                np.cbrt(R * R * G),
                np.cbrt(R * R * B),
                np.cbrt(G * G * R),
                np.cbrt(G * G * B),
                np.cbrt(B * B * R),
                np.cbrt(B * B * G),
                np.cbrt(R * G * B),
                ones,
            ],
            axis=1,
        )

    raise ValueError(f"Unknown model: {model}. Supported models: {SUPPORTED_MODELS}")


def fit_correction_model(
    captured_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    model: str = "linear_bias",
    ridge_alpha: float = 1e-6,
) -> np.ndarray:
    """Fit captured RGB -> reference RGB in linear RGB space."""
    captured_linear = srgb_to_linear(captured_rgb)
    reference_linear = srgb_to_linear(reference_rgb)

    X = build_features(captured_linear, model=model)
    Y = reference_linear.reshape(-1, 3).astype(np.float64)

    ridge_alpha = float(ridge_alpha)
    if ridge_alpha > 0:
        reg = ridge_alpha * np.eye(X.shape[1], dtype=np.float64)
        # Do not penalize bias term.
        reg[-1, -1] = 0.0
        try:
            W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
        except np.linalg.LinAlgError:
            W, _, _, _ = np.linalg.lstsq(X.T @ X + reg, X.T @ Y, rcond=None)
    else:
        W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    return W.astype(np.float32)


def apply_correction_to_image(img_bgr: np.ndarray, W: np.ndarray, model: str = "linear_bias") -> np.ndarray:
    """Apply a fitted color correction model to a whole BGR image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_linear = srgb_to_linear(img_rgb)

    h, w = img_linear.shape[:2]
    X = build_features(img_linear, model=model)
    corrected_linear = (X @ W).reshape(h, w, 3)

    corrected_rgb = linear_to_srgb(corrected_linear)
    return cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
