# 颜色校正模型拟合：支持 root_poly2 + 加权最小二乘
from __future__ import annotations

import cv2
import numpy as np

from .color_math import linear_to_srgb, srgb_to_linear, rgb_to_lab


def build_features(linear_rgb: np.ndarray, model: str = "linear_bias") -> np.ndarray:
    """
    在线性 RGB 空间构建颜色校正回归特征。

    linear_bias:
        [R, G, B, 1]

    poly2:
        [R, G, B, R^2, G^2, B^2, RG, RB, GB, 1]

    root_poly2:
        [R, G, B, sqrt(RG), sqrt(RB), sqrt(GB), 1]

    poly3:
        简单三阶多项式扩展，样本少时容易过拟合，谨慎使用。

    root_poly3:
        root_poly2 + 部分三阶根特征，谨慎使用。
    """
    rgb = np.asarray(linear_rgb, dtype=np.float32)
    flat = rgb.reshape(-1, 3)

    R = flat[:, 0:1]
    G = flat[:, 1:2]
    B = flat[:, 2:3]
    ones = np.ones_like(R)

    eps = np.float32(1e-12)

    if model == "linear_bias":
        return np.concatenate([R, G, B, ones], axis=1)

    if model == "poly2":
        return np.concatenate(
            [R, G, B, R * R, G * G, B * B, R * G, R * B, G * B, ones],
            axis=1,
        )

    if model == "root_poly2":
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

    if model == "root_poly3":
        return np.concatenate(
            [
                R,
                G,
                B,
                np.sqrt(np.maximum(R * G, eps)),
                np.sqrt(np.maximum(R * B, eps)),
                np.sqrt(np.maximum(G * B, eps)),
                np.cbrt(np.maximum(R * R * G, eps)),
                np.cbrt(np.maximum(R * R * B, eps)),
                np.cbrt(np.maximum(G * G * R, eps)),
                np.cbrt(np.maximum(G * G * B, eps)),
                np.cbrt(np.maximum(B * B * R, eps)),
                np.cbrt(np.maximum(B * B * G, eps)),
                np.cbrt(np.maximum(R * G * B, eps)),
                ones,
            ],
            axis=1,
        )

    raise ValueError(f"Unknown model: {model}")


def build_chart_sample_weights(
    reference_rgb: np.ndarray,
    *,
    mode: str = "none",
    gray_weight: float = 4.0,
    light_weight: float = 2.5,
    light_l_threshold: float = 70.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    为 ColorChecker 24 个色块构建拟合权重。

    mode:
        none:
            全部权重为 1，等价普通 root_poly2。

        gray:
            提高最后一排灰阶权重。常见 4x6 ColorChecker 最后一排是中性色，
            对白平衡、明度、黄蓝偏差很关键。

        light:
            提高高 L 浅色块权重。适合胶块以浅黄、米色、浅灰为主的场景。

        gray_light:
            同时提高灰阶和浅色块权重，当前胶块项目建议先试这个。

    注意：
        这里仍然只用色卡 24 点拟合，不把当前胶块标准值直接塞进训练。
    """
    ref = np.asarray(reference_rgb, dtype=np.float32).reshape(-1, 3)
    n = ref.shape[0]
    weights = np.ones(n, dtype=np.float32)

    mode = (mode or "none").lower().strip()

    if mode == "none":
        return weights

    if mode not in {"gray", "light", "gray_light"}:
        raise ValueError(
            f"Unknown chart weight mode: {mode}. "
            f"Use none / gray / light / gray_light."
        )

    # 4x6 ColorChecker 行优先时，最后一排索引 18~23 通常是灰阶。
    if mode in {"gray", "gray_light"} and n >= 24:
        weights[18:24] = np.maximum(weights[18:24], float(gray_weight))

    if mode in {"light", "gray_light"}:
        lab = rgb_to_lab(ref)
        L = lab[:, 0]
        light_mask = L >= float(light_l_threshold)
        weights[light_mask] = np.maximum(weights[light_mask], float(light_weight))

    if normalize:
        # 归一化到均值 1，避免 ridge_alpha 的相对强度被权重整体缩放影响太大。
        mean_w = float(np.mean(weights))
        if mean_w > 1e-8:
            weights = weights / mean_w

    return weights.astype(np.float32)


def fit_correction_model(
    captured_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    model: str = "linear_bias",
    ridge_alpha: float = 1e-6,
    sample_weights: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """
    在线性 RGB 空间中拟合实拍 RGB 到参考 RGB 的映射关系。

    普通岭回归：
        min ||XW - Y||^2 + ridge * ||W||^2

    加权岭回归：
        min Σ wi * ||XiW - Yi||^2 + ridge * ||W||^2

    sample_weights:
        None 或长度等于色卡 patch 数的权重数组。
        权重大表示该色卡点在拟合中更重要。
    """
    captured_linear = srgb_to_linear(captured_rgb)
    reference_linear = srgb_to_linear(reference_rgb)

    X = build_features(captured_linear, model=model).astype(np.float32)
    Y = reference_linear.reshape(-1, 3).astype(np.float32)

    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if len(weights) != X.shape[0]:
            raise ValueError(
                f"sample_weights length mismatch: "
                f"got {len(weights)}, expected {X.shape[0]}"
            )
        weights = np.maximum(weights, 1e-8)
        # 归一化到均值 1，保持 ridge_alpha 可比。
        weights = weights / np.mean(weights)
        sw = np.sqrt(weights)[:, None].astype(np.float32)
        X_fit = X * sw
        Y_fit = Y * sw
    else:
        X_fit = X
        Y_fit = Y

    if ridge_alpha > 0:
        reg = float(ridge_alpha) * np.eye(X_fit.shape[1], dtype=np.float32)
        # 最后一列通常是 bias，不惩罚 bias。
        reg[-1, -1] = 0.0
        W = np.linalg.solve(X_fit.T @ X_fit + reg, X_fit.T @ Y_fit)
    else:
        W, _, _, _ = np.linalg.lstsq(X_fit, Y_fit, rcond=None)

    return W.astype(np.float32)


def apply_correction_to_image(
    img_bgr: np.ndarray,
    W: np.ndarray,
    model: str = "linear_bias",
) -> np.ndarray:
    """
    将颜色校正模型应用到整张 BGR 图像。
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_linear = srgb_to_linear(img_rgb)

    h, w = img_linear.shape[:2]
    X = build_features(img_linear, model=model)
    corrected_linear = (X @ W).reshape(h, w, 3)
    corrected_linear = np.clip(corrected_linear, 0.0, 1.0)

    corrected_rgb = linear_to_srgb(corrected_linear)
    return cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
