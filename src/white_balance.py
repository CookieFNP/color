# 白平衡预处理：用于验证“强光/阴影场景下，白平衡是否能降低偏色”。

from __future__ import annotations

import cv2
import numpy as np

from .color_math import linear_to_srgb, srgb_to_linear


GRAY_PATCH_SLICE = slice(18, 24)  # ColorChecker 24 色卡最后一行通常为 6 个灰阶块


def _safe_gains(gains_rgb: np.ndarray, max_gain: float = 4.0) -> np.ndarray:
    """限制增益，避免某个通道因为异常像素被放大到离谱。"""
    gains = np.asarray(gains_rgb, dtype=np.float32).reshape(3)
    gains = np.nan_to_num(gains, nan=1.0, posinf=max_gain, neginf=1.0)
    return np.clip(gains, 1.0 / max_gain, max_gain)


def apply_rgb_gains_bgr(img_bgr: np.ndarray, gains_rgb: np.ndarray) -> np.ndarray:
    """
    在线性 RGB 空间给整张 BGR 图像乘 RGB 三通道白平衡增益。

    参数：
        img_bgr: OpenCV 读入的 BGR 图像，uint8
        gains_rgb: RGB 顺序的三通道增益，例如 [R_gain, G_gain, B_gain]
    """
    gains = _safe_gains(gains_rgb)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_linear = srgb_to_linear(img_rgb)
    balanced_linear = img_linear * gains.reshape(1, 1, 3)
    balanced_rgb = linear_to_srgb(balanced_linear)

    return cv2.cvtColor(balanced_rgb, cv2.COLOR_RGB2BGR)


def gray_world_gains_from_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    灰度世界白平衡。

    假设图像整体平均颜色应接近灰色，因此让线性 RGB 三通道均值相等。
    会自动忽略接近黑/白的极端像素，减少过曝/死黑区域影响。
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_linear = srgb_to_linear(img_rgb)

    # 忽略过暗和过曝像素，避免强光白斑或黑色背景主导均值
    valid = (img_rgb.max(axis=2) < 250) & (img_rgb.min(axis=2) > 5)
    if np.count_nonzero(valid) < 100:
        valid = np.ones(img_rgb.shape[:2], dtype=bool)

    mean_rgb = img_linear[valid].reshape(-1, 3).mean(axis=0)
    gray = float(np.mean(mean_rgb))
    gains = gray / np.maximum(mean_rgb, 1e-6)
    return _safe_gains(gains)


def chart_gray_gains_from_samples(captured_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
    """
    基于 ColorChecker 灰阶块的白平衡。

    使用最后一行 6 个灰阶块估计光源/白平衡偏差，使 captured 灰阶均值靠近 reference 灰阶均值。
    这比普通灰度世界更适合“图中有标准色卡”的测色流程。
    """
    captured_rgb = np.asarray(captured_rgb, dtype=np.float32).reshape(-1, 3)
    reference_rgb = np.asarray(reference_rgb, dtype=np.float32).reshape(-1, 3)

    if len(captured_rgb) < 24 or len(reference_rgb) < 24:
        raise ValueError("chart_gray white balance needs 24 ColorChecker samples.")

    cap_gray_linear = srgb_to_linear(captured_rgb[GRAY_PATCH_SLICE])
    ref_gray_linear = srgb_to_linear(reference_rgb[GRAY_PATCH_SLICE])

    cap_mean = cap_gray_linear.mean(axis=0)
    ref_mean = ref_gray_linear.mean(axis=0)

    gains = ref_mean / np.maximum(cap_mean, 1e-6)
    return _safe_gains(gains)


def gains_to_report(gains_rgb: np.ndarray) -> dict:
    gains = _safe_gains(gains_rgb)
    return {
        "R_gain": float(gains[0]),
        "G_gain": float(gains[1]),
        "B_gain": float(gains[2]),
    }
