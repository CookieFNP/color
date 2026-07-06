# 颜色空间转换、色差计算

from __future__ import annotations

import numpy as np

D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)


def srgb_to_linear(rgb_255: np.ndarray) -> np.ndarray:
    # 将 sRGB（0~255）转换为线性 RGB（0~1）
    rgb = np.asarray(rgb_255, dtype=np.float32) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
    # 将线性 RGB（0~1）转换为 sRGB（0~255）
    linear_rgb = np.clip(np.asarray(linear_rgb, dtype=np.float32), 0.0, 1.0)
    srgb = np.where(
        linear_rgb <= 0.0031308,
        linear_rgb * 12.92,
        1.055 * (linear_rgb ** (1 / 2.4)) - 0.055,
    )
    return np.clip(srgb * 255.0, 0, 255).astype(np.uint8)


def rgb_to_xyz(rgb_255: np.ndarray) -> np.ndarray:
    # 将 sRGB（0~255）转换为 D65 白点下的 CIE XYZ
    rgb_linear = srgb_to_linear(rgb_255)
    return rgb_linear @ SRGB_TO_XYZ.T


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    # 将 CIE XYZ 转换为 D65 白点下的 CIE Lab
    xyz_scaled = np.asarray(xyz, dtype=np.float32) / D65_WHITE

    epsilon = 216 / 24389
    kappa = 24389 / 27

    f = np.where(
        xyz_scaled > epsilon,
        np.cbrt(xyz_scaled),
        (kappa * xyz_scaled + 16) / 116,
    )

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb_255: np.ndarray) -> np.ndarray:
    # 将 sRGB（0~255）转换为 CIE Lab
    return xyz_to_lab(rgb_to_xyz(rgb_255))


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    # 计算 CIEDE2000 色差
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    kL = kC = kH = 1.0

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2

    G = 0.5 * (1 - np.sqrt((C_bar**7) / (C_bar**7 + 25**7 + 1e-12)))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360

    h1_prime = np.where(C1_prime == 0, 0, h1_prime)
    h2_prime = np.where(C2_prime == 0, 0, h2_prime)

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    delta_h = h2_prime - h1_prime
    delta_h_prime = np.where(
        C1_prime * C2_prime == 0,
        0,
        np.where(delta_h > 180, delta_h - 360, np.where(delta_h < -180, delta_h + 360, delta_h)),
    )

    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2))

    L_bar_prime = (L1 + L2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2

    h_sum = h1_prime + h2_prime
    h_diff = np.abs(h1_prime - h2_prime)

    h_bar_prime = np.where(
        C1_prime * C2_prime == 0,
        h_sum,
        np.where(h_diff <= 180, h_sum / 2, np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2)),
    )

    T = (
        1
        - 0.17 * np.cos(np.radians(h_bar_prime - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3 * h_bar_prime + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar_prime - 63))
    )

    delta_theta = 30 * np.exp(-(((h_bar_prime - 275) / 25) ** 2))
    R_C = 2 * np.sqrt((C_bar_prime**7) / (C_bar_prime**7 + 25**7 + 1e-12))

    S_L = 1 + (0.015 * ((L_bar_prime - 50) ** 2)) / np.sqrt(20 + ((L_bar_prime - 50) ** 2))
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T

    R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

    return np.sqrt(
        (delta_L_prime / (kL * S_L)) ** 2
        + (delta_C_prime / (kC * S_C)) ** 2
        + (delta_H_prime / (kH * S_H)) ** 2
        + R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H))
    )
