# ColorChecker 色卡提取、矫正透视

from __future__ import annotations

import cv2
import numpy as np


def warp_chart_from_photo(photo_bgr: np.ndarray, corners: np.ndarray, output_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    # 色卡透视纠正
    dst_w, dst_h = output_size
    src = np.asarray(corners, dtype=np.float32)
    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(photo_bgr, matrix, (dst_w, dst_h))
    return warped, matrix


def extract_chart_means(chart_bgr: np.ndarray, rows: int = 4, cols: int = 6, center_ratio: float = 0.50) -> np.ndarray:
    # 从对齐后的 4×6 ColorChecker 色卡图像中提取 24 个 RGB 均值
    h, w = chart_bgr.shape[:2]
    cell_w = w / cols
    cell_h = h / rows
    margin = (1.0 - center_ratio) / 2.0

    means_rgb: list[np.ndarray] = []

    for r in range(rows):
        for c in range(cols):
            x1 = int((c + margin) * cell_w)
            x2 = int((c + 1 - margin) * cell_w)
            y1 = int((r + margin) * cell_h)
            y2 = int((r + 1 - margin) * cell_h)

            patch = chart_bgr[y1:y2, x1:x2]
            if patch.size == 0:
                raise RuntimeError(f"Patch extraction failed at row={r + 1}, col={c + 1}.")

            mean_bgr = patch.reshape(-1, 3).mean(axis=0)
            means_rgb.append(mean_bgr[::-1])

    return np.asarray(means_rgb, dtype=np.float32)
