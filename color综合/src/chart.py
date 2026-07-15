from __future__ import annotations

import cv2
import numpy as np


def warp_chart_from_photo(photo_bgr: np.ndarray, corners: np.ndarray, output_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    dst_w, dst_h = output_size
    src = np.asarray(corners, dtype=np.float32)
    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(photo_bgr, matrix, (dst_w, dst_h))
    return warped, matrix


def _trimmed_mean_rgb(patch_bgr: np.ndarray, trim_percent: float = 10.0) -> np.ndarray:
    rgb = patch_bgr.reshape(-1, 3)[:, ::-1].astype(np.float64)
    if rgb.shape[0] == 0:
        raise RuntimeError("空色块")

    if trim_percent <= 0:
        return rgb.mean(axis=0)

    lo = np.percentile(rgb, trim_percent, axis=0)
    hi = np.percentile(rgb, 100 - trim_percent, axis=0)
    keep = np.all((rgb >= lo) & (rgb <= hi), axis=1)
    if keep.sum() < max(10, rgb.shape[0] * 0.2):
        return rgb.mean(axis=0)
    return rgb[keep].mean(axis=0)


def extract_chart_means(
    chart_bgr: np.ndarray,
    rows: int = 4,
    cols: int = 6,
    center_ratio: float = 0.50,
    trim_percent: float = 10.0,
) -> np.ndarray:
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
                raise RuntimeError(f"ColorChecker 色块提取失败 row={r+1}, col={c+1}")
            means_rgb.append(_trimmed_mean_rgb(patch, trim_percent=trim_percent))

    return np.asarray(means_rgb, dtype=np.float64)
