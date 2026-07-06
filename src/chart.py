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


def _representative_bgr(patch_bgr: np.ndarray, method: str = "mean", trim_percent: float = 10.0) -> np.ndarray:
    pixels = patch_bgr.reshape(-1, 3).astype(np.float32)
    method = (method or "mean").strip().lower()

    if method == "mean":
        return pixels.mean(axis=0)

    if method == "median":
        return np.median(pixels, axis=0)

    if method in {"trimmed_mean", "trim", "trimmed"}:
        trim_percent = float(np.clip(trim_percent, 0.0, 40.0))
        if len(pixels) < 10 or trim_percent <= 0:
            return pixels.mean(axis=0)

        result = []
        for ch in range(3):
            vals = pixels[:, ch]
            low = np.percentile(vals, trim_percent)
            high = np.percentile(vals, 100.0 - trim_percent)
            kept = vals[(vals >= low) & (vals <= high)]
            result.append(float(kept.mean() if len(kept) else vals.mean()))
        return np.asarray(result, dtype=np.float32)

    raise ValueError("sample_method must be one of: mean, median, trimmed_mean")


def extract_chart_means(
    chart_bgr: np.ndarray,
    rows: int = 4,
    cols: int = 6,
    center_ratio: float = 0.50,
    sample_method: str = "mean",
    trim_percent: float = 10.0,
) -> np.ndarray:
    # 从对齐后的 4×6 ColorChecker 色卡图像中提取 24 个 RGB 代表值
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

            bgr = _representative_bgr(patch, method=sample_method, trim_percent=trim_percent)
            means_rgb.append(bgr[::-1])

    return np.asarray(means_rgb, dtype=np.float32)
