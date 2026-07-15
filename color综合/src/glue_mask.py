from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import imwrite_unicode


def _clip_roi(img_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, roi)
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"无效 ROI：{roi}")
    return x1, y1, x2, y2


def _center_ellipse_mask(h: int, w: int, scale_x: float = 0.36, scale_y: float = 0.36) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (max(2, int(w * scale_x)), max(2, int(h * scale_y)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def build_glue_block_mask(
    img_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    debug_path: str | Path | None = None,
) -> np.ndarray:
    x1, y1, x2, y2 = _clip_roi(img_bgr, roi)
    crop = img_bgr[y1:y2, x1:x2].copy()
    h, w = crop.shape[:2]
    min_pixels = max(20, int(h * w * 0.02))

    fallback = _center_ellipse_mask(h, w)

    try:
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        border = max(2, int(min(h, w) * 0.06))
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD
        mask[_center_ellipse_mask(h, w, 0.45, 0.45) == 255] = cv2.GC_PR_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        kernel = np.ones((3 if min(h, w) < 90 else 5, 3 if min(h, w) < 90 else 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        if int((fg > 0).sum()) < min_pixels:
            fg = fallback
    except Exception:
        fg = fallback

    # 只取内部，避开边缘胶线/背景
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    if dist.max() > 0:
        inner = (dist > dist.max() * 0.12).astype(np.uint8) * 255
        if int((inner > 0).sum()) >= min_pixels:
            fg = inner

    # 过滤高光/阴影：只在 mask 内取 L/V 的中间区域
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    inside = fg > 0

    if inside.sum() >= min_pixels:
        L_vals = L[inside]
        V_vals = V[inside]
        L_lo, L_hi = np.percentile(L_vals, [8, 92])
        V_lo, V_hi = np.percentile(V_vals, [5, 96])
        keep = inside & (L >= L_lo) & (L <= L_hi) & (V >= V_lo) & (V <= V_hi)
        if keep.sum() >= min_pixels:
            fg = keep.astype(np.uint8) * 255

    if debug_path is not None:
        debug = crop.copy()
        overlay = debug.copy()
        overlay[fg > 0] = (0, 0, 255)
        debug = cv2.addWeighted(debug, 0.65, overlay, 0.35, 0)
        imwrite_unicode(debug_path, debug)

    return fg


def get_glue_block_representative_rgb(
    img_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    mask: np.ndarray | None = None,
    trim_percent: float = 10.0,
) -> np.ndarray:
    x1, y1, x2, y2 = _clip_roi(img_bgr, roi)
    crop = img_bgr[y1:y2, x1:x2]
    if mask is None:
        mask = np.ones(crop.shape[:2], dtype=np.uint8) * 255

    rgb = crop[:, :, ::-1][mask > 0].astype(np.float64)
    if rgb.shape[0] == 0:
        raise RuntimeError("ROI mask 内没有有效像素")

    if trim_percent > 0 and rgb.shape[0] >= 20:
        lo = np.percentile(rgb, trim_percent, axis=0)
        hi = np.percentile(rgb, 100 - trim_percent, axis=0)
        keep = np.all((rgb >= lo) & (rgb <= hi), axis=1)
        if keep.sum() >= max(10, rgb.shape[0] * 0.2):
            rgb = rgb[keep]

    return rgb.mean(axis=0)


def draw_roi_and_mask(img_bgr: np.ndarray, roi: tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = _clip_roi(img_bgr, roi)
    out = img_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    crop = out[y1:y2, x1:x2]
    overlay = crop.copy()
    overlay[mask > 0] = (0, 0, 255)
    out[y1:y2, x1:x2] = cv2.addWeighted(crop, 0.68, overlay, 0.32, 0)
    return out
