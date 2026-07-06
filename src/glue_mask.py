# 目标胶块分割、采样
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _clip_roi_to_image(img_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(1, min(w, int(x2)))
    y2 = max(1, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("目标 ROI 无效，请重新框选。")
    return x1, y1, x2, y2


def _center_ellipse_mask(h: int, w: int, scale_x: float = 0.34, scale_y: float = 0.34) -> np.ndarray:
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
    # 当 GrabCut 或高光过滤后保留像素过少时，程序会回退到 ROI 中心椭圆区
    x1, y1, x2, y2 = _clip_roi_to_image(img_bgr, roi)
    crop = img_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise RuntimeError("目标 ROI 为空，请检查坐标。")

    h, w = crop.shape[:2]
    min_pixels = max(12, int(h * w * 0.015))

    fallback_mask = _center_ellipse_mask(h, w)

    try:
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        border = max(2, int(min(h, w) * 0.05))
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        ellipse_mask = _center_ellipse_mask(h, w, scale_x=0.42, scale_y=0.42)
        mask[ellipse_mask == 255] = cv2.GC_PR_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)

        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        kernel_size = 3 if min(h, w) < 80 else 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        if np.sum(fg_mask > 0) < min_pixels:
            fg_mask = fallback_mask.copy()

        dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
        if dist.max() > 0:
            inner_mask = (dist > dist.max() * 0.10).astype(np.uint8) * 255
        else:
            inner_mask = fg_mask.copy()

        if np.sum(inner_mask > 0) < min_pixels:
            inner_mask = fg_mask.copy()

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        L = lab[:, :, 0].astype(np.float32)
        S = hsv[:, :, 1].astype(np.float32)
        V = hsv[:, :, 2].astype(np.float32)

        valid_pixels = inner_mask > 0
        if np.sum(valid_pixels) < min_pixels:
            final_mask = fallback_mask.copy()
        else:
            L_vals = L[valid_pixels]
            S_vals = S[valid_pixels]
            V_vals = V[valid_pixels]

            # 去除极端高光和阴影区
            L_low = np.percentile(L_vals, 8)
            L_high = np.percentile(L_vals, 96)
            V_high = np.percentile(V_vals, 97)
            S_low = np.percentile(S_vals, 1)

            stable_mask = (
                (inner_mask > 0)
                & (L >= L_low)
                & (L <= L_high)
                & (V <= V_high)
                & (S >= S_low)
            )
            final_mask = stable_mask.astype(np.uint8) * 255

        if np.sum(final_mask > 0) < min_pixels:
            final_mask = fallback_mask.copy()

    except cv2.error:
        final_mask = fallback_mask.copy()

    if debug_path is not None:
        debug_path = Path(debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug = crop.copy()
        overlay = debug.copy()
        overlay[final_mask > 0] = [0, 0, 255]
        debug = cv2.addWeighted(debug, 0.65, overlay, 0.35, 0)
        cv2.imwrite(str(debug_path), debug)

    return final_mask


def robust_rgb_from_mask(
    img_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    mask: np.ndarray,
    trim_percent: float = 10.0,
) -> np.ndarray:
    x1, y1, x2, y2 = _clip_roi_to_image(img_bgr, roi)
    crop = img_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise RuntimeError("目标 ROI 为空，请检查坐标。")

    h, w = crop.shape[:2]
    valid = mask > 0
    if valid.shape != (h, w):
        valid = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    pixels_bgr = crop[valid]
    if len(pixels_bgr) < max(8, int(h * w * 0.01)):
        fallback = _center_ellipse_mask(h, w, scale_x=0.32, scale_y=0.32) > 0
        pixels_bgr = crop[fallback]

    if len(pixels_bgr) == 0:
        pixels_bgr = crop.reshape(-1, 3)

    pixels_rgb = pixels_bgr[:, ::-1].astype(np.float32)
    trim_percent = float(np.clip(trim_percent, 0.0, 40.0))

    result = []
    for ch in range(3):
        vals = pixels_rgb[:, ch]
        if len(vals) >= 10 and trim_percent > 0:
            low = np.percentile(vals, trim_percent)
            high = np.percentile(vals, 100.0 - trim_percent)
            kept = vals[(vals >= low) & (vals <= high)]
            if len(kept) > 0:
                vals = kept
        result.append(float(np.mean(vals)))

    return np.asarray(result, dtype=np.float32)


def get_glue_block_representative_rgb(
    img_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    mask: np.ndarray | None = None,
    trim_percent: float = 10.0,
) -> np.ndarray:
    """Build mask if needed, then return robust representative RGB."""
    if mask is None:
        mask = build_glue_block_mask(img_bgr, roi)
    return robust_rgb_from_mask(img_bgr, roi, mask, trim_percent=trim_percent)


def draw_roi_and_mask(
    img_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
    mask: np.ndarray,
    text: str,
) -> np.ndarray:
    """Draw ROI rectangle and actual stable sampling mask on image."""
    x1, y1, x2, y2 = _clip_roi_to_image(img_bgr, roi)
    out = img_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    crop = out[y1:y2, x1:x2]
    overlay = crop.copy()
    if mask.shape != crop.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay[mask > 0] = [0, 0, 255]
    crop = cv2.addWeighted(crop, 0.70, overlay, 0.30, 0)
    out[y1:y2, x1:x2] = crop

    cv2.putText(
        out,
        text,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return out
