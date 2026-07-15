from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.io_utils import imread_unicode, imwrite_unicode


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def lab_std_to_cv(lab: list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
    L, a, b = map(float, lab)
    return L * 255.0 / 100.0, a + 128.0, b + 128.0


def feather_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.max() > 1:
        m = m / 255.0
    if feather > 0:
        k = max(3, int(feather) | 1)
        m = cv2.GaussianBlur(m, (k, k), 0)
    return np.clip(m, 0.0, 1.0)


def build_rect_mask(h: int, w: int, feather: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=np.float32)
    return feather_mask(mask, feather)


def protect_light_mask(crop_lab: np.ndarray) -> np.ndarray:
    L = crop_lab[:, :, 0].astype(np.float32)
    # 极暗/极亮少改，中间正常改
    dark = np.clip((L - 18) / 35, 0, 1)
    bright = np.clip((245 - L) / 35, 0, 1)
    return np.clip(dark * bright, 0.15, 1.0)


def build_background_mask(bgr: np.ndarray, target_colors: list[dict], bg_min_L: float, bg_max_saturation: float) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0].astype(np.float32) * 100.0 / 255.0
    S = hsv[:, :, 1].astype(np.float32)

    mask = ((L >= bg_min_L) & (S <= bg_max_saturation)).astype(np.uint8)

    # 排除所有胶块 ROI，避免背景中性化影响胶块
    for item in target_colors:
        roi = item.get("roi_xyxy")
        if not roi:
            continue
        x1, y1, x2, y2 = map(int, roi)
        x1 = max(0, min(mask.shape[1], x1))
        x2 = max(0, min(mask.shape[1], x2))
        y1 = max(0, min(mask.shape[0], y1))
        y2 = max(0, min(mask.shape[0], y2))
        mask[y1:y2, x1:x2] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return feather_mask(mask, 31)


def make_candidate(
    bgr: np.ndarray,
    target_colors: list[dict],
    glue_ab_alpha: float,
    bg_alpha: float,
    l_alpha: float,
    feather: int,
    bg_min_L: float,
    bg_max_saturation: float,
) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 背景：L 保留，a/b 向 128 靠近
    bg_mask = build_background_mask(bgr, target_colors, bg_min_L, bg_max_saturation)
    lab[:, :, 1] = lab[:, :, 1] + bg_mask * bg_alpha * (128.0 - lab[:, :, 1])
    lab[:, :, 2] = lab[:, :, 2] + bg_mask * bg_alpha * (128.0 - lab[:, :, 2])

    # 胶块：L 少动，a/b 向标准色靠近
    for item in target_colors:
        roi = item.get("roi_xyxy")
        standard = item.get("standard") or {}
        std_lab = standard.get("lab")
        if not roi or not std_lab:
            continue

        x1, y1, x2, y2 = map(int, roi)
        x1 = max(0, min(lab.shape[1] - 1, x1))
        x2 = max(1, min(lab.shape[1], x2))
        y1 = max(0, min(lab.shape[0] - 1, y1))
        y2 = max(1, min(lab.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = lab[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        m = build_rect_mask(h, w, feather)
        p = protect_light_mask(crop)
        m = m * p

        target_L, target_a, target_b = lab_std_to_cv(std_lab)
        crop[:, :, 0] = crop[:, :, 0] + m * l_alpha * (target_L - crop[:, :, 0])
        crop[:, :, 1] = crop[:, :, 1] + m * glue_ab_alpha * (target_a - crop[:, :, 1])
        crop[:, :, 2] = crop[:, :, 2] + m * glue_ab_alpha * (target_b - crop[:, :, 2])
        lab[y1:y2, x1:x2] = crop

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    h, w = img.shape[:2]
    top = 48
    out = np.full((h + top, w, 3), 245, dtype=np.uint8)
    out[top:, :] = img
    cv2.putText(out, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def resize_keep(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def make_sheet(images: list[np.ndarray], labels: list[str], cols: int, thumb_width: int) -> np.ndarray:
    thumbs = [put_label(resize_keep(img, thumb_width), label) for img, label in zip(images, labels)]
    gap = 18
    tw = max(t.shape[1] for t in thumbs)
    th = max(t.shape[0] for t in thumbs)
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * th + (rows + 1) * gap, cols * tw + (cols + 1) * gap, 3), 250, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        y = gap + r * (th + gap)
        x = gap + c * (tw + gap)
        sheet[y:y + t.shape[0], x:x + t.shape[1]] = t
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual alpha grid from report.json")
    parser.add_argument("--report", required=True, help="main.py 输出的 report.json")
    parser.add_argument("--photo", default=None, help="原图路径，不填则从 report 里读")
    parser.add_argument("--glue-ab-list", default="0.08,0.15,0.25", help="胶块色度修正强度，默认偏保守，避免过艳")
    parser.add_argument("--bg-list", default="0.15,0.30,0.45", help="背景中性化强度")
    parser.add_argument("--l-alpha", type=float, default=0.0, help="胶块亮度修正，默认 0，避免变深/变假")
    parser.add_argument("--feather", type=int, default=31)
    parser.add_argument("--bg-min-L", type=float, default=45.0)
    parser.add_argument("--bg-max-saturation", type=float, default=85.0)
    parser.add_argument("--thumb-width", type=int, default=430)
    args = parser.parse_args()

    report_path = Path(args.report)
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    photo_path = Path(args.photo) if args.photo else Path(report["input"]["photo"])
    bgr = imread_unicode(photo_path)
    target_colors = report.get("target_colors") or []

    out_dir = report_path.parent / "visual_alpha_grid"
    out_dir.mkdir(parents=True, exist_ok=True)

    glue_list = parse_float_list(args.glue_ab_list)
    bg_list = parse_float_list(args.bg_list)

    images = []
    labels = []
    records = []
    idx = 1

    for bg_alpha in bg_list:
        for glue_ab_alpha in glue_list:
            img = make_candidate(
                bgr=bgr,
                target_colors=target_colors,
                glue_ab_alpha=glue_ab_alpha,
                bg_alpha=bg_alpha,
                l_alpha=args.l_alpha,
                feather=args.feather,
                bg_min_L=args.bg_min_L,
                bg_max_saturation=args.bg_max_saturation,
            )
            filename = f"v{idx:02d}_glue{glue_ab_alpha:.2f}_bg{bg_alpha:.2f}_L{args.l_alpha:.2f}.png"
            path = out_dir / filename
            imwrite_unicode(path, img)
            label = f"V{idx:02d} glue={glue_ab_alpha:.2f} bg={bg_alpha:.2f} L={args.l_alpha:.2f}"
            images.append(img)
            labels.append(label)
            records.append({"index": idx, "file": str(path), "glue_ab_alpha": glue_ab_alpha, "bg_alpha": bg_alpha, "l_alpha": args.l_alpha})
            idx += 1

    sheet = make_sheet(images, labels, cols=len(glue_list), thumb_width=args.thumb_width)
    sheet_path = out_dir / "contact_sheet.png"
    imwrite_unicode(sheet_path, sheet)

    info = {
        "report": str(report_path),
        "photo": str(photo_path),
        "meaning": {
            "glue_ab_alpha": "胶块 a/b 向标准色靠近强度，越大越标准但越可能过艳。",
            "bg_alpha": "背景 a/b 向中性灰靠近强度，越大越干净但越可能丢现场氛围。",
            "l_alpha": "胶块亮度修正，默认 0，避免变深。",
        },
        "candidates": records,
    }
    (out_dir / "candidates_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("生成完成：", sheet_path)


if __name__ == "__main__":
    main()
