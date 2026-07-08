from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from visual_preview_from_report import (
    read_json_safe,
    get_photo_path,
    imread_unicode,
    imwrite_unicode,
    bgr_to_lab_image,
    lab_to_bgr_image,
    build_background_mask,
    apply_background_neutralization,
    apply_target_visual_correction,
)


def parse_float_list(text: str) -> list[float]:
    """
    把 "0.3,0.5,0.7" 转成 [0.3, 0.5, 0.7]
    """
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def put_label(img: np.ndarray, label: str) -> np.ndarray:
    """
    在图片顶部加参数标签。
    """
    h, w = img.shape[:2]
    top = 48

    canvas = np.full((h + top, w, 3), 255, dtype=np.uint8)
    canvas[top:top + h, 0:w] = img

    cv2.putText(
        canvas,
        label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return canvas


def resize_keep_ratio(img: np.ndarray, target_width: int) -> np.ndarray:
    """
    等比例缩放到指定宽度。
    """
    h, w = img.shape[:2]
    scale = target_width / float(w)
    new_h = int(h * scale)
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)


def make_contact_sheet(
    images: list[np.ndarray],
    labels: list[str],
    cols: int = 3,
    thumb_width: int = 420,
    gap: int = 18,
) -> np.ndarray:
    """
    把多张候选图拼成一张总览图。
    """
    thumbs = []

    for img, label in zip(images, labels):
        thumb = resize_keep_ratio(img, thumb_width)
        thumb = put_label(thumb, label)
        thumbs.append(thumb)

    if not thumbs:
        raise ValueError("没有可拼接的候选图")

    thumb_h = max(t.shape[0] for t in thumbs)
    thumb_w = max(t.shape[1] for t in thumbs)

    rows = int(np.ceil(len(thumbs) / cols))

    sheet_h = rows * thumb_h + (rows + 1) * gap
    sheet_w = cols * thumb_w + (cols + 1) * gap

    sheet = np.full((sheet_h, sheet_w, 3), 245, dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        r = idx // cols
        c = idx % cols

        y = gap + r * (thumb_h + gap)
        x = gap + c * (thumb_w + gap)

        h, w = thumb.shape[:2]
        sheet[y:y + h, x:x + w] = thumb

    return sheet


def make_one_visual_candidate(
    bgr: np.ndarray,
    target_colors: list[dict],
    glue_ab_alpha: float,
    bg_alpha: float,
    l_alpha: float,
    bg_min_L: float,
    bg_max_saturation: float,
    feather: int,
    mask_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    生成一张视觉候选图。

    核心逻辑：
    1. 背景：保留 L，a/b 向 0 靠近
    2. 胶块：L 少动，a/b 向标准色靠近
    3. 高光/阴影区域自动降低修正强度
    """
    lab_before = bgr_to_lab_image(bgr)

    bg_mask = build_background_mask(
        bgr=bgr,
        lab=lab_before,
        target_colors=target_colors,
        bg_min_L=bg_min_L,
        bg_max_saturation=bg_max_saturation,
        exclude_targets=True,
    )

    lab_bg = apply_background_neutralization(
        lab=lab_before,
        bg_mask=bg_mask,
        strength=bg_alpha,
    )

    lab_preview = apply_target_visual_correction(
        bgr=bgr,
        lab=lab_bg,
        target_colors=target_colors,
        ab_strength=glue_ab_alpha,
        l_strength=l_alpha,
        feather=feather,
        mask_mode=mask_mode,
        protect_extreme_light=True,
    )

    preview_bgr = lab_to_bgr_image(lab_preview)

    info = {
        "glue_ab_alpha": glue_ab_alpha,
        "bg_alpha": bg_alpha,
        "l_alpha": l_alpha,
        "bg_min_L": bg_min_L,
        "bg_max_saturation": bg_max_saturation,
        "feather": feather,
        "mask_mode": mask_mode,
    }

    return preview_bgr, bg_mask, info


def process_report_grid(
    report_path: Path,
    root: Path,
    photo_override: Optional[str],
    glue_ab_list: list[float],
    bg_alpha_list: list[float],
    l_alpha: float,
    bg_min_L: float,
    bg_max_saturation: float,
    feather: int,
    mask_mode: str,
    thumb_width: int,
) -> None:
    report = read_json_safe(report_path)

    if report is None:
        raise RuntimeError(f"report 无效：{report_path}")

    photo_path = get_photo_path(
        report=report,
        report_path=report_path,
        root=root,
        photo_override=photo_override,
    )

    bgr = imread_unicode(photo_path)

    if bgr is None:
        raise RuntimeError(f"图片读取失败：{photo_path}")

    target_colors = report.get("target_colors") or []

    if not target_colors:
        raise RuntimeError(f"{report_path} 里没有 target_colors")

    out_dir = report_path.parent / "visual_alpha_grid"
    out_dir.mkdir(parents=True, exist_ok=True)

    images_for_sheet = []
    labels_for_sheet = []
    records = []

    idx = 1

    for bg_alpha in bg_alpha_list:
        for glue_ab_alpha in glue_ab_list:
            preview_bgr, bg_mask, info = make_one_visual_candidate(
                bgr=bgr,
                target_colors=target_colors,
                glue_ab_alpha=glue_ab_alpha,
                bg_alpha=bg_alpha,
                l_alpha=l_alpha,
                bg_min_L=bg_min_L,
                bg_max_saturation=bg_max_saturation,
                feather=feather,
                mask_mode=mask_mode,
            )

            name = (
                f"v{idx:02d}"
                f"_glue{glue_ab_alpha:.2f}"
                f"_bg{bg_alpha:.2f}"
                f"_L{l_alpha:.2f}"
                ".png"
            )

            out_path = out_dir / name

            ok = imwrite_unicode(out_path, preview_bgr)

            if not ok:
                raise RuntimeError(f"保存失败：{out_path}")

            label = f"V{idx:02d}  glue={glue_ab_alpha:.2f}  bg={bg_alpha:.2f}  L={l_alpha:.2f}"

            images_for_sheet.append(preview_bgr)
            labels_for_sheet.append(label)

            record = {
                "index": idx,
                "label": label,
                "file": str(out_path),
                "params": info,
            }

            records.append(record)

            print("生成：", label)
            print("文件：", out_path)

            idx += 1

    mask_path = out_dir / "background_mask.png"
    imwrite_unicode(mask_path, (bg_mask * 255).astype(np.uint8))

    sheet = make_contact_sheet(
        images=images_for_sheet,
        labels=labels_for_sheet,
        cols=len(glue_ab_list),
        thumb_width=thumb_width,
        gap=18,
    )

    sheet_path = out_dir / "contact_sheet.png"
    imwrite_unicode(sheet_path, sheet)

    info_path = out_dir / "candidates_info.json"

    info = {
        "report": str(report_path),
        "photo": str(photo_path),
        "output_dir": str(out_dir),
        "contact_sheet": str(sheet_path),
        "background_mask": str(mask_path),
        "meaning": {
            "glue_ab_alpha": "胶块颜色向标准色 a/b 靠近的强度，越大越接近标准色，但越可能不像现场光。",
            "bg_alpha": "背景 a/b 向中性白灰靠近的强度，越大背景越不偏黄/偏蓝，但越可能丢失现场氛围。",
            "l_alpha": "胶块 L 亮度向标准 L 靠近的强度，建议小，一般 0~0.1。",
        },
        "candidates": records,
    }

    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n全部候选图生成完成")
    print("候选图目录：", out_dir)
    print("总览图：", sheet_path)
    print("参数记录：", info_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=".",
        help="项目根目录，默认当前目录",
    )

    parser.add_argument(
        "--report",
        required=True,
        help="report.json 路径，例如 output_root_poly2/report.json",
    )

    parser.add_argument(
        "--photo",
        default=None,
        help="手动指定原图路径。路径找不到时使用。",
    )

    parser.add_argument(
        "--glue-ab-list",
        default="0.30,0.50,0.70",
        help="胶块 a/b 修正强度列表，默认 0.30,0.50,0.70",
    )

    parser.add_argument(
        "--bg-list",
        default="0.20,0.40,0.60",
        help="背景 a/b 中性化强度列表，默认 0.20,0.40,0.60",
    )

    parser.add_argument(
        "--l-alpha",
        type=float,
        default=0.05,
        help="胶块 L 修正强度，默认 0.05。想完全保留明暗可设为 0",
    )

    parser.add_argument(
        "--bg-min-L",
        type=float,
        default=50.0,
        help="背景候选区域最低 L，默认 50",
    )

    parser.add_argument(
        "--bg-max-saturation",
        type=float,
        default=80.0,
        help="背景候选区域最高 HSV 饱和度，默认 80",
    )

    parser.add_argument(
        "--feather",
        type=int,
        default=25,
        help="胶块 ROI 边缘羽化大小，默认 25",
    )

    parser.add_argument(
        "--mask-mode",
        choices=["rectangle", "ellipse"],
        default="rectangle",
        help="胶块视觉修正 mask 形状，默认 rectangle",
    )

    parser.add_argument(
        "--thumb-width",
        type=int,
        default=420,
        help="总览图中每张候选图的宽度，默认 420",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    report_path = Path(args.report).resolve()

    glue_ab_list = parse_float_list(args.glue_ab_list)
    bg_alpha_list = parse_float_list(args.bg_list)

    process_report_grid(
        report_path=report_path,
        root=root,
        photo_override=args.photo,
        glue_ab_list=glue_ab_list,
        bg_alpha_list=bg_alpha_list,
        l_alpha=args.l_alpha,
        bg_min_L=args.bg_min_L,
        bg_max_saturation=args.bg_max_saturation,
        feather=args.feather,
        mask_mode=args.mask_mode,
        thumb_width=args.thumb_width,
    )


if __name__ == "__main__":
    main()