from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.color_math import rgb_to_lab, delta_e_2000
from src.io_utils import imread_unicode, imwrite_unicode


def safe_name(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r'[\\/:*?"<>|\s]+', "_", text)
    return text.strip("_") or "unknown"


def resolve_path(path_text: str | None, base_dir: Path) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text)
    if p.exists():
        return p
    p2 = base_dir / p
    if p2.exists():
        return p2
    p3 = Path.cwd() / p
    if p3.exists():
        return p3
    return p


def stat_pack(x: list[float] | np.ndarray) -> dict:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "p95": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def parse_report_targets(report: dict) -> list[dict]:
    targets = report.get("target_colors") or []
    if not targets:
        raise RuntimeError("report.json 里没有 target_colors，无法建立胶块视觉库。")
    return targets


def get_target_code_name(target: dict) -> tuple[str, str]:
    standard = target.get("standard") or {}
    code = str(standard.get("code") or target.get("code") or target.get("input_label") or target.get("index") or "").strip()
    name = str(standard.get("name") or target.get("name") or "").strip()
    return code, name


def load_visual_circles(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_key: dict[str, dict] = {}
    for row in rows:
        code = row.get("code")
        index = row.get("index")
        if code:
            by_key[str(code).strip().upper()] = row
        if index is not None:
            by_key[str(index)] = row
    return by_key


def fallback_circle_from_roi(target: dict) -> dict:
    roi = target.get("roi_xyxy")
    if not roi:
        raise RuntimeError(f"目标没有 roi_xyxy，无法生成 fallback circle：{target}")
    x1, y1, x2, y2 = map(int, roi)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r = int(min(x2 - x1, y2 - y1) * 0.45)
    return {"cx": cx, "cy": cy, "r": max(3, r)}


def get_circle_for_target(target: dict, circles_by_key: dict[str, dict]) -> dict:
    code, _ = get_target_code_name(target)
    index = target.get("index")
    row = None
    if code and code.upper() in circles_by_key:
        row = circles_by_key[code.upper()]
    elif index is not None and str(index) in circles_by_key:
        row = circles_by_key[str(index)]
    if row and row.get("visual_circle"):
        c = row["visual_circle"]
        return {"cx": int(c["cx"]), "cy": int(c["cy"]), "r": int(c["r"])}
    return fallback_circle_from_roi(target)


def load_rule_rows(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    by_key: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get("code") or row.get("input_code")
            index = row.get("index")
            if code:
                by_key[str(code).strip().upper()] = row
            if index is not None and str(index).strip():
                by_key[str(index).strip()] = row
    return by_key


def get_rule_info_for_target(target: dict, rule_by_key: dict[str, dict]) -> dict:
    code, _ = get_target_code_name(target)
    index = target.get("index")
    if code and code.upper() in rule_by_key:
        return dict(rule_by_key[code.upper()])
    if index is not None and str(index) in rule_by_key:
        return dict(rule_by_key[str(index)])
    return {}


def make_circle_mask_for_crop(
    crop_h: int,
    crop_w: int,
    cx_in_crop: int,
    cy_in_crop: int,
    radius: int,
    radius_scale: float = 1.0,
    feather: int = 0,
) -> np.ndarray:
    r = max(1.0, float(radius) * float(radius_scale))
    yy, xx = np.mgrid[0:crop_h, 0:crop_w]
    dist = np.sqrt((xx - float(cx_in_crop)) ** 2 + (yy - float(cy_in_crop)) ** 2)
    mask = (dist <= r).astype(np.float32)
    if feather > 0:
        k = max(3, int(feather) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)


def circle_bbox(cx: int, cy: int, r: int, image_shape: tuple[int, int, int], padding: int = 0) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    rr = int(max(1, r) + max(0, padding))
    x1 = max(0, cx - rr)
    y1 = max(0, cy - rr)
    x2 = min(w, cx + rr + 1)
    y2 = min(h, cy + rr + 1)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"无效圆形裁剪框：cx={cx}, cy={cy}, r={r}")
    return x1, y1, x2, y2


def representative_rgb_from_masked_crop(crop_bgr: np.ndarray, mask: np.ndarray, trim_percent: float) -> np.ndarray:
    if mask.shape[:2] != crop_bgr.shape[:2]:
        raise ValueError("mask 与 crop 尺寸不一致")
    pixels_rgb = crop_bgr[:, :, ::-1][mask > 0.5].astype(np.float64)
    if pixels_rgb.size == 0:
        raise RuntimeError("mask 内没有有效像素")
    if trim_percent > 0 and pixels_rgb.shape[0] >= 20:
        lo = np.percentile(pixels_rgb, trim_percent, axis=0)
        hi = np.percentile(pixels_rgb, 100.0 - trim_percent, axis=0)
        keep = np.all((pixels_rgb >= lo) & (pixels_rgb <= hi), axis=1)
        if keep.sum() >= max(10, pixels_rgb.shape[0] * 0.2):
            pixels_rgb = pixels_rgb[keep]
    return pixels_rgb.mean(axis=0)


def rgb_to_lab_1(rgb: np.ndarray) -> np.ndarray:
    return rgb_to_lab(np.asarray(rgb, dtype=np.float64).reshape(1, 3))[0]


def de2000_1(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return float(delta_e_2000(np.asarray(lab1).reshape(1, 3), np.asarray(lab2).reshape(1, 3))[0])


def save_visual_crop(
    *,
    preview_bgr: np.ndarray,
    circle: dict,
    out_path: Path,
    crop_padding: int,
    transparent: bool,
    alpha_feather: int,
) -> None:
    cx, cy, r = int(circle["cx"]), int(circle["cy"]), int(circle["r"])
    x1, y1, x2, y2 = circle_bbox(cx, cy, r, preview_bgr.shape, padding=crop_padding)
    crop = preview_bgr[y1:y2, x1:x2].copy()
    if transparent:
        alpha = make_circle_mask_for_crop(
            crop_h=crop.shape[0],
            crop_w=crop.shape[1],
            cx_in_crop=cx - x1,
            cy_in_crop=cy - y1,
            radius=r,
            radius_scale=1.0,
            feather=alpha_feather,
        )
        alpha_u8 = np.clip(np.round(alpha * 255), 0, 255).astype(np.uint8)
        crop_bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        crop_bgra[:, :, 3] = alpha_u8
        imwrite_unicode(out_path, crop_bgra)
    else:
        imwrite_unicode(out_path, crop)


def target_lab_from_report(target: dict, key: str) -> np.ndarray | None:
    v = target.get(key)
    if v is None:
        return None
    return np.asarray(v, dtype=np.float64)


def standard_lab_from_target(target: dict) -> np.ndarray:
    standard = target.get("standard") or {}
    lab = standard.get("lab")
    if lab is None:
        raise RuntimeError(f"目标缺少 standard.lab：{target}")
    return np.asarray(lab, dtype=np.float64)


def build_library(
    *,
    report_path: Path,
    preview_path: Path,
    circle_file: Path,
    rules_file: Path | None,
    out_dir: Path,
    version: str,
    alpha: float,
    rule_strength: float,
    background_mode: str,
    bg_scale: float,
    trim_percent: float,
    sample_radius_scale: float,
    crop_padding: int,
    transparent_crops: bool,
    crop_alpha_feather: int,
) -> tuple[list[dict], dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    targets = parse_report_targets(report)
    preview_bgr = imread_unicode(preview_path)
    circles_by_key = load_visual_circles(circle_file)
    rule_by_key = load_rule_rows(rules_file)
    crops_dir = out_dir / "visual_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    de_corrected_list: list[float] = []
    de_visual_list: list[float] = []

    for target in targets:
        code, name = get_target_code_name(target)
        index = target.get("index")
        standard_lab = standard_lab_from_target(target)
        before_lab = target_lab_from_report(target, "before_lab")
        corrected_lab = target_lab_from_report(target, "after_lab")
        if corrected_lab is None:
            corrected_lab = target_lab_from_report(target, "corrected_lab")
        if corrected_lab is None:
            raise RuntimeError(f"{code} 缺少 after_lab / corrected_lab，无法写 corrected_lab。")

        circle = get_circle_for_target(target, circles_by_key)
        cx, cy, r = int(circle["cx"]), int(circle["cy"]), int(circle["r"])
        x1, y1, x2, y2 = circle_bbox(cx, cy, r, preview_bgr.shape, padding=0)
        crop_for_sample = preview_bgr[y1:y2, x1:x2]
        sample_mask = make_circle_mask_for_crop(
            crop_h=crop_for_sample.shape[0],
            crop_w=crop_for_sample.shape[1],
            cx_in_crop=cx - x1,
            cy_in_crop=cy - y1,
            radius=r,
            radius_scale=sample_radius_scale,
            feather=0,
        )
        visual_rgb = representative_rgb_from_masked_crop(crop_for_sample, sample_mask, trim_percent=trim_percent)
        visual_lab = rgb_to_lab_1(visual_rgb)
        de_corrected = de2000_1(corrected_lab, standard_lab)
        de_visual_to_standard = de2000_1(visual_lab, standard_lab)
        de_corrected_list.append(de_corrected)
        de_visual_list.append(de_visual_to_standard)

        crop_filename = f"{safe_name(code)}_{safe_name(name)}.png" if name else f"{safe_name(code)}.png"
        crop_path = crops_dir / crop_filename
        save_visual_crop(
            preview_bgr=preview_bgr,
            circle=circle,
            out_path=crop_path,
            crop_padding=crop_padding,
            transparent=transparent_crops,
            alpha_feather=crop_alpha_feather,
        )
        rule_info = get_rule_info_for_target(target, rule_by_key)
        delta = target.get("delta_e_2000_to_standard") or {}

        row = {
            "index": index,
            "code": code,
            "name": name,
            "machine_L": float(standard_lab[0]),
            "machine_a": float(standard_lab[1]),
            "machine_b": float(standard_lab[2]),
            "standard_L": float(standard_lab[0]),
            "standard_a": float(standard_lab[1]),
            "standard_b": float(standard_lab[2]),
            "before_L": float(before_lab[0]) if before_lab is not None else None,
            "before_a": float(before_lab[1]) if before_lab is not None else None,
            "before_b": float(before_lab[2]) if before_lab is not None else None,
            "corrected_L": float(corrected_lab[0]),
            "corrected_a": float(corrected_lab[1]),
            "corrected_b": float(corrected_lab[2]),
            "visual_display_L": float(visual_lab[0]),
            "visual_display_a": float(visual_lab[1]),
            "visual_display_b": float(visual_lab[2]),
            "visual_display_R": float(visual_rgb[0]),
            "visual_display_G": float(visual_rgb[1]),
            "visual_display_B": float(visual_rgb[2]),
            "deltaE_report_before_to_standard": delta.get("before"),
            "deltaE_report_corrected_to_standard": delta.get("after"),
            "deltaE_corrected_to_standard_recomputed": de_corrected,
            "deltaE_visual_to_standard": de_visual_to_standard,
            "visual_circle_cx": cx,
            "visual_circle_cy": cy,
            "visual_circle_r": r,
            "sample_radius_scale": sample_radius_scale,
            "visual_crop_path": str(crop_path.as_posix()),
            "rule_info_json": json.dumps(rule_info, ensure_ascii=False),
            "alpha": alpha,
            "rule_strength": rule_strength,
            "background_mode": background_mode,
            "bg_scale": bg_scale,
            "version": version,
        }

        # 常见规则字段平铺出来，方便 Excel 查看。不存在就跳过。
        for k in ["rule_group", "group", "b_scale", "b_pos_cap", "b_neg_cap", "l_original_mix", "ab_scale", "l_scale"]:
            if k in rule_info:
                out_key = k if k.startswith("rule_") else f"rule_{k}"
                row[out_key] = rule_info.get(k)
        rows.append(row)

    summary = {
        "version": version,
        "report": str(report_path),
        "preview": str(preview_path),
        "circle_file": str(circle_file),
        "rules_file": str(rules_file) if rules_file else None,
        "out_dir": str(out_dir),
        "count": len(rows),
        "params": {
            "alpha": alpha,
            "rule_strength": rule_strength,
            "background_mode": background_mode,
            "bg_scale": bg_scale,
            "trim_percent": trim_percent,
            "sample_radius_scale": sample_radius_scale,
            "crop_padding": crop_padding,
            "transparent_crops": transparent_crops,
            "crop_alpha_feather": crop_alpha_feather,
        },
        "deltaE_corrected_to_standard": stat_pack(de_corrected_list),
        "deltaE_visual_to_standard": stat_pack(de_visual_list),
        "note": "本库是已知 128 胶块的视觉参考库，不是未知样本通用校正算法。后续板材匹配应优先比较 board_visual_lab 与 visual_display_lab。",
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "index", "code", "name",
        "machine_L", "machine_a", "machine_b",
        "standard_L", "standard_a", "standard_b",
        "before_L", "before_a", "before_b",
        "corrected_L", "corrected_a", "corrected_b",
        "visual_display_L", "visual_display_a", "visual_display_b",
        "visual_display_R", "visual_display_G", "visual_display_B",
        "deltaE_report_before_to_standard", "deltaE_report_corrected_to_standard",
        "deltaE_corrected_to_standard_recomputed", "deltaE_visual_to_standard",
        "visual_circle_cx", "visual_circle_cy", "visual_circle_r", "sample_radius_scale",
        "visual_crop_path",
        "rule_group", "rule_b_scale", "rule_b_pos_cap", "rule_b_neg_cap", "rule_l_original_mix",
        "alpha", "rule_strength", "background_mode", "bg_scale", "version", "rule_info_json",
    ]
    keys: list[str] = []
    seen = set()
    for k in preferred:
        if any(k in row for row in rows) and k not in seen:
            keys.append(k)
            seen.add(k)
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(path: Path, summary: dict) -> None:
    text = f"""# Glue Visual Library

本目录由 `build_glue_visual_library.py` 自动生成。

## 定位

这是已知 128 胶块的视觉参考库，不是未知样本通用颜色校正算法。

胶块端由于 W001~W128 的编号和标准 Lab 已知，因此可以基于 v0.7 视觉流程生成每个胶块的 `visual_display_lab` 和 `visual_crop_path`。

后续板材匹配应优先比较：

```text
board_visual_lab  vs  glue_visual_library.visual_display_lab
```

## 版本

```text
version = {summary.get('version')}
alpha = {summary.get('params', {}).get('alpha')}
rule_strength = {summary.get('params', {}).get('rule_strength')}
background_mode = {summary.get('params', {}).get('background_mode')}
bg_scale = {summary.get('params', {}).get('bg_scale')}
```

## 主要文件

```text
glue_visual_library.csv
    128 个胶块视觉库主表。

glue_visual_library.json
    与 CSV 同内容的 JSON 版本。

visual_crops/
    每个胶块的视觉小图。

preview_best_rule07.png
    整版 v0.7 视觉结果图备份。
```

## 字段说明

```text
machine_L/a/b
    data.csv/report.json 中的标准 Lab，作为机采/标准参考值。

corrected_L/a/b
    ColorChecker 基础校正后，从 report.json 中得到的胶块 Lab。

visual_display_L/a/b
    v0.7 最终视觉图中该胶块实际呈现出来的 Lab。
    后续板材 TopK 匹配建议使用该值。

visual_crop_path
    该胶块对应的视觉小图路径。

rule_info_json
    该胶块在 v0.7 中使用的分组视觉规则信息。
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build glue_visual_library.csv from v0.7 visual preview image.")
    parser.add_argument("--report", required=True, help="main.py 输出的 report.json，例如 output_128/report.json")
    parser.add_argument("--preview", default=None, help="v0.7 最终视觉图。默认读取 corrected_residual_alpha_sweep/preview_alpha_1.00.png")
    parser.add_argument("--circle-file", default=None, help="手动画圆文件。默认读取 corrected_residual_alpha_sweep/visual_circles_manual.json")
    parser.add_argument("--rules-file", default=None, help="视觉规则 CSV。默认读取 corrected_residual_alpha_sweep/visual_rules_alpha_1.00.csv；不存在则跳过。")
    parser.add_argument("--out-dir", default=None, help="输出目录。默认 output_128/glue_visual_library")
    parser.add_argument("--version", default="v0.7")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rule-strength", type=float, default=0.7)
    parser.add_argument("--background-mode", default="original")
    parser.add_argument("--bg-scale", type=float, default=0.0)
    parser.add_argument("--sample-radius-scale", type=float, default=0.72, help="提取 visual_display_lab 时使用圆心半径比例，默认 0.72，避开边缘羽化和背景。")
    parser.add_argument("--trim-percent", type=float, default=10.0, help="取代表色时的 trimmed mean 百分比，默认 10。")
    parser.add_argument("--crop-padding", type=int, default=8, help="输出小图时在圆外额外保留的像素。")
    parser.add_argument("--transparent-crops", action="store_true", help="输出透明背景的圆形 PNG crop。")
    parser.add_argument("--crop-alpha-feather", type=int, default=9, help="透明 crop 的 alpha 羽化。")
    args = parser.parse_args()

    report_path = Path(args.report)
    report_dir = report_path.parent
    preview_path = resolve_path(args.preview, report_dir) or (report_dir / "corrected_residual_alpha_sweep" / "preview_alpha_1.00.png")
    circle_file = resolve_path(args.circle_file, report_dir) or (report_dir / "corrected_residual_alpha_sweep" / "visual_circles_manual.json")
    rules_file = resolve_path(args.rules_file, report_dir)
    if rules_file is None:
        default_rules = report_dir / "corrected_residual_alpha_sweep" / "visual_rules_alpha_1.00.csv"
        rules_file = default_rules if default_rules.exists() else None
    out_dir = resolve_path(args.out_dir, report_dir) or (report_dir / "glue_visual_library")

    if not report_path.exists():
        raise FileNotFoundError(f"找不到 report.json：{report_path}")
    if not preview_path.exists():
        raise FileNotFoundError(f"找不到 v0.7 预览图：{preview_path}")
    if not circle_file.exists():
        raise FileNotFoundError(f"找不到手动画圆文件：{circle_file}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = build_library(
        report_path=report_path,
        preview_path=preview_path,
        circle_file=circle_file,
        rules_file=rules_file,
        out_dir=out_dir,
        version=args.version,
        alpha=args.alpha,
        rule_strength=args.rule_strength,
        background_mode=args.background_mode,
        bg_scale=args.bg_scale,
        trim_percent=args.trim_percent,
        sample_radius_scale=args.sample_radius_scale,
        crop_padding=args.crop_padding,
        transparent_crops=args.transparent_crops,
        crop_alpha_feather=args.crop_alpha_feather,
    )
    csv_path = out_dir / "glue_visual_library.csv"
    json_path = out_dir / "glue_visual_library.json"
    summary_path = out_dir / "library_summary.json"
    readme_path = out_dir / "README.md"
    write_csv(csv_path, rows)
    write_json(json_path, rows)
    write_json(summary_path, summary)
    write_readme(readme_path, summary)
    try:
        shutil.copyfile(preview_path, out_dir / "preview_best_rule07.png")
    except Exception:
        pass

    print("\n==== 胶块视觉库生成完成 ====")
    print("输出目录：", out_dir)
    print("CSV：", csv_path)
    print("JSON：", json_path)
    print("Summary：", summary_path)
    print("Crops：", out_dir / "visual_crops")
    print("\n视觉库数量：", len(rows))
    print("visual_display_lab vs standard_lab ΔE 统计：")
    for k, v in summary["deltaE_visual_to_standard"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
