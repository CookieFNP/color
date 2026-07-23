# -*- coding: utf-8 -*-
"""
用途：
    在 visual_alpha_sweep_eval_series_rules.py 已经生成的整张 preview 图基础上，
    按 targets_alpha_1.00.csv 的 code 顺序和 visual_circles_manual.json 的圆形区域，
    对指定胶块区域直接做“最终图渲染”。

解决的问题：
    你原来的命令已经生成了一张：
        preview_alpha_1.00.png

    现在不想只看 CSV 色块预览，而是想直接在这张最终胶块图上继续微调：
        - 大部分胶块整体更深一点
        - 大部分胶块整体稍微偏红一点
        - 可选整体饱和度微调
        - 指定编号不动，保留原 preview 图上的调色结果

默认保护编号：
    W177 W178 W179 W180
    W193 W194 W195 W196
    W225 W226

典型用法：
    python render_zhengwei2_final_tweak.py ^
      --preview output_zhengwei2/corrected_residual_alpha_sweep/preview_alpha_1.00.png ^
      --targets-csv output_zhengwei2/corrected_residual_alpha_sweep/targets_alpha_1.00.csv ^
      --circle-file output_zhengwei2/corrected_residual_alpha_sweep/visual_circles_manual.json ^
      --out output_zhengwei2/corrected_residual_alpha_sweep/preview_alpha_1.00_final_tweak.png ^
      --global-l-offset -2.0 ^
      --global-a-offset 0.8 ^
      --global-chroma-scale 1.00

注意：
    这是“直接改最终整图”的脚本。
    它不会重新跑 ColorChecker，也不会重新计算 residual。
    它是在已有 preview_alpha_1.00.png 的胶块圆形区域上叠加最后一层视觉渲染。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# =========================
# sRGB <-> Lab，D65
# =========================
D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb_lin):
    rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
    srgb = np.where(
        rgb_lin <= 0.0031308,
        12.92 * rgb_lin,
        1.055 * np.power(rgb_lin, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb * 255.0, 0, 255)


def rgb_to_lab(rgb):
    """
    rgb: (..., 3), uint8/float, sRGB 0~255
    return: (..., 3), Lab
    """
    rgb_lin = srgb_to_linear(rgb)
    xyz = rgb_lin @ SRGB_TO_XYZ.T
    xyz_n = xyz / D65

    eps = 216 / 24389
    kappa = 24389 / 27

    f = np.where(xyz_n > eps, np.cbrt(xyz_n), (kappa * xyz_n + 16) / 116)

    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab):
    """
    lab: (..., 3)
    return: (..., 3), sRGB 0~255 float
    """
    lab = np.asarray(lab, dtype=np.float64)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    eps = 216 / 24389
    kappa = 24389 / 27

    def finv(t):
        t3 = t ** 3
        return np.where(t3 > eps, t3, (116 * t - 16) / kappa)

    x = D65[0] * finv(fx)
    y = D65[1] * finv(fy)
    z = D65[2] * finv(fz)

    xyz = np.stack([x, y, z], axis=-1)
    rgb_lin = xyz @ XYZ_TO_SRGB.T
    return linear_to_srgb(rgb_lin)


# =========================
# circle 文件读取
# =========================
def normalize_circle(obj):
    """
    尽量兼容不同版本保存的 circle 格式。
    支持：
        {"cx":..., "cy":..., "r":...}
        {"x":..., "y":..., "r":...}
        {"center_x":..., "center_y":..., "radius":...}
        {"circle":[cx,cy,r]}
        [cx,cy,r]
    """
    if obj is None:
        return None

    if isinstance(obj, (list, tuple)) and len(obj) >= 3:
        return float(obj[0]), float(obj[1]), float(obj[2])

    if isinstance(obj, dict):
        if all(k in obj for k in ["cx", "cy", "r"]):
            return float(obj["cx"]), float(obj["cy"]), float(obj["r"])

        if all(k in obj for k in ["x", "y", "r"]):
            return float(obj["x"]), float(obj["y"]), float(obj["r"])

        if all(k in obj for k in ["center_x", "center_y", "radius"]):
            return float(obj["center_x"]), float(obj["center_y"]), float(obj["radius"])

        if "circle" in obj:
            return normalize_circle(obj["circle"])

        if "roi_circle" in obj:
            return normalize_circle(obj["roi_circle"])

    return None


def load_circles(circle_file, targets_df):
    """
    返回长度与 targets_df 一致的 circles 列表。
    如果找不到某个 circle，返回 None，后面会退回 ROI ellipse。
    """
    if not circle_file or not Path(circle_file).exists():
        return [None] * len(targets_df)

    data = json.loads(Path(circle_file).read_text(encoding="utf-8"))

    # 有些文件最外层可能是 {"circles":[...]}
    if isinstance(data, dict) and "circles" in data:
        data = data["circles"]

    circles = []

    # 情况 1：list，默认按 targets 顺序对应
    if isinstance(data, list):
        for i in range(len(targets_df)):
            if i < len(data):
                circles.append(normalize_circle(data[i]))
            else:
                circles.append(None)
        return circles

    # 情况 2：dict，优先按 code 找，其次按 index 找
    if isinstance(data, dict):
        for _, row in targets_df.iterrows():
            code = str(row.get("code", row.get("编号", ""))).strip()
            idx = str(row.get("index", "")).strip()

            obj = None
            for key in [code, code.upper(), idx, str(int(float(idx))) if idx else ""]:
                if key and key in data:
                    obj = data[key]
                    break

            circles.append(normalize_circle(obj))
        return circles

    return [None] * len(targets_df)


def fallback_circle_from_roi(row):
    """
    如果 circle 文件中没有该项，就用 targets_csv 里的 roi 生成一个近似圆。
    """
    x1 = float(row["roi_x1"])
    y1 = float(row["roi_y1"])
    x2 = float(row["roi_x2"])
    y2 = float(row["roi_y2"])
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    r = min(x2 - x1, y2 - y1) * 0.45
    return cx, cy, r


# =========================
# 渲染
# =========================
def adjust_lab_pixels(lab, args):
    """
    对一个局部 Lab 图块做：
        L += global_l_offset
        chroma *= global_chroma_scale
        a += global_a_offset
        b += global_b_offset
    """
    out = lab.copy()

    out[..., 0] = np.clip(out[..., 0] + float(args.global_l_offset), 0, 100)

    a = out[..., 1]
    b = out[..., 2]

    out[..., 1] = a * float(args.global_chroma_scale) + float(args.global_a_offset)
    out[..., 2] = b * float(args.global_chroma_scale) + float(args.global_b_offset)

    out[..., 1] = np.clip(out[..., 1], -128, 127)
    out[..., 2] = np.clip(out[..., 2], -128, 127)

    return out


def circle_alpha_mask(h, w, cx, cy, r, feather):
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if feather <= 0:
        return (dist <= r).astype(np.float64)

    # r-feather 内部完全生效，r 附近羽化到 0
    alpha = (r - dist) / max(float(feather), 1e-6)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def render_one_circle(img_rgb, cx, cy, r, args):
    H, W = img_rgb.shape[:2]
    feather = float(args.feather)

    x1 = max(0, int(math.floor(cx - r - feather - 2)))
    y1 = max(0, int(math.floor(cy - r - feather - 2)))
    x2 = min(W, int(math.ceil(cx + r + feather + 2)))
    y2 = min(H, int(math.ceil(cy + r + feather + 2)))

    if x2 <= x1 or y2 <= y1:
        return img_rgb

    patch = img_rgb[y1:y2, x1:x2, :].astype(np.float64)
    local_cx = cx - x1
    local_cy = cy - y1

    alpha = circle_alpha_mask(y2 - y1, x2 - x1, local_cx, local_cy, r, feather)
    if alpha.max() <= 0:
        return img_rgb

    lab = rgb_to_lab(patch)
    lab2 = adjust_lab_pixels(lab, args)
    rgb2 = lab_to_rgb(lab2)

    alpha3 = alpha[..., None]
    mixed = patch * (1 - alpha3) + rgb2 * alpha3

    out = img_rgb.copy()
    out[y1:y2, x1:x2, :] = np.clip(mixed, 0, 255)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="在已有 preview_alpha_1.00.png 上按 circle 区域直接做最终渲染。"
    )

    ap.add_argument("--preview", required=True, help="已有的 preview_alpha_1.00.png。")
    ap.add_argument("--targets-csv", required=True, help="同一轮输出的 targets_alpha_1.00.csv。")
    ap.add_argument("--circle-file", default="", help="visual_circles_manual.json。可不填，不填则退回使用 roi 椭圆。")
    ap.add_argument("--out", required=True, help="输出最终渲染图路径。")
    ap.add_argument("--out-csv", default="", help="可选，输出记录每个 code 是否被保护/使用的 circle。")

    ap.add_argument("--global-l-offset", type=float, default=-2.0, help="整体 L 偏移，负数更深。默认 -2。")
    ap.add_argument("--global-a-offset", type=float, default=0.8, help="整体 a 偏移，正数偏红。默认 +0.8。")
    ap.add_argument("--global-b-offset", type=float, default=0.0, help="整体 b 偏移，正数偏黄暖，负数偏蓝冷。默认 0。")
    ap.add_argument("--global-chroma-scale", type=float, default=1.00, help="整体色度倍率，>1 更饱和。默认 1。")
    ap.add_argument("--feather", type=float, default=6.0, help="圆边缘羽化像素。默认 6。")

    ap.add_argument(
        "--protect-codes",
        default="W177,W178,W179,W180,W193,W194,W195,W196,W225,W226",
        help="不做这层额外渲染的编号，逗号分隔。",
    )

    args = ap.parse_args()

    preview_path = Path(args.preview)
    targets_path = Path(args.targets_csv)
    circle_path = Path(args.circle_file) if args.circle_file else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(targets_path, encoding="utf-8-sig")

    if "code" not in df.columns:
        raise RuntimeError("targets_csv 里找不到 code 列。")

    img = Image.open(preview_path).convert("RGB")
    img_rgb = np.asarray(img, dtype=np.float64)

    protect_codes = {
        x.strip().upper()
        for x in str(args.protect_codes).split(",")
        if x.strip()
    }

    circles = load_circles(circle_path, df)

    records = []
    for i, row in df.iterrows():
        code = str(row["code"]).strip().upper()
        name = str(row.get("name", "")).strip()

        protected = code in protect_codes

        c = circles[i] if i < len(circles) else None
        source = "circle_file"
        if c is None:
            c = fallback_circle_from_roi(row)
            source = "roi_fallback"

        cx, cy, r = c

        if not protected:
            img_rgb = render_one_circle(img_rgb, cx, cy, r, args)

        rec = dict(row)
        rec.update(
            {
                "protect_keep": bool(protected),
                "render_circle_cx": float(cx),
                "render_circle_cy": float(cy),
                "render_circle_r": float(r),
                "circle_source": source,
                "applied_global_l_offset": 0.0 if protected else float(args.global_l_offset),
                "applied_global_a_offset": 0.0 if protected else float(args.global_a_offset),
                "applied_global_b_offset": 0.0 if protected else float(args.global_b_offset),
                "applied_global_chroma_scale": 1.0 if protected else float(args.global_chroma_scale),
            }
        )
        records.append(rec)

    out_img = Image.fromarray(np.clip(img_rgb, 0, 255).astype(np.uint8))
    out_img.save(out_path)

    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = out_path.with_suffix(".render_log.csv")

    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = {
        "preview": str(preview_path),
        "targets_csv": str(targets_path),
        "circle_file": str(circle_path) if circle_path else "",
        "out": str(out_path),
        "out_csv": str(out_csv),
        "protect_codes": sorted(protect_codes),
        "protect_count": int(sum(str(r["code"]).strip().upper() in protect_codes for _, r in df.iterrows())),
        "tweak_count": int(len(df) - sum(str(r["code"]).strip().upper() in protect_codes for _, r in df.iterrows())),
        "params": {
            "global_l_offset": args.global_l_offset,
            "global_a_offset": args.global_a_offset,
            "global_b_offset": args.global_b_offset,
            "global_chroma_scale": args.global_chroma_scale,
            "feather": args.feather,
        },
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Done ===")
    print("input preview:", preview_path)
    print("targets csv:", targets_path)
    print("circle file:", circle_path)
    print("output image:", out_path)
    print("render log:", out_csv)
    print("summary:", summary_path)
    print("protect count:", summary["protect_count"])
    print("tweak count:", summary["tweak_count"])


if __name__ == "__main__":
    main()
