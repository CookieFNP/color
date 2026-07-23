# -*- coding: utf-8 -*-
"""
用途：
    在已经生成好的 preview_alpha_1.00.png 上，
    根据 targets_alpha_1.00.csv + visual_circles_manual.json，
    对不同胶块按“分组规则”做最终二次渲染。

特点：
    1. 直接改最终整图，不是只看 CSV 预览
    2. 每组胶块有自己的一套：
         - global_l_offset
         - global_a_offset
         - global_b_offset
         - global_chroma_scale
    3. 如果一个 code 同时出现在多个组里：
         按“更靠前的规则优先”
    4. 不在任何规则里的胶块：保持原图不动

输入：
    --preview      已有最终图 preview_alpha_1.00.png
    --targets-csv  同轮输出的 targets_alpha_1.00.csv
    --circle-file  visual_circles_manual.json
    --out          输出最终图

示例：
    python render_zhengwei2_group_rules_final.py ^
      --preview output_zhengwei2/corrected_residual_alpha_sweep/preview_alpha_1.00.png ^
      --targets-csv output_zhengwei2/corrected_residual_alpha_sweep/targets_alpha_1.00.csv ^
      --circle-file output_zhengwei2/corrected_residual_alpha_sweep/visual_circles_manual.json ^
      --out output_zhengwei2/corrected_residual_alpha_sweep/preview_alpha_1.00_group_final.png
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
# 颜色空间：sRGB <-> Lab
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
# 你的最终规则（按“更靠前优先”）
# =========================
RULES = [
    # 规则 A：深棕 / 更深更浓
    {
        "name": "rule_A_deep_brown",
        "codes": {
            "W151", "W152", "W153", "W154", "W155",
            "W203", "W204",
            "W217", "W218", "W219", "W220", "W221",
            "T10", "T114",
        },
        "l_offset": -7.0,
        "a_offset": 0.8,
        "b_offset": 0.0,
        "chroma_scale": 1.2,
    },

    # 规则 B：默认主组 + T系列剩下全部
    {
        "name": "rule_B_main_default",
        "codes": {
            "W157", "W158", "W159", "W173", "W174", "W175", "W176",
            "W166", "W167", "W178", "W179", "W180", "W181", "W182",
            "W196", "W197", "W198", "W199", "W200", "W201", "W202",
            "W206", "W207", "W208", "W209", "W210", "W212", "W213",
            "W214", "W215", "W216", "W225", "W226", "W227",
        },
        "apply_to_other_T_series": True,  # 除了前面已经匹配到的 T07/T08/T09/T10/T114，剩下 T 全部走这组
        "l_offset": -1.2,
        "a_offset": 0.8,
        "b_offset": 0.0,
        "chroma_scale": 1.0,
    },

    # 规则 C：提亮浅色组
    {
        "name": "rule_C_light_group",
        "codes": {"W129", "W130", "W131", "W132", "W160"},
        "l_offset": 11.8,
        "a_offset": 0.8,
        "b_offset": 0.0,
        "chroma_scale": 0.9,
    },

    # 规则 D：蓝色系
    {
        "name": "rule_D_blue_group",
        "codes": {"W168", "W169", "W170", "W171", "W172", "W173", "W174", "W175", "W176"},
        "l_offset": -4.0,
        "a_offset": 2.8,
        "b_offset": 0.0,
        "chroma_scale": 1.1,
    },

    # 规则 E：偏灰偏淡组
    {
        "name": "rule_E_soft_group",
        "codes": {"W183", "W184", "W185", "W186", "W201", "W202", "W203", "W204", "W205", "W216", "W217", "W218", "W219", "W220"},
        "l_offset": -1.0,
        "a_offset": 0.8,
        "b_offset": 0.0,
        "chroma_scale": 0.9,
    },

    # 规则 F：局部提亮组
    {
        "name": "rule_F_brighten_group",
        "codes": {"W181","W182","W187", "W189", "W190", "W191", "W192", "W206"},
        "l_offset": 6.0,
        "a_offset": 0.8,
        "b_offset": 0.0,
        "chroma_scale": 0.9,
    },
]


# =========================
# circle 文件读取
# =========================
def normalize_circle(obj):
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
    if not circle_file or not Path(circle_file).exists():
        return [None] * len(targets_df)

    data = json.loads(Path(circle_file).read_text(encoding="utf-8"))

    if isinstance(data, dict) and "circles" in data:
        data = data["circles"]

    circles = []

    if isinstance(data, list):
        for i in range(len(targets_df)):
            circles.append(normalize_circle(data[i]) if i < len(data) else None)
        return circles

    if isinstance(data, dict):
        for _, row in targets_df.iterrows():
            code = str(row.get("code", "")).strip()
            idx = str(row.get("index", "")).strip()
            obj = None
            for key in [code, code.upper(), idx]:
                if key and key in data:
                    obj = data[key]
                    break
            circles.append(normalize_circle(obj))
        return circles

    return [None] * len(targets_df)


def fallback_circle_from_roi(row):
    x1 = float(row["roi_x1"])
    y1 = float(row["roi_y1"])
    x2 = float(row["roi_x2"])
    y2 = float(row["roi_y2"])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    r = min(x2 - x1, y2 - y1) * 0.45
    return cx, cy, r


# =========================
# 规则匹配：前面的优先
# =========================
def match_rule_for_code(code: str):
    code = str(code).strip().upper()

    matched_names = []
    chosen_rule = None

    for rule in RULES:
        hit = False

        if code in rule.get("codes", set()):
            hit = True

        # T系列剩下全部
        if (not hit) and rule.get("apply_to_other_T_series", False):
            if code.startswith("T"):
                hit = True

        if hit:
            matched_names.append(rule["name"])
            if chosen_rule is None:
                chosen_rule = rule  # 前面的优先
            # 不 break，继续记冲突信息，但最终还是第一个生效

    return chosen_rule, matched_names


# =========================
# 渲染
# =========================
def adjust_lab_pixels(lab, l_offset, a_offset, b_offset, chroma_scale):
    out = lab.copy()

    out[..., 0] = np.clip(out[..., 0] + float(l_offset), 0, 100)
    out[..., 1] = out[..., 1] * float(chroma_scale) + float(a_offset)
    out[..., 2] = out[..., 2] * float(chroma_scale) + float(b_offset)

    out[..., 1] = np.clip(out[..., 1], -128, 127)
    out[..., 2] = np.clip(out[..., 2], -128, 127)

    return out


def circle_alpha_mask(h, w, cx, cy, r, feather):
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if feather <= 0:
        return (dist <= r).astype(np.float64)

    alpha = (r - dist) / max(float(feather), 1e-6)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def render_one_circle(img_rgb, cx, cy, r, feather, l_offset, a_offset, b_offset, chroma_scale):
    H, W = img_rgb.shape[:2]

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
    lab2 = adjust_lab_pixels(lab, l_offset, a_offset, b_offset, chroma_scale)
    rgb2 = lab_to_rgb(lab2)

    alpha3 = alpha[..., None]
    mixed = patch * (1 - alpha3) + rgb2 * alpha3

    out = img_rgb.copy()
    out[y1:y2, x1:x2, :] = np.clip(mixed, 0, 255)
    return out


def main():
    parser = argparse.ArgumentParser(description="按分组规则直接渲染最终胶块整图。")
    parser.add_argument("--preview", required=True, help="已有 preview_alpha_1.00.png")
    parser.add_argument("--targets-csv", required=True, help="targets_alpha_1.00.csv")
    parser.add_argument("--circle-file", required=True, help="visual_circles_manual.json")
    parser.add_argument("--out", required=True, help="输出最终图片")
    parser.add_argument("--feather", type=float, default=6.0, help="圆边缘羽化，默认 6")
    parser.add_argument("--out-log-csv", default="", help="输出渲染日志 csv，不填则自动生成")
    args = parser.parse_args()

    preview_path = Path(args.preview)
    targets_path = Path(args.targets_csv)
    circle_path = Path(args.circle_file)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(targets_path, encoding="utf-8-sig")
    if "code" not in df.columns:
        raise RuntimeError("targets_alpha_1.00.csv 中找不到 code 列。")

    img = Image.open(preview_path).convert("RGB")
    img_rgb = np.asarray(img, dtype=np.float64)

    circles = load_circles(circle_path, df)

    records = []
    for i, row in df.iterrows():
        code = str(row["code"]).strip().upper()
        name = str(row.get("name", "")).strip()

        rule, matched_names = match_rule_for_code(code)

        c = circles[i] if i < len(circles) else None
        circle_source = "circle_file"
        if c is None:
            c = fallback_circle_from_roi(row)
            circle_source = "roi_fallback"

        cx, cy, r = c

        applied = False
        if rule is not None:
            img_rgb = render_one_circle(
                img_rgb=img_rgb,
                cx=cx,
                cy=cy,
                r=r,
                feather=float(args.feather),
                l_offset=rule["l_offset"],
                a_offset=rule["a_offset"],
                b_offset=rule["b_offset"],
                chroma_scale=rule["chroma_scale"],
            )
            applied = True

        rec = dict(row)
        rec.update(
            {
                "render_circle_cx": float(cx),
                "render_circle_cy": float(cy),
                "render_circle_r": float(r),
                "circle_source": circle_source,
                "matched_rule_names": "|".join(matched_names),
                "applied_rule_name": rule["name"] if rule else "",
                "applied": applied,
                "applied_l_offset": rule["l_offset"] if rule else "",
                "applied_a_offset": rule["a_offset"] if rule else "",
                "applied_b_offset": rule["b_offset"] if rule else "",
                "applied_chroma_scale": rule["chroma_scale"] if rule else "",
            }
        )
        records.append(rec)

    out_img = Image.fromarray(np.clip(img_rgb, 0, 255).astype(np.uint8))
    out_img.save(out_path)

    if args.out_log_csv:
        log_csv = Path(args.out_log_csv)
    else:
        log_csv = out_path.with_suffix(".render_log.csv")
    pd.DataFrame(records).to_csv(log_csv, index=False, encoding="utf-8-sig")

    summary = {
        "preview": str(preview_path),
        "targets_csv": str(targets_path),
        "circle_file": str(circle_path),
        "out": str(out_path),
        "log_csv": str(log_csv),
        "rule_order_priority": [r["name"] for r in RULES],
        "note": "重复编号按更靠前规则优先；不在任何规则里的 code 不做处理。",
        "feather": args.feather,
        "total_targets": int(len(df)),
        "applied_count": int(sum(1 for x in records if x["applied"])),
        "not_applied_count": int(sum(1 for x in records if not x["applied"])),
    }

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Done ===")
    print("output image:", out_path)
    print("render log :", log_csv)
    print("summary    :", summary_path)
    print("applied    :", summary["applied_count"])
    print("not applied:", summary["not_applied_count"])


if __name__ == "__main__":
    main()