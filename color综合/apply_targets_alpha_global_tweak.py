# -*- coding: utf-8 -*-
"""
用途：
    读取 visual_alpha_sweep_eval_series_rules.py 输出的 targets_alpha_1.00.csv，
    对“已经生成的每个胶块视觉 Lab”做后处理微调，并输出新的 CSV 和 before/after 色块预览。

适用输入：
    output_zhengwei2/corrected_residual_alpha_sweep/targets_alpha_1.00.csv

这个 CSV 里常见列名是：
    code, name
    measured_L, measured_a, measured_b
    standard_L, standard_a, standard_b

本脚本默认读取：
    measured_L / measured_a / measured_b

处理目标：
    - 整体偏深一点：global_l_offset < 0
    - 整体偏红一点：global_a_offset > 0
    - 可选整体饱和度更高：global_chroma_scale > 1
    - 指定保护编号不动：
      W177 W178 W179 W180 W193 W194 W195 W196 W225 W226

注意：
    这是 CSV/Lab 后处理和预览脚本。
    它不会直接改已经生成的 preview_alpha_1.00.png。
    如果要把这个效果真正叠回整图，需要在主图渲染脚本里加入同样的 global offset 逻辑。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)


def lab_to_srgb(lab):
    L, a, b = [float(x) for x in lab]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    eps = 216 / 24389
    k = 24389 / 27

    def finv(t):
        t3 = t ** 3
        return t3 if t3 > eps else (116 * t - 16) / k

    xyz = np.array(
        [finv(fx) * D65[0], finv(fy) * D65[1], finv(fz) * D65[2]],
        dtype=np.float64,
    )
    rgb_lin = XYZ_TO_SRGB @ xyz
    rgb_lin = np.clip(rgb_lin, 0, 1)

    rgb = np.where(
        rgb_lin <= 0.0031308,
        12.92 * rgb_lin,
        1.055 * np.power(rgb_lin, 1 / 2.4) - 0.055,
    )
    return tuple(np.clip(rgb * 255, 0, 255).astype(np.uint8).tolist())


def de00(lab1, lab2):
    L1, a1, b1 = [float(x) for x in lab1]
    L2, a2, b2 = [float(x) for x in lab2]

    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    avgC = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(avgC**7 / (avgC**7 + 25**7))) if avgC else 0
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    def hp(ap, bb):
        if ap == 0 and bb == 0:
            return 0.0
        h = math.degrees(math.atan2(bb, ap))
        return h + 360 if h < 0 else h

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0
    else:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        dhp = dh

    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    avgLp = (L1 + L2) / 2
    avgCp = (C1p + C2p) / 2

    if C1p * C2p == 0:
        avghp = h1p + h2p
    else:
        if abs(h1p - h2p) <= 180:
            avghp = (h1p + h2p) / 2
        elif h1p + h2p < 360:
            avghp = (h1p + h2p + 360) / 2
        else:
            avghp = (h1p + h2p - 360) / 2

    T = (
        1
        - 0.17 * math.cos(math.radians(avghp - 30))
        + 0.24 * math.cos(math.radians(2 * avghp))
        + 0.32 * math.cos(math.radians(3 * avghp + 6))
        - 0.20 * math.cos(math.radians(4 * avghp - 63))
    )

    dt = 30 * math.exp(-(((avghp - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(avgCp**7 / (avgCp**7 + 25**7)) if avgCp else 0
    Sl = 1 + (0.015 * ((avgLp - 50) ** 2)) / math.sqrt(20 + (avgLp - 50) ** 2)
    Sc = 1 + 0.045 * avgCp
    Sh = 1 + 0.015 * avgCp * T
    Rt = -math.sin(math.radians(2 * dt)) * Rc

    return float(
        math.sqrt(
            (dLp / Sl) ** 2
            + (dCp / Sc) ** 2
            + (dHp / Sh) ** 2
            + Rt * (dCp / Sc) * (dHp / Sh)
        )
    )


def pick_code_col(df):
    for c in ["code", "编号"]:
        if c in df.columns:
            return c
    raise RuntimeError("找不到编号列，需要 code 或 编号。")


def pick_name_col(df):
    for c in ["name", "名称"]:
        if c in df.columns:
            return c
    raise RuntimeError("找不到名称列，需要 name 或 名称。")


def pick_lab_cols(df, prefix):
    candidates = []
    if prefix:
        candidates.append((f"{prefix}_L", f"{prefix}_a", f"{prefix}_b"))

    candidates += [
        ("measured_L", "measured_a", "measured_b"),
        ("visual_display_L", "visual_display_a", "visual_display_b"),
        ("tweak_L", "tweak_a", "tweak_b"),
        ("fix_L", "fix_a", "fix_b"),
        ("final_L", "final_a", "final_b"),
        ("visual_L", "visual_a", "visual_b"),
        ("corrected_L", "corrected_a", "corrected_b"),
        ("L", "a", "b"),
    ]

    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols

    raise RuntimeError(
        "找不到可用 Lab 列。这个脚本支持 measured_L/a/b、visual_display_L/a/b、"
        "fix_L/a/b、final_L/a/b、corrected_L/a/b 或 L/a/b。"
    )


def adjust_lab(L, a, b, args):
    L2 = float(L) + float(args.global_l_offset)
    a2 = float(a)
    b2 = float(b)

    chroma = math.hypot(a2, b2)
    if chroma > 1e-8:
        ratio = float(args.global_chroma_scale)
        a2 *= ratio
        b2 *= ratio

    a2 += float(args.global_a_offset)
    b2 += float(args.global_b_offset)

    return (
        float(np.clip(L2, 0, 100)),
        float(np.clip(a2, -128, 127)),
        float(np.clip(b2, -128, 127)),
    )


def draw_board(df, out, lab_cols, code_col, name_col, title):
    Lc, ac, bc = lab_cols
    cols = 8
    pw = 112
    ph = 72
    lh = 38
    gap = 8
    rows = math.ceil(len(df) / cols)

    img = Image.new(
        "RGB",
        (cols * pw + (cols + 1) * gap, rows * (ph + lh) + (rows + 1) * gap + 36),
        (245, 245, 245),
    )
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("simhei.ttf", 13)
        tfont = ImageFont.truetype("simhei.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        tfont = ImageFont.load_default()

    d.text((gap, 8), title, fill=(0, 0, 0), font=tfont)

    for i, (_, r) in enumerate(df.iterrows()):
        rr = i // cols
        cc = i % cols
        x = gap + cc * (pw + gap)
        y = 36 + gap + rr * (ph + lh + gap)

        rgb = lab_to_srgb((r[Lc], r[ac], r[bc]))
        d.rectangle([x, y, x + pw, y + ph], fill=rgb, outline=(60, 60, 60))

        code = str(r.get(code_col, ""))
        name = str(r.get(name_col, ""))
        mark = "protect" if bool(r.get("protect_keep", False)) else "tweak"

        d.text((x, y + ph + 2), (code + " " + name)[:15], fill=(0, 0, 0), font=font)
        d.text((x, y + ph + 18), mark, fill=(80, 80, 80), font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def main():
    ap = argparse.ArgumentParser(
        description="读取 targets_alpha_1.00.csv，对 measured_L/a/b 做整体偏深、偏红、增饱和后处理，并保护指定编号。"
    )
    ap.add_argument("--in-csv", required=True, help="输入 CSV，通常是 targets_alpha_1.00.csv。")
    ap.add_argument("--out-dir", default="zhengwei2_targets_tweak_out")
    ap.add_argument("--lab-prefix", default="measured", help="默认读取 measured_L/a/b。")

    ap.add_argument("--global-l-offset", type=float, default=-2.0, help="整体 L 偏移。负数更深，默认 -2。")
    ap.add_argument("--global-a-offset", type=float, default=0.8, help="整体 a 偏移。正数偏红，默认 +0.8。")
    ap.add_argument("--global-b-offset", type=float, default=0.0, help="整体 b 偏移。正数偏黄，默认 0。")
    ap.add_argument("--global-chroma-scale", type=float, default=1.00, help="整体色度倍率。>1 更饱和，默认 1。")

    ap.add_argument(
        "--protect-codes",
        default="W177,W178,W179,W180,W193,W194,W195,W196,W225,W226",
        help="保护编号，不做新增后处理，逗号分隔。",
    )

    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv, encoding="utf-8-sig")

    code_col = pick_code_col(df)
    name_col = pick_name_col(df)
    Lc, ac, bc = pick_lab_cols(df, args.lab_prefix.strip())

    protect_codes = {
        x.strip().upper()
        for x in str(args.protect_codes).split(",")
        if x.strip()
    }

    has_standard = all(c in df.columns for c in ["standard_L", "standard_a", "standard_b"])

    rows = []
    for _, r in df.iterrows():
        row = dict(r)

        code = str(r.get(code_col, "")).strip().upper()

        L = float(pd.to_numeric(r[Lc], errors="coerce"))
        a = float(pd.to_numeric(r[ac], errors="coerce"))
        b = float(pd.to_numeric(r[bc], errors="coerce"))

        protect = code in protect_codes
        if protect:
            L2, a2, b2 = L, a, b
        else:
            L2, a2, b2 = adjust_lab(L, a, b, args)

        row["protect_keep"] = bool(protect)
        row["tweak_L"] = L2
        row["tweak_a"] = a2
        row["tweak_b"] = b2
        row["delta_L"] = L2 - L
        row["delta_a"] = a2 - a
        row["delta_b"] = b2 - b

        if has_standard:
            std = (float(r["standard_L"]), float(r["standard_a"]), float(r["standard_b"]))
            row["before_deltaE_to_standard"] = de00((L, a, b), std)
            row["after_deltaE_to_standard"] = de00((L2, a2, b2), std)
            row["deltaE_change"] = row["after_deltaE_to_standard"] - row["before_deltaE_to_standard"]

        rows.append(row)

    outdf = pd.DataFrame(rows)

    out_csv = out / "targets_alpha_1.00_tweak.csv"
    outdf.to_csv(out_csv, index=False, encoding="utf-8-sig")

    draw_board(outdf, out / "preview_before.png", (Lc, ac, bc), code_col, name_col, "before tweak")
    draw_board(outdf, out / "preview_after.png", ("tweak_L", "tweak_a", "tweak_b"), code_col, name_col, "after tweak")

    summary = {
        "in_csv": str(in_csv),
        "out_csv": str(out_csv),
        "input_lab_cols": [Lc, ac, bc],
        "protect_codes": sorted(protect_codes),
        "protect_count": int(outdf["protect_keep"].sum()),
        "tweak_count": int((~outdf["protect_keep"]).sum()),
        "params": vars(args),
    }

    if has_standard:
        summary.update(
            {
                "before_mean_deltaE": float(outdf["before_deltaE_to_standard"].mean()),
                "after_mean_deltaE": float(outdf["after_deltaE_to_standard"].mean()),
                "before_p95_deltaE": float(outdf["before_deltaE_to_standard"].quantile(0.95)),
                "after_p95_deltaE": float(outdf["after_deltaE_to_standard"].quantile(0.95)),
            }
        )

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Done ===")
    print("read csv:", in_csv)
    print("input lab cols:", Lc, ac, bc)
    print("protect count:", summary["protect_count"])
    print("tweak count:", summary["tweak_count"])
    if has_standard:
        print("before mean ΔE:", summary["before_mean_deltaE"])
        print("after  mean ΔE:", summary["after_mean_deltaE"])
    print("out csv:", out_csv)
    print("preview before:", out / "preview_before.png")
    print("preview after :", out / "preview_after.png")
    print("summary:", out / "summary.json")


if __name__ == "__main__":
    main()
