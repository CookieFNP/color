# -*- coding: utf-8 -*-
"""
用途：
    对“正伟2”这一批胶块的当前 Lab 结果做一层统一的视觉后处理：
    1) 整体更深一点（L 减小）
    2) 整体稍微偏红一点（a 增加）
    3) 可选轻微调整体饱和度（chroma_scale）
    4) 对指定保护色号不做这层额外后处理，保留它们原先的调色结果

适用场景：
    - 你已经有一份“当前结果 CSV”，里面至少包含：
        编号/名称，以及某一组 Lab 列
    - 例如：
        corrected_L, corrected_a, corrected_b
        或 fix_L, fix_a, fix_b
        或 final_L, final_a, final_b
        或 visual_L, visual_a, visual_b
        或 L, a, b

注意：
    - 这是“后处理微调脚本”，不是替代 visual_alpha_sweep_eval_series_rules.py 的主流程
    - 保护色号默认：
        W177 W178 W179 W180
        W193 W194 W195 W196
        W225 W226
      这些编号不会吃这层新的整体偏红/偏深修正
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# =========================
# Lab <-> sRGB 工具
# =========================
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


# =========================
# 列自动识别
# =========================
def pick_lab_cols(df, prefix=""):
    candidates = []
    if prefix:
        candidates.append((f"{prefix}_L", f"{prefix}_a", f"{prefix}_b"))

    candidates += [
        ("fix_L", "fix_a", "fix_b"),
        ("final_L", "final_a", "final_b"),
        ("visual_L", "visual_a", "visual_b"),
        ("visual_display_L", "visual_display_a", "visual_display_b"),
        ("corrected_L", "corrected_a", "corrected_b"),
        ("L", "a", "b"),
    ]

    for c in candidates:
        if all(x in df.columns for x in c):
            return c

    raise RuntimeError(
        "找不到可用的 Lab 列。请确认 CSV 中含有 "
        "corrected_L/a/b、fix_L/a/b、final_L/a/b、visual_display_L/a/b 或 L/a/b。"
    )


def pick_code_col(df):
    for c in ["code", "编号"]:
        if c in df.columns:
            return c
    raise RuntimeError("找不到编号列，请确认有 code 或 编号。")


def pick_name_col(df):
    for c in ["name", "名称"]:
        if c in df.columns:
            return c
    raise RuntimeError("找不到名称列，请确认有 name 或 名称。")


# =========================
# 视觉后处理
# =========================
def adjust_lab(L, a, b, args):
    """
    统一视觉后处理：
        1. 先整体亮度偏移
        2. 再按色度整体缩放（保持 hue 方向）
        3. 最后叠加红绿/黄蓝偏移
    """
    L2 = float(L) + float(args.global_l_offset)
    a2 = float(a)
    b2 = float(b)

    # 整体色度缩放（更浓 / 更淡）
    C = math.hypot(a2, b2)
    if C > 1e-8:
        C2 = C * float(args.global_chroma_scale)
        ratio = C2 / C
        a2 *= ratio
        b2 *= ratio

    # 整体轻微偏红 / 偏黄
    a2 += float(args.global_a_offset)
    b2 += float(args.global_b_offset)

    # 再限制一下范围
    L2 = float(np.clip(L2, 0, 100))
    a2 = float(np.clip(a2, -128, 127))
    b2 = float(np.clip(b2, -128, 127))
    return L2, a2, b2


def draw_board(df, out, cols_lab, code_col, name_col, title):
    Lc, ac, bc = cols_lab
    cols = 8
    pw = 104
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
        prot = "protect" if bool(r.get("protect_keep", False)) else "tweak"

        d.text((x, y + ph + 2), (code + " " + name)[:14], fill=(0, 0, 0), font=font)
        d.text((x, y + ph + 18), prot, fill=(80, 80, 80), font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def main():
    ap = argparse.ArgumentParser(
        description="对正伟2胶块结果做整体偏深、偏红的视觉后处理，并保护指定色号。"
    )
    ap.add_argument("--in-csv", required=True, help="输入 CSV，建议是当前处理结果 CSV。")
    ap.add_argument("--out-dir", default="zhengwei2_global_tweak_out")
    ap.add_argument("--lab-prefix", default="", help="如果想强制指定某组 Lab 前缀，例如 corrected / fix / final。")
    ap.add_argument("--std-prefix", default="std", help="若 CSV 中含 std_L/std_a/std_b，可自动统计前后自比较 ΔE。")

    # 你当前诉求：整体更深一点、整体偏红一点
    ap.add_argument("--global-l-offset", type=float, default=-2.0, help="整体 L 偏移。负值更深，正值更浅。默认 -2.0。")
    ap.add_argument("--global-a-offset", type=float, default=0.8, help="整体 a 偏移。正值偏红，负值偏绿。默认 +0.8。")
    ap.add_argument("--global-b-offset", type=float, default=0.0, help="整体 b 偏移。正值偏黄暖，负值偏蓝冷。默认 0。")
    ap.add_argument("--global-chroma-scale", type=float, default=1.00, help="整体色度倍率。>1 更浓，<1 更淡。默认 1.00。")

    ap.add_argument(
        "--protect-codes",
        default="W177,W178,W179,W180,W193,W194,W195,W196,W225,W226",
        help="不做这层额外后处理、保持原结果的编号列表，用逗号分隔。",
    )

    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv, encoding="utf-8-sig")
    code_col = pick_code_col(df)
    name_col = pick_name_col(df)
    Lc, ac, bc = pick_lab_cols(df, args.lab_prefix.strip())

    protect_codes = {
        x.strip().upper()
        for x in str(args.protect_codes).split(",")
        if x.strip()
    }

    std_cols = (f"{args.std_prefix}_L", f"{args.std_prefix}_a", f"{args.std_prefix}_b")
    has_std = all(c in df.columns for c in std_cols)

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

        row.update(
            {
                "protect_keep": protect,
                "tweak_L": L2,
                "tweak_a": a2,
                "tweak_b": b2,
                "delta_L": L2 - L,
                "delta_a": a2 - a,
                "delta_b": b2 - b,
            }
        )

        if has_std:
            std = (
                float(r[std_cols[0]]),
                float(r[std_cols[1]]),
                float(r[std_cols[2]]),
            )
            row["before_self_deltaE"] = de00((L, a, b), std)
            row["after_self_deltaE"] = de00((L2, a2, b2), std)
            row["self_deltaE_change"] = row["after_self_deltaE"] - row["before_self_deltaE"]

        rows.append(row)

    outdf = pd.DataFrame(rows)

    out_csv = out / "zhengwei2_global_tweak.csv"
    outdf.to_csv(out_csv, index=False, encoding="utf-8-sig")

    draw_board(
        outdf,
        out / "preview_before.png",
        (Lc, ac, bc),
        code_col,
        name_col,
        "before global tweak",
    )
    draw_board(
        outdf,
        out / "preview_after.png",
        ("tweak_L", "tweak_a", "tweak_b"),
        code_col,
        name_col,
        "after global tweak",
    )

    summary = {
        "in_csv": args.in_csv,
        "lab_cols": [Lc, ac, bc],
        "protect_codes": sorted(protect_codes),
        "protect_count": int(outdf["protect_keep"].sum()),
        "tweak_count": int((~outdf["protect_keep"]).sum()),
        "params": vars(args),
        "out_csv": str(out_csv),
    }

    if has_std:
        summary.update(
            {
                "before_mean_deltaE": float(outdf["before_self_deltaE"].mean()),
                "after_mean_deltaE": float(outdf["after_self_deltaE"].mean()),
                "before_p95_deltaE": float(outdf["before_self_deltaE"].quantile(0.95)),
                "after_p95_deltaE": float(outdf["after_self_deltaE"].quantile(0.95)),
            }
        )

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== Done ===")
    print("lab cols:", Lc, ac, bc)
    print("protect count:", summary["protect_count"])
    print("tweak count:", summary["tweak_count"])
    if has_std:
        print("before mean ΔE:", summary["before_mean_deltaE"])
        print("after  mean ΔE:", summary["after_mean_deltaE"])
    print("out csv:", out_csv)
    print("preview before:", out / "preview_before.png")
    print("preview after :", out / "preview_after.png")
    print("summary:", out / "summary.json")


if __name__ == "__main__":
    main()
