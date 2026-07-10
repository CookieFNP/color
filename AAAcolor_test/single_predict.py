from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage import color


# =========================
# 基础 I/O
# =========================

def parse_lab(text: str) -> np.ndarray:
    text = str(text).strip().strip('"').strip("'")
    parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Lab 格式错误: {text}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def read_standards_csv(path: str | Path) -> dict[str, dict[str, Any]]:
    standards: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            code = row[0].strip().upper()
            if not code or code.lower() in {"code", "编号"}:
                continue
            name = row[1].strip()
            try:
                lab = parse_lab(",".join(row[2:]))
            except Exception:
                continue
            standards[code] = {"code": code, "name": name, "lab": lab}
    if not standards:
        raise RuntimeError(f"没有从标准 CSV 读到数据: {path}")
    return standards


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


# =========================
# 显示缩放 / 手动点选
# =========================

def make_display_image(bgr: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return disp, scale


def select_points(
    bgr: np.ndarray,
    n: int,
    title: str,
    max_w: int = 1400,
    max_h: int = 850,
) -> list[list[float]]:
    disp, scale = make_display_image(bgr, max_w, max_h)
    points: list[tuple[int, int]] = []
    img_show = disp.copy()

    help_text = f"{title}: left click {n} points, right click undo, Enter finish"
    print(help_text)

    def redraw() -> None:
        nonlocal img_show
        img_show = disp.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(img_show, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(img_show, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img_show, help_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < n:
                points.append((x, y))
                redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
                redraw()

    redraw()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, cb)

    while True:
        cv2.imshow(title, img_show)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # Enter
            if len(points) == n:
                break
            print(f"还需要 {n - len(points)} 个点")
        elif key == 27:  # Esc
            cv2.destroyWindow(title)
            raise RuntimeError("用户取消点选")
        elif key in (ord("u"), ord("U")):
            if points:
                points.pop()
                redraw()

    cv2.destroyWindow(title)
    return [[x / scale, y / scale] for x, y in points]


def select_roi(
    bgr: np.ndarray,
    title: str,
    max_w: int = 1400,
    max_h: int = 850,
) -> list[int]:
    disp, scale = make_display_image(bgr, max_w, max_h)
    print(f"{title}: 框选单个未知胶块 ROI，Enter/Space 确认，Esc 取消")
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    x, y, w, h = cv2.selectROI(title, disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if w <= 0 or h <= 0:
        raise RuntimeError("没有选中 ROI")
    roi = [
        int(round(x / scale)),
        int(round(y / scale)),
        int(round((x + w) / scale)),
        int(round((y + h) / scale)),
    ]
    return roi


# =========================
# 色彩转换
# =========================

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float64)
    if x.max() > 1.0:
        x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)


def rgb_to_lab_image(rgb_u8: np.ndarray) -> np.ndarray:
    rgb01 = np.asarray(rgb_u8, dtype=np.float64) / 255.0
    return color.rgb2lab(rgb01)


def rgb_to_lab_one(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64).reshape(1, 1, 3)
    if rgb.max() > 1:
        rgb = rgb / 255.0
    return color.rgb2lab(rgb)[0, 0].astype(np.float64)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return color.deltaE_ciede2000(np.asarray(lab1, dtype=np.float64), np.asarray(lab2, dtype=np.float64))


# =========================
# ColorChecker 提取和 rootpoly2
# =========================

def order_corners(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_quad_rgb(rgb: np.ndarray, corners: list[list[float]], output_size: tuple[int, int]) -> np.ndarray:
    w, h = output_size
    src = order_corners(np.asarray(corners, dtype=np.float32))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgb, M, (w, h), flags=cv2.INTER_LINEAR)


def extract_grid_rgb_means(
    rgb: np.ndarray,
    rows: int = 4,
    cols: int = 6,
    inner: float = 0.45,
    trim_percent: float = 10.0,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * w / cols
            x1 = (c + 1) * w / cols
            y0 = r * h / rows
            y1 = (r + 1) * h / rows

            cx0 = int(round(x0 + (1 - inner) * 0.5 * (x1 - x0)))
            cx1 = int(round(x1 - (1 - inner) * 0.5 * (x1 - x0)))
            cy0 = int(round(y0 + (1 - inner) * 0.5 * (y1 - y0)))
            cy1 = int(round(y1 - (1 - inner) * 0.5 * (y1 - y0)))

            patch = rgb[cy0:cy1, cx0:cx1].reshape(-1, 3).astype(np.float64)
            if patch.size == 0:
                out.append([0, 0, 0])
                continue

            if trim_percent > 0 and len(patch) > 20:
                lo = np.percentile(patch, trim_percent, axis=0)
                hi = np.percentile(patch, 100 - trim_percent, axis=0)
                mask = np.all((patch >= lo) & (patch <= hi), axis=1)
                if mask.sum() > 10:
                    patch = patch[mask]
            out.append(np.mean(patch, axis=0))
    return np.asarray(out, dtype=np.float64)


def rootpoly2_features(linear_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(linear_rgb, dtype=np.float64)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]
    return np.stack(
        [
            R,
            G,
            B,
            np.sqrt(np.clip(R * G, 0, None)),
            np.sqrt(np.clip(R * B, 0, None)),
            np.sqrt(np.clip(G * B, 0, None)),
            np.ones_like(R),
        ],
        axis=-1,
    )


def fit_rootpoly2(captured_rgb: np.ndarray, reference_rgb: np.ndarray, alpha: float = 1e-6) -> np.ndarray:
    X = rootpoly2_features(srgb_to_linear(captured_rgb))
    Y = srgb_to_linear(reference_rgb)

    X2 = X.reshape(-1, X.shape[-1])
    Y2 = Y.reshape(-1, 3)

    reg = alpha * np.eye(X2.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    W = np.linalg.solve(X2.T @ X2 + reg, X2.T @ Y2)
    return W


def apply_rootpoly2_to_rgb(rgb_u8: np.ndarray, W: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(rgb_u8)
    X = rootpoly2_features(lin).reshape(-1, 7)
    out_lin = (X @ W).reshape(rgb_u8.shape)
    return linear_to_srgb(out_lin)


# =========================
# ROI / 背景稳健取 Lab
# =========================

def clip_roi(roi: list[int], w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"ROI 无效: {roi}")
    return [x1, y1, x2, y2]


def robust_lab_from_rgb_roi(
    rgb_u8: np.ndarray,
    roi: list[int],
    center_ratio: float = 0.72,
    trim_percent: float = 10.0,
) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    x1, y1, x2, y2 = clip_roi(roi, w, h)

    rw = x2 - x1
    rh = y2 - y1
    dx = int(round((1 - center_ratio) * 0.5 * rw))
    dy = int(round((1 - center_ratio) * 0.5 * rh))
    x1c, x2c = x1 + dx, x2 - dx
    y1c, y2c = y1 + dy, y2 - dy

    patch_rgb = rgb_u8[y1c:y2c, x1c:x2c]
    lab = rgb_to_lab_image(patch_rgb).reshape(-1, 3)
    if lab.size == 0:
        raise RuntimeError("ROI patch 为空")

    if trim_percent > 0 and len(lab) > 30:
        L = lab[:, 0]
        lo = np.percentile(L, trim_percent)
        hi = np.percentile(L, 100 - trim_percent)
        mask = (L >= lo) & (L <= hi)
        if mask.sum() > 20:
            lab = lab[mask]

    return np.median(lab, axis=0).astype(np.float64)


def local_background_lab(
    corrected_rgb: np.ndarray,
    roi: list[int],
    ref_bg_lab: np.ndarray,
    margin: int = 36,
    bright_percentile: float = 70,
    bg_max_chroma: float = 18,
    bg_ab_max_dist: float = 24,
) -> tuple[np.ndarray, int]:
    h, w = corrected_rgb.shape[:2]
    x1, y1, x2, y2 = clip_roi(roi, w, h)

    ex1 = max(0, x1 - margin)
    ey1 = max(0, y1 - margin)
    ex2 = min(w, x2 + margin)
    ey2 = min(h, y2 + margin)

    crop_rgb = corrected_rgb[ey1:ey2, ex1:ex2]
    lab = rgb_to_lab_image(crop_rgb)

    mask = np.ones(lab.shape[:2], dtype=bool)
    rx1, ry1 = x1 - ex1, y1 - ey1
    rx2, ry2 = x2 - ex1, y2 - ey1
    mask[ry1:ry2, rx1:rx2] = False

    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    C = np.sqrt(a * a + b * b)
    ab_dist = np.sqrt((a - ref_bg_lab[1]) ** 2 + (b - ref_bg_lab[2]) ** 2)

    cand = mask & (C <= bg_max_chroma) & (ab_dist <= bg_ab_max_dist)
    if cand.sum() < 50:
        cand = mask & (C <= bg_max_chroma)
    if cand.sum() < 50:
        cand = mask

    L_cand = L[cand]
    if len(L_cand) > 0:
        th = np.percentile(L_cand, bright_percentile)
        cand2 = cand & (L >= th)
        if cand2.sum() >= 30:
            cand = cand2

    vals = lab[cand].reshape(-1, 3)
    if len(vals) == 0:
        return ref_bg_lab.astype(np.float64), 0

    # 取较亮背景，避开阴影；用中位数抗污染
    return np.median(vals, axis=0).astype(np.float64), int(len(vals))


def apply_known_bg_correction(
    root_lab: np.ndarray,
    local_bg: np.ndarray,
    ref_bg: np.ndarray,
    bg_strength: float = 0.25,
    gains: tuple[float, float, float] = (0.75, 0.35, 0.90),
    caps: tuple[float, float, float] = (8.0, 5.0, 10.0),
    family_limit: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    raw_shift = np.asarray(gains, dtype=np.float64) * (ref_bg - local_bg)
    raw_shift = np.clip(raw_shift, -np.asarray(caps), np.asarray(caps))
    shift = bg_strength * raw_shift

    if family_limit:
        L, a, b = root_lab
        C = math.sqrt(a * a + b * b)
        if C < 8:
            fam_cap = np.array([4.0, 2.0, 4.0])
        elif b > 18 and a < 18:
            fam_cap = np.array([5.0, 2.5, 6.0])
        elif a > 8 and b > -5:
            fam_cap = np.array([4.0, 3.0, 5.0])
        elif L > 70:
            fam_cap = np.array([4.0, 2.5, 5.0])
        else:
            fam_cap = np.array([5.0, 3.0, 6.0])
        shift = np.clip(shift, -fam_cap, fam_cap)

    final_lab = root_lab + shift
    final_lab[0] = np.clip(final_lab[0], 0, 100)
    final_lab[1:] = np.clip(final_lab[1:], -128, 127)
    return final_lab.astype(np.float64), shift.astype(np.float64)


# =========================
# single residual model
# =========================

def build_single_features(
    raw_lab: np.ndarray,
    root_lab: np.ndarray,
    base_lab: np.ndarray,
    local_bg: np.ndarray,
    ref_bg: np.ndarray,
) -> np.ndarray:
    base_L, base_a, base_b = [float(x) for x in base_lab]
    root_L, root_a, root_b = [float(x) for x in root_lab]
    raw_L, raw_a, raw_b = [float(x) for x in raw_lab]
    bg_L, bg_a, bg_b = [float(x) for x in local_bg]

    chroma = math.sqrt(base_a * base_a + base_b * base_b)
    hue = math.atan2(base_b, base_a)

    is_gray = 1.0 if chroma < 8 else 0.0
    is_red_orange = 1.0 if base_a > 8 and base_b > -5 else 0.0
    is_yellow = 1.0 if base_b > 18 else 0.0
    is_light = 1.0 if base_L > 70 else 0.0
    is_dark = 1.0 if base_L < 42 else 0.0

    return np.array([
        1.0,
        base_L / 100.0, base_a / 128.0, base_b / 128.0,
        root_L / 100.0, root_a / 128.0, root_b / 128.0,
        raw_L / 100.0, raw_a / 128.0, raw_b / 128.0,
        bg_L / 100.0, bg_a / 128.0, bg_b / 128.0,
        (bg_L - ref_bg[0]) / 100.0,
        (bg_a - ref_bg[1]) / 128.0,
        (bg_b - ref_bg[2]) / 128.0,
        chroma / 128.0, math.sin(hue), math.cos(hue),
        is_gray, is_red_orange, is_yellow, is_light, is_dark,
        (base_L / 100.0) * (chroma / 128.0),
        (base_a / 128.0) * (base_b / 128.0),
    ], dtype=np.float64)


def apply_single_residual_model(
    model: dict[str, Any],
    raw_lab: np.ndarray,
    root_lab: np.ndarray,
    base_lab: np.ndarray,
    local_bg: np.ndarray,
    ref_bg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    W = np.asarray(model["W"], dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)

    feat = build_single_features(raw_lab, root_lab, base_lab, local_bg, ref_bg)

    if len(feat) != len(mean):
        raise RuntimeError(
            f"模型特征数不匹配：当前脚本 {len(feat)}，模型 {len(mean)}。"
            f"请确认使用的是 single_residual_model.json。"
        )

    x = (feat - mean) / std
    pred_shift = x @ W

    cap_obj = model.get("cap", {"L": 12, "a": 28, "b": 36})
    cap = np.array([cap_obj["L"], cap_obj["a"], cap_obj["b"]], dtype=np.float64)
    pred_shift = np.clip(pred_shift, -cap, cap)

    lab = base_lab + pred_shift
    lab[0] = np.clip(lab[0], 0, 100)
    lab[1:] = np.clip(lab[1:], -128, 127)
    return lab.astype(np.float64), pred_shift.astype(np.float64)


# =========================
# 分类
# =========================

def nearest_standards(lab: np.ndarray, standards: dict[str, dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    codes = list(standards.keys())
    labs = np.vstack([standards[c]["lab"] for c in codes])
    de = delta_e_2000(lab[None, :], labs)
    order = np.argsort(de)[:top_k]
    out = []
    for idx in order:
        code = codes[int(idx)]
        out.append({
            "rank": len(out) + 1,
            "code": code,
            "name": standards[code]["name"],
            "deltaE2000": float(de[int(idx)]),
            "std_L": float(standards[code]["lab"][0]),
            "std_a": float(standards[code]["lab"][1]),
            "std_b": float(standards[code]["lab"][2]),
        })
    return out


def confidence_from_top(top: list[dict[str, Any]]) -> tuple[str, float]:
    if not top:
        return "none", 0.0
    d1 = float(top[0]["deltaE2000"])
    d2 = float(top[1]["deltaE2000"]) if len(top) > 1 else 999.0
    margin = d2 - d1

    if d1 <= 4.0 and margin >= 1.5:
        return "high", margin
    if d1 <= 7.0 and margin >= 1.0:
        return "medium", margin
    if d1 <= 9.0 and margin >= 0.6:
        return "low", margin
    return "very_low", margin


# =========================
# 主流程
# =========================

def main() -> None:
    ap = argparse.ArgumentParser(description="单个未知胶块识别：ColorChecker rootpoly2 + bg0.25 + single residual model + 128标准色最近邻。")

    ap.add_argument("--photo", required=True, help="待识别照片")
    ap.add_argument("--standard-chart", required=True, help="标准 ColorChecker 图片，例如 standard_chart.png")
    ap.add_argument("--standards-csv", required=True, help="128 标准 Lab CSV，例如 data.csv")
    ap.add_argument("--model", required=True, help="single_residual_model.json")
    ap.add_argument("--out", default="single_predict_out")

    ap.add_argument("--chart-corners-json", default="", help="可选：复用色卡四角 JSON")
    ap.add_argument("--save-chart-corners-json", default="", help="可选：保存本次色卡四角 JSON")
    ap.add_argument("--roi", default="", help='可选：直接传 ROI，格式 "x1,y1,x2,y2"')
    ap.add_argument("--roi-json", default="", help="可选：读取单个 ROI JSON，格式可为 {'roi':[x1,y1,x2,y2]} 或 [x1,y1,x2,y2]")
    ap.add_argument("--save-roi-json", default="", help="可选：保存本次 ROI JSON")

    ap.add_argument("--background-lab", default="", help='默认读取模型内 reference_bg_lab；也可手动传 "84.71,-1.14,-3.64"')
    ap.add_argument("--bg-strength", type=float, default=0.25)
    ap.add_argument("--bg-margin", type=int, default=36)
    ap.add_argument("--bg-bright-percentile", type=float, default=70.0)
    ap.add_argument("--bg-max-chroma", type=float, default=18.0)
    ap.add_argument("--bg-ab-max-dist", type=float, default=24.0)
    ap.add_argument("--no-family-limit", action="store_true")

    ap.add_argument("--ridge-alpha", type=float, default=1e-6)
    ap.add_argument("--chart-warp-w", type=int, default=600)
    ap.add_argument("--chart-warp-h", type=int, default=400)
    ap.add_argument("--view-max-w", type=int, default=1400)
    ap.add_argument("--view-max-h", type=int, default=850)
    ap.add_argument("--top-k", type=int, default=5)

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(args.photo, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.photo)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    std_chart_bgr = cv2.imread(args.standard_chart, cv2.IMREAD_COLOR)
    if std_chart_bgr is None:
        raise FileNotFoundError(args.standard_chart)
    std_chart_rgb = cv2.cvtColor(std_chart_bgr, cv2.COLOR_BGR2RGB)

    standards = read_standards_csv(args.standards_csv)
    model = json.loads(Path(args.model).read_text(encoding="utf-8-sig"))

    if args.background_lab.strip():
        ref_bg_lab = parse_lab(args.background_lab)
    else:
        ref_bg_lab = np.asarray(model.get("reference_bg_lab", [84.71, -1.14, -3.64]), dtype=np.float64)

    # 色卡四角
    if args.chart_corners_json:
        obj = json.loads(Path(args.chart_corners_json).read_text(encoding="utf-8-sig"))
        chart_corners = obj["corners"] if isinstance(obj, dict) and "corners" in obj else obj
    else:
        chart_corners = select_points(
            bgr,
            4,
            "Select ColorChecker corners: TL, TR, BR, BL",
            max_w=args.view_max_w,
            max_h=args.view_max_h,
        )

    save_chart_path = args.save_chart_corners_json or str(out_dir / "chart_corners.json")
    save_json(save_chart_path, {"corners": chart_corners})

    # ROI
    if args.roi.strip():
        roi = [int(round(float(x))) for x in args.roi.replace("，", ",").split(",")]
        if len(roi) != 4:
            raise ValueError("--roi 需要 x1,y1,x2,y2")
    elif args.roi_json:
        obj = json.loads(Path(args.roi_json).read_text(encoding="utf-8-sig"))
        if isinstance(obj, dict):
            roi = obj.get("roi") or obj.get("target_roi")
        else:
            roi = obj
        roi = [int(round(float(x))) for x in roi]
    else:
        roi = select_roi(bgr, "Select target ROI", max_w=args.view_max_w, max_h=args.view_max_h)

    roi = clip_roi(roi, rgb.shape[1], rgb.shape[0])
    save_roi_path = args.save_roi_json or str(out_dir / "target_roi.json")
    save_json(save_roi_path, {"roi": roi})

    # 1. 拟合 rootpoly2
    captured_chart_rgb = warp_quad_rgb(rgb, chart_corners, (args.chart_warp_w, args.chart_warp_h))
    reference_chart_rgb = cv2.resize(std_chart_rgb, (args.chart_warp_w, args.chart_warp_h), interpolation=cv2.INTER_AREA)

    cap24 = extract_grid_rgb_means(captured_chart_rgb, rows=4, cols=6)
    ref24 = extract_grid_rgb_means(reference_chart_rgb, rows=4, cols=6)

    W_root = fit_rootpoly2(cap24, ref24, alpha=args.ridge_alpha)

    # 2. 只校正 ROI 周围 crop，避免整张大图爆内存
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = roi
    margin = max(args.bg_margin + 10, 50)
    ex1 = max(0, x1 - margin)
    ey1 = max(0, y1 - margin)
    ex2 = min(w, x2 + margin)
    ey2 = min(h, y2 + margin)

    rgb_crop = rgb[ey1:ey2, ex1:ex2]
    root_crop_rgb = apply_rootpoly2_to_rgb(rgb_crop, W_root)

    roi_in_crop = [x1 - ex1, y1 - ey1, x2 - ex1, y2 - ey1]

    raw_lab = robust_lab_from_rgb_roi(rgb, roi)
    root_lab = robust_lab_from_rgb_roi(root_crop_rgb, roi_in_crop)

    local_bg, bg_count = local_background_lab(
        root_crop_rgb,
        roi_in_crop,
        ref_bg_lab,
        margin=args.bg_margin,
        bright_percentile=args.bg_bright_percentile,
        bg_max_chroma=args.bg_max_chroma,
        bg_ab_max_dist=args.bg_ab_max_dist,
    )

    # 3. 背景轻校正，得到 base/final Lab
    base_lab, bg_shift = apply_known_bg_correction(
        root_lab=root_lab,
        local_bg=local_bg,
        ref_bg=ref_bg_lab,
        bg_strength=args.bg_strength,
        family_limit=not args.no_family_limit,
    )

    # 4. single residual
    model_lab, model_shift = apply_single_residual_model(
        model=model,
        raw_lab=raw_lab,
        root_lab=root_lab,
        base_lab=base_lab,
        local_bg=local_bg,
        ref_bg=ref_bg_lab,
    )

    # 5. 分类
    top = nearest_standards(model_lab, standards, top_k=args.top_k)
    conf, margin_top2 = confidence_from_top(top)

    # 也输出 base 不加 residual 的分类，方便对比
    base_top = nearest_standards(base_lab, standards, top_k=args.top_k)
    root_top = nearest_standards(root_lab, standards, top_k=args.top_k)
    raw_top = nearest_standards(raw_lab, standards, top_k=args.top_k)

    result = {
        "input": {
            "photo": args.photo,
            "standard_chart": args.standard_chart,
            "standards_csv": args.standards_csv,
            "model": args.model,
            "roi": roi,
            "chart_corners": chart_corners,
        },
        "lab": {
            "raw_lab": [float(x) for x in raw_lab],
            "root_lab": [float(x) for x in root_lab],
            "local_bg_lab": [float(x) for x in local_bg],
            "reference_bg_lab": [float(x) for x in ref_bg_lab],
            "bg_shift": [float(x) for x in bg_shift],
            "base_lab_after_bg": [float(x) for x in base_lab],
            "model_shift": [float(x) for x in model_shift],
            "model_lab": [float(x) for x in model_lab],
            "bg_pixel_count": int(bg_count),
        },
        "prediction": {
            "top1_code": top[0]["code"],
            "top1_name": top[0]["name"],
            "top1_deltaE2000": top[0]["deltaE2000"],
            "top2_code": top[1]["code"] if len(top) > 1 else "",
            "top2_name": top[1]["name"] if len(top) > 1 else "",
            "top2_deltaE2000": top[1]["deltaE2000"] if len(top) > 1 else None,
            "top2_margin": margin_top2,
            "confidence": conf,
            "topk": top,
        },
        "debug_compare": {
            "raw_topk": raw_top,
            "root_topk": root_top,
            "base_topk": base_top,
            "model_topk": top,
        },
    }

    save_json(out_dir / "single_predict_result.json", result)

    pred_rows = []
    for item in top:
        row = dict(item)
        row.update({
            "confidence": conf,
            "top2_margin": margin_top2,
            "model_L": float(model_lab[0]),
            "model_a": float(model_lab[1]),
            "model_b": float(model_lab[2]),
        })
        pred_rows.append(row)
    write_csv(out_dir / "single_predict_topk.csv", pred_rows)

    # 保存 overlay
    overlay = bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)
    text1 = f"{top[0]['code']} {top[0]['name']} dE={top[0]['deltaE2000']:.2f} {conf}"
    cv2.putText(overlay, text1, (max(0, x1), max(40, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    cv2.imwrite(str(out_dir / "single_predict_overlay.jpg"), overlay)

    # 保存 crop 对照
    cv2.imwrite(str(out_dir / "target_crop_original.jpg"), cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "target_crop_root_corrected.jpg"), cv2.cvtColor(root_crop_rgb[roi_in_crop[1]:roi_in_crop[3], roi_in_crop[0]:roi_in_crop[2]], cv2.COLOR_RGB2BGR))

    print("\n=== Single Predict Result ===")
    print(f"Top1: {top[0]['code']} {top[0]['name']}  ΔE2000={top[0]['deltaE2000']:.3f}")
    if len(top) > 1:
        print(f"Top2: {top[1]['code']} {top[1]['name']}  ΔE2000={top[1]['deltaE2000']:.3f}")
    if len(top) > 2:
        print(f"Top3: {top[2]['code']} {top[2]['name']}  ΔE2000={top[2]['deltaE2000']:.3f}")
    print(f"Confidence: {conf}, top2_margin={margin_top2:.3f}")

    print("\nLab:")
    print(f"raw   = {raw_lab[0]:.2f}, {raw_lab[1]:.2f}, {raw_lab[2]:.2f}")
    print(f"root  = {root_lab[0]:.2f}, {root_lab[1]:.2f}, {root_lab[2]:.2f}")
    print(f"base  = {base_lab[0]:.2f}, {base_lab[1]:.2f}, {base_lab[2]:.2f}")
    print(f"model = {model_lab[0]:.2f}, {model_lab[1]:.2f}, {model_lab[2]:.2f}")
    print(f"local_bg = {local_bg[0]:.2f}, {local_bg[1]:.2f}, {local_bg[2]:.2f}, count={bg_count}")

    print("\nSaved:")
    print(out_dir / "single_predict_result.json")
    print(out_dir / "single_predict_topk.csv")
    print(out_dir / "single_predict_overlay.jpg")


if __name__ == "__main__":
    main()
