from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 标准 ColorChecker 24 色 sRGB
# 顺序：从左到右、从上到下，4行x6列
# ============================================================

REF_RGB = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
    [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
], dtype=np.float32)


CLICK_NAMES = [
    "1 white patch center / 白色块中心，第4行第1列",
    "2 black patch center / 黑色块中心，第4行第6列",
    "3 brown patch center / 棕色块中心，第1行第1列",
    "4 cyan-green patch center / 青绿色块中心，第1行第6列",
]

# canonical 600x400, cell=100
# 点击顺序：白、黑、棕、青绿
CANON_CLICK_POINTS = np.array([
    [50, 350],   # white: row4 col1
    [550, 350],  # black: row4 col6
    [50, 50],    # brown: row1 col1
    [550, 50],   # cyan/green: row1 col6
], dtype=np.float32)


# ============================================================
# Windows 中文路径安全读写
# ============================================================

def imread_u(path: str | Path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_u(path: str | Path, img: np.ndarray, params=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def resize_max(img: np.ndarray, max_side: int):
    if max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    s = max_side / float(m)
    out = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return out, s


def collect_images(inp: Path, patterns: str):
    if inp.is_file():
        return [inp]
    files = []
    for pat in patterns.split(","):
        pat = pat.strip()
        if pat:
            files += list(inp.glob(pat))
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted({p for p in files if p.suffix.lower() in exts})


# ============================================================
# 色彩空间与 rootpoly2
# ============================================================

def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb_u8(lin):
    lin = np.clip(np.asarray(lin, np.float32), 0, 1)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * (lin ** (1 / 2.4)) - 0.055)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)


def rootpoly_features(lin):
    lin = np.clip(lin, 0, 1).astype(np.float32)
    R, G, B = lin[..., 0], lin[..., 1], lin[..., 2]
    return np.stack([
        R, G, B,
        np.sqrt(np.clip(R * G, 0, None)),
        np.sqrt(np.clip(R * B, 0, None)),
        np.sqrt(np.clip(G * B, 0, None)),
        np.ones_like(R),
    ], axis=-1)


def fit_rootpoly(src_rgb, ref_rgb=REF_RGB, alpha=1e-6):
    X = rootpoly_features(srgb_to_linear(src_rgb)).reshape(-1, 7)
    Y = srgb_to_linear(ref_rgb).reshape(-1, 3)
    A = X.T @ X + alpha * np.eye(7, dtype=np.float32)
    B = X.T @ Y
    return np.linalg.solve(A.astype(np.float64), B.astype(np.float64)).astype(np.float32)


def apply_rootpoly(rgb_u8, W, chunk=600000):
    h, w = rgb_u8.shape[:2]
    flat = rgb_u8.reshape(-1, 3)
    out = np.empty_like(flat)
    for s in range(0, len(flat), chunk):
        e = min(len(flat), s + chunk)
        X = rootpoly_features(srgb_to_linear(flat[s:e])).reshape(-1, 7)
        out[s:e] = linear_to_srgb_u8(X @ W)
    return out.reshape(h, w, 3)


def rgb_to_lab_cv(rgb):
    arr = np.asarray(rgb, np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    lab[:, 0] = lab[:, 0] * 100.0 / 255.0
    lab[:, 1] = lab[:, 1] - 128.0
    lab[:, 2] = lab[:, 2] - 128.0
    return lab.reshape(np.asarray(rgb).shape)


def delta_e_2000(lab1, lab2):
    L1, a1, b1 = [float(x) for x in lab1]
    L2, a2, b2 = [float(x) for x in lab2]
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7))) if Cb else 0
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)

    def hp(a, b):
        if a == 0 and b == 0:
            return 0
        h = math.degrees(math.atan2(b, a))
        return h + 360 if h < 0 else h

    h1 = hp(a1p, b1)
    h2 = hp(a2p, b2)
    dL = L2 - L1
    dC = C2p - C1p

    if C1p * C2p == 0:
        dh = 0
    else:
        dh = h2 - h1
        if dh > 180:
            dh -= 360
        if dh < -180:
            dh += 360

    dH = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dh / 2))
    Lp = (L1 + L2) / 2
    Cp = (C1p + C2p) / 2

    if C1p * C2p == 0:
        hpbar = h1 + h2
    else:
        if abs(h1 - h2) <= 180:
            hpbar = (h1 + h2) / 2
        elif h1 + h2 < 360:
            hpbar = (h1 + h2 + 360) / 2
        else:
            hpbar = (h1 + h2 - 360) / 2

    T = (
        1
        - 0.17 * math.cos(math.radians(hpbar - 30))
        + 0.24 * math.cos(math.radians(2 * hpbar))
        + 0.32 * math.cos(math.radians(3 * hpbar + 6))
        - 0.20 * math.cos(math.radians(4 * hpbar - 63))
    )
    dt = 30 * math.exp(-((hpbar - 275) / 25) ** 2)
    Rc = 2 * math.sqrt(Cp ** 7 / (Cp ** 7 + 25 ** 7)) if Cp else 0
    Sl = 1 + (0.015 * (Lp - 50) ** 2) / math.sqrt(20 + (Lp - 50) ** 2)
    Sc = 1 + 0.045 * Cp
    Sh = 1 + 0.015 * Cp * T
    Rt = -math.sin(math.radians(2 * dt)) * Rc

    return float(math.sqrt((dL / Sl) ** 2 + (dC / Sc) ** 2 + (dH / Sh) ** 2 + Rt * (dC / Sc) * (dH / Sh)))


def chart_de(src_rgb, W):
    ref_lab = rgb_to_lab_cv(REF_RGB.astype(np.uint8))
    src_lab = rgb_to_lab_cv(np.clip(src_rgb, 0, 255).astype(np.uint8))
    corr = apply_rootpoly(np.clip(src_rgb.reshape(1, 24, 3), 0, 255).astype(np.uint8), W).reshape(24, 3)
    corr_lab = rgb_to_lab_cv(corr)
    before = [delta_e_2000(s, r) for s, r in zip(src_lab, ref_lab)]
    after = [delta_e_2000(c, r) for c, r in zip(corr_lab, ref_lab)]
    return float(np.mean(before)), float(np.mean(after))


# ============================================================
# 手动四点：白、黑、棕、青绿
# ============================================================

def draw_points_view(img, points, current_idx):
    view = img.copy()
    for i, p in enumerate(points):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(view, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(view, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    y0 = 28
    cv2.rectangle(view, (0, 0), (view.shape[1], 86), (0, 0, 0), -1)
    cv2.putText(view, "Click 4 patch CENTERS in order. r=reset, s=skip, q=quit",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if current_idx < 4:
        cv2.putText(view, CLICK_NAMES[current_idx], (10, y0 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
    else:
        cv2.putText(view, "Done. Press any key.", (10, y0 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
    return view


def manual_get_points(img_bgr, title, view_max_side=1100):
    disp, s = resize_max(img_bgr, view_max_side)
    points_disp = []

    win = title[:80]
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def callback(event, x, y, flags, param):
        nonlocal points_disp
        if event == cv2.EVENT_LBUTTONDOWN and len(points_disp) < 4:
            points_disp.append([float(x), float(y)])

    cv2.setMouseCallback(win, callback)

    while True:
        view = draw_points_view(disp, points_disp, len(points_disp))
        cv2.imshow(win, view)
        key = cv2.waitKey(30) & 0xFF

        if len(points_disp) >= 4:
            cv2.imshow(win, draw_points_view(disp, points_disp, 4))
            cv2.waitKey(250)
            break

        if key == ord("r"):
            points_disp = []
        elif key == ord("s"):
            cv2.destroyWindow(win)
            return None, "skip"
        elif key == ord("q") or key == 27:
            cv2.destroyWindow(win)
            return None, "quit"

    cv2.destroyWindow(win)
    pts = np.array(points_disp, dtype=np.float32) / max(s, 1e-8)
    return pts, "ok"


def save_points(path: Path, points):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "order": CLICK_NAMES,
        "points": np.asarray(points, dtype=float).tolist(),
        "note": "points are in processed image coordinates, order = white, black, brown, cyan-green",
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_points(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    pts = np.asarray(obj["points"], dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"points json invalid: {path}")
    return pts


# ============================================================
# 从四个色块中心点抽取 24 色
# ============================================================

def warp_chart_from_center_points(img_bgr, points):
    H, _ = cv2.findHomography(points.astype(np.float32), CANON_CLICK_POINTS, method=0)
    if H is None:
        raise RuntimeError("findHomography failed")
    warped = cv2.warpPerspective(img_bgr, H, (600, 400))
    return warped, H


def extract_24_from_warped(warp_bgr, patch_frac=0.42, debug_path=None):
    cell = 100
    rgb = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2RGB)
    patches = []
    dbg = warp_bgr.copy()

    half = int(round(cell * patch_frac * 0.5))
    for r in range(4):
        for c in range(6):
            cx = int(round((c + 0.5) * cell))
            cy = int(round((r + 0.5) * cell))
            x1 = max(0, cx - half)
            x2 = min(rgb.shape[1], cx + half)
            y1 = max(0, cy - half)
            y2 = min(rgb.shape[0], cy + half)

            pix = rgb[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
            lab = rgb_to_lab_cv(np.clip(pix, 0, 255).astype(np.uint8))
            L = lab[:, 0]
            lo, hi = np.percentile(L, [10, 90])
            keep = (L >= lo) & (L <= hi)
            if keep.sum() > 20:
                pix = pix[keep]
            patches.append(np.median(pix, axis=0))

            if debug_path:
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(dbg, str(r * 6 + c + 1), (x1, max(0, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    if debug_path:
        imwrite_u(debug_path, dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    return np.asarray(patches, dtype=np.float32)


# ============================================================
# 背景矫正 / 视觉微调
# ============================================================

def parse_lab(s):
    if not str(s).strip():
        return None
    p = [float(x.strip()) for x in str(s).replace("，", ",").split(",") if x.strip()]
    if len(p) != 3:
        raise ValueError("Lab 应为 L,a,b")
    return np.array(p, np.float32)


def estimate_bg_lab_from_border(bgr, bright_pct=65, max_chroma=22):
    h, w = bgr.shape[:2]
    bw = max(4, int(min(h, w) * 0.08))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pix = np.concatenate([
        rgb[:bw].reshape(-1, 3),
        rgb[-bw:].reshape(-1, 3),
        rgb[:, :bw].reshape(-1, 3),
        rgb[:, -bw:].reshape(-1, 3),
    ], axis=0)
    if len(pix) > 80000:
        pix = pix[np.linspace(0, len(pix) - 1, 80000).astype(int)]

    lab = rgb_to_lab_cv(pix.astype(np.uint8))
    L = lab[:, 0]
    C = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
    keep = (L >= np.percentile(L, bright_pct)) & (C <= max_chroma)
    if keep.sum() < 100:
        keep = L >= np.percentile(L, 70)
    return np.median(lab[keep], axis=0).astype(np.float32)


def apply_lab_shift_bgr(bgr, shift_lab):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + shift_lab[0] * 255.0 / 100.0, 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + shift_lab[1], 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + shift_lab[2], 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def visual_tweak(bgr, gamma=1.0, contrast=1.0, brightness=0.0, saturation=1.0):
    out = bgr
    if abs(gamma - 1) > 1e-6:
        table = np.array([(i / 255) ** (1 / max(gamma, 1e-6)) * 255 for i in range(256)], np.uint8)
        out = cv2.LUT(out, table)
    if abs(contrast - 1) > 1e-6 or abs(brightness) > 1e-6:
        out = np.clip(out.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    if abs(saturation - 1) > 1e-6:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def draw_overlay(bgr, points, text):
    out = bgr.copy()
    pts = np.asarray(points, dtype=np.float32)
    for i, p in enumerate(pts):
        cv2.circle(out, tuple(np.round(p).astype(int)), 7, (0, 255, 0), -1)
        cv2.putText(out, str(i + 1), tuple(np.round(p + np.array([8, -8])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.polylines(out, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    cv2.putText(out, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    return out


def write_csv_report(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


# ============================================================
# 主流程
# ============================================================

def process_one(path: Path, args, out_dir: Path, first_points_holder: dict):
    row = {"file": str(path), "status": "failed"}
    img = imread_u(path)
    if img is None:
        row["error"] = "read failed"
        return row

    work, scale = resize_max(img, args.process_max_side)
    row.update({
        "orig_w": img.shape[1], "orig_h": img.shape[0],
        "work_w": work.shape[1], "work_h": work.shape[0],
    })

    points_dir = out_dir / "points"
    point_json = points_dir / f"{path.stem}_points.json"

    if args.use_first_points and first_points_holder.get("points") is not None:
        points = first_points_holder["points"].copy()
    elif args.reuse_points and point_json.exists():
        points = load_points(point_json)
    else:
        points, status = manual_get_points(work, f"{path.name}", view_max_side=args.view_max_side)
        if status == "quit":
            raise KeyboardInterrupt("user quit")
        if status == "skip" or points is None:
            row["error"] = "manual skipped"
            return row
        save_points(point_json, points)
        if args.use_first_points and first_points_holder.get("points") is None:
            first_points_holder["points"] = points.copy()

    try:
        warped, H = warp_chart_from_center_points(work, points)
        if args.save_debug:
            imwrite_u(out_dir / "debug" / f"{path.stem}_chart_warped.jpg", warped, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        src_rgb = extract_24_from_warped(
            warped,
            patch_frac=args.patch_frac,
            debug_path=(out_dir / "debug" / f"{path.stem}_patches.jpg") if args.save_debug else None,
        )
        Wfit = fit_rootpoly(src_rgb, REF_RGB, alpha=args.ridge_alpha)
        before_de, after_de = chart_de(src_rgb, Wfit)
    except Exception as e:
        row["error"] = f"chart fitting failed: {repr(e)}"
        return row

    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    corr_rgb = apply_rootpoly(rgb, Wfit, chunk=args.chunk_pixels)
    corr = cv2.cvtColor(corr_rgb, cv2.COLOR_RGB2BGR)

    bg_lab = None
    bg_shift = np.array([0, 0, 0], np.float32)
    ref_bg = parse_lab(args.background_lab)
    if ref_bg is not None and args.bg_strength > 0:
        bg_lab = estimate_bg_lab_from_border(corr, args.bg_bright_percentile, args.bg_max_chroma)
        raw = ref_bg - bg_lab
        gains = np.array([args.bg_gain_L, args.bg_gain_a, args.bg_gain_b], np.float32)
        caps = np.array([args.bg_cap_L, args.bg_cap_a, args.bg_cap_b], np.float32)
        bg_shift = np.clip(raw * gains * args.bg_strength, -caps, caps)
        corr = apply_lab_shift_bgr(corr, bg_shift)

    manual_shift = parse_lab(args.lab_shift)
    if manual_shift is not None and np.linalg.norm(manual_shift) > 1e-8:
        corr = apply_lab_shift_bgr(corr, manual_shift)

    corr = visual_tweak(corr, args.gamma, args.contrast, args.brightness, args.saturation)

    ext = args.output_ext.lower().lstrip(".")
    if ext not in ["jpg", "jpeg", "png", "webp"]:
        ext = "jpg"
    out_path = out_dir / "corrected" / f"{path.stem}_corrected.{ext}"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality] if ext in ["jpg", "jpeg"] else []
    imwrite_u(out_path, corr, params)

    if args.save_overlay:
        ov = draw_overlay(work, points, f"{path.stem} chartDE {before_de:.2f}->{after_de:.2f}")
        imwrite_u(out_dir / "overlay" / f"{path.stem}_overlay.jpg", ov, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    if args.save_compare:
        imwrite_u(out_dir / "compare" / f"{path.stem}_compare.jpg",
                  np.concatenate([work, corr], axis=1), [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    row.update({
        "status": "ok",
        "output": str(out_path),
        "points_json": str(point_json),
        "chart_before_deltaE_mean": before_de,
        "chart_after_deltaE_mean": after_de,
        "bg_L": "" if bg_lab is None else float(bg_lab[0]),
        "bg_a": "" if bg_lab is None else float(bg_lab[1]),
        "bg_b": "" if bg_lab is None else float(bg_lab[2]),
        "bg_shift_L": float(bg_shift[0]),
        "bg_shift_a": float(bg_shift[1]),
        "bg_shift_b": float(bg_shift[2]),
    })
    return row


def main():
    ap = argparse.ArgumentParser(
        description="手动四点批量色彩校正：点击白/黑/棕/青绿色块中心，避免自动色卡识别方向混乱。"
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="batch_manual_color_correct_out")
    ap.add_argument("--patterns", default="*.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff,*.webp")

    ap.add_argument("--process-max-side", type=int, default=1024)
    ap.add_argument("--view-max-side", type=int, default=1100)
    ap.add_argument("--ridge-alpha", type=float, default=1e-6)
    ap.add_argument("--patch-frac", type=float, default=0.42)
    ap.add_argument("--chunk-pixels", type=int, default=600000)

    ap.add_argument("--reuse-points", action="store_true", help="如果已有 out/points/xxx_points.json，则直接复用")
    ap.add_argument("--use-first-points", action="store_true", help="只标第一张，后续全部复用同一组点；仅适合同一机位同一构图")

    ap.add_argument("--background-lab", default="84.71,-1.14,-3.64")
    ap.add_argument("--bg-strength", type=float, default=0.25)
    ap.add_argument("--bg-bright-percentile", type=float, default=65)
    ap.add_argument("--bg-max-chroma", type=float, default=22)
    ap.add_argument("--bg-gain-L", type=float, default=0.75)
    ap.add_argument("--bg-gain-a", type=float, default=0.35)
    ap.add_argument("--bg-gain-b", type=float, default=0.90)
    ap.add_argument("--bg-cap-L", type=float, default=8.0)
    ap.add_argument("--bg-cap-a", type=float, default=5.0)
    ap.add_argument("--bg-cap-b", type=float, default=10.0)

    ap.add_argument("--lab-shift", default="0,0,0")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--brightness", type=float, default=0.0)
    ap.add_argument("--saturation", type=float, default=1.0)

    ap.add_argument("--output-ext", default="jpg")
    ap.add_argument("--jpg-quality", type=int, default=90)
    ap.add_argument("--save-overlay", action="store_true")
    ap.add_argument("--save-compare", action="store_true")
    ap.add_argument("--save-debug", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_images(Path(args.input), args.patterns)
    if not files:
        print("没有找到图片")
        sys.exit(1)

    print("found", len(files), "images")
    print("点击顺序固定为：")
    for s in CLICK_NAMES:
        print(" ", s)
    print("按 r 重点，s 跳过，q/esc 退出。")

    rows = []
    first_points_holder = {}

    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p}")
        try:
            row = process_one(p, args, out_dir, first_points_holder)
        except KeyboardInterrupt:
            print("用户退出")
            break
        except Exception as e:
            row = {"file": str(p), "status": "failed", "error": repr(e)}

        rows.append(row)

        if row.get("status") == "ok":
            print(f"  ok chartDE {row['chart_before_deltaE_mean']:.2f}->{row['chart_after_deltaE_mean']:.2f}")
        else:
            print("  failed", row.get("error"))

        write_csv_report(out_dir / "batch_report.csv", rows)

    summary = {
        "n_total_processed": len(rows),
        "n_ok": sum(r.get("status") == "ok" for r in rows),
        "n_failed": sum(r.get("status") != "ok" for r in rows),
        "out": str(out_dir),
        "args": vars(args),
        "click_order": CLICK_NAMES,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Done ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
