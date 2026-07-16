from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.color_math import rgb_to_lab, delta_e_2000
from src.glue_mask import build_glue_block_mask, get_glue_block_representative_rgb
from src.io_utils import imread_unicode, imwrite_unicode


def resolve_path(path_text: str | None, report_path: Path) -> Path | None:
    if not path_text:
        return None

    p = Path(path_text)

    if p.exists():
        return p

    p2 = report_path.parent / p
    if p2.exists():
        return p2

    p3 = Path.cwd() / p
    if p3.exists():
        return p3

    return p


def frange_0_1(step: float = 0.1) -> list[float]:
    vals = []
    x = 0.0
    while x < 1.0 + 1e-9:
        vals.append(round(x, 6))
        x += step
    return vals


def parse_alpha_list(text: str | None, step: float) -> list[float]:
    if text:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    return frange_0_1(step)


def stat_pack(x: list[float] | np.ndarray) -> dict:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p95": None,
        }

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def lab_std_to_cv_delta(delta_lab: np.ndarray) -> tuple[float, float, float]:
    """
    标准 Lab 残差转成 OpenCV Lab 图像里的残差。

    标准 Lab:
        L: 0~100
        a/b: 正常 Lab 坐标

    OpenCV uint8 Lab:
        L: 0~255
        a/b: +128 偏移
    """
    dL, da, db = map(float, delta_lab)
    return dL * 255.0 / 100.0, da, db




def code_to_number(code: str | None) -> int | None:
    """W053 -> 53。解析失败返回 None。"""
    if code is None:
        return None
    s = str(code).strip().upper()
    if s.startswith("W") and s[1:].isdigit():
        return int(s[1:])
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else None


def clamp_signed_delta(value: float, pos_cap: float | None = None, neg_cap: float | None = None) -> float:
    """
    对 Lab 残差做限幅。
    pos_cap 限制正向，例如 b 正向就是“变黄”的最大幅度。
    neg_cap 限制负向，例如 b 负向就是“变蓝”的最大幅度。
    """
    v = float(value)
    if pos_cap is not None and v > float(pos_cap):
        v = float(pos_cap)
    if neg_cap is not None and v < -float(neg_cap):
        v = -float(neg_cap)
    return v


def visual_rule_for_code(
    code: str | None,
    *,
    enable_rules: bool,
    rule_strength: float,
    default_l_original_mix: float,
    default_b_scale: float,
    default_b_pos_cap: float | None,
    default_b_neg_cap: float | None,
) -> dict:
    """
    视觉调参规则，不影响测色统计的 ROI/mask，只影响最终 preview 怎么渲染。

    当前规则来自你对 128 色肉眼观察：
    - W017-W032、W097-W112 大致像肉眼：尽量不动。
    - W033-W048 偏浅偏黄：减少 b 正向残差，L 更多回原图。
    - W049-W064 浅灰/灰白高风险，尤其 W053：强限制变黄，L 更多回原图。
    - W081-W095、W113-W127 深色/棕咖偏浅：L 更多回原图。
    """
    strength = float(np.clip(rule_strength, 0.0, 1.0))
    n = code_to_number(code)

    # 默认规则：不启用系列规则时，由命令行统一控制。
    base = {
        "group": "default",
        "l_original_mix": float(np.clip(default_l_original_mix, 0.0, 1.0)),
        "b_scale": float(default_b_scale),
        "b_pos_cap": default_b_pos_cap,
        "b_neg_cap": default_b_neg_cap,
    }

    if not enable_rules or n is None:
        return base

    # 目标规则。后面会用 rule_strength 从 base 插值过去。
    target = dict(base)

    if 17 <= n <= 32:
        target.update(group="W017-W032_keep", l_original_mix=0.0, b_scale=1.0, b_pos_cap=None, b_neg_cap=None)

    elif 33 <= n <= 48:
        target.update(group="W033-W048_less_yellow_less_light", l_original_mix=0.65, b_scale=0.45, b_pos_cap=1.50, b_neg_cap=3.00)

    elif 49 <= n <= 64:
        target.update(group="W049-W064_gray_white_protect", l_original_mix=0.75, b_scale=0.25, b_pos_cap=0.80, b_neg_cap=2.00)

    elif 65 <= n <= 80:
        target.update(group="W065-W080_mid_gray_soft", l_original_mix=0.50, b_scale=0.50, b_pos_cap=1.20, b_neg_cap=2.50)

    elif 81 <= n <= 95:
        target.update(group="W081-W095_deep_gray_keep_L", l_original_mix=0.85, b_scale=0.60, b_pos_cap=1.20, b_neg_cap=2.50)

    elif n == 96:
        target.update(group="W096_keep", l_original_mix=0.0, b_scale=1.0, b_pos_cap=None, b_neg_cap=None)

    elif 97 <= n <= 112:
        target.update(group="W097-W112_keep", l_original_mix=0.0, b_scale=1.0, b_pos_cap=None, b_neg_cap=None)

    elif 113 <= n <= 127:
        target.update(group="W113-W127_brown_keep_L", l_original_mix=0.75, b_scale=0.60, b_pos_cap=1.80, b_neg_cap=3.00)

    elif n == 128:
        target.update(group="W128_keep", l_original_mix=0.0, b_scale=1.0, b_pos_cap=None, b_neg_cap=None)

    # W053 单独强保护：你看到的“非常暖的白色”就是这里最该压住。
    if n == 53:
        target.update(group="W053_special_warm_white_protect", l_original_mix=0.90, b_scale=0.10, b_pos_cap=0.40, b_neg_cap=1.50)

    def interp_float(a, b):
        return float(a) * (1.0 - strength) + float(b) * strength

    out = dict(base)
    out["group"] = target["group"]
    out["l_original_mix"] = interp_float(base["l_original_mix"], target["l_original_mix"])
    out["b_scale"] = interp_float(base["b_scale"], target["b_scale"])

    # cap 的插值比较麻烦：如果 base 没 cap，则直接用 target cap；这样规则一开就能保护异常点。
    out["b_pos_cap"] = target["b_pos_cap"] if target["b_pos_cap"] is not None else base["b_pos_cap"]
    out["b_neg_cap"] = target["b_neg_cap"] if target["b_neg_cap"] is not None else base["b_neg_cap"]
    return out


def effective_lab_delta_for_target(
    target: dict,
    *,
    alpha: float,
    ab_scale: float,
    l_scale: float,
    enable_rules: bool,
    rule_strength: float,
    default_l_original_mix: float,
    default_b_scale: float,
    default_b_pos_cap: float | None,
    default_b_neg_cap: float | None,
) -> tuple[float, float, float, dict]:
    """返回用于视觉渲染的有效 Lab 残差 dL, da, db，以及该色号使用的规则。"""
    dL, da, db = map(float, target["residual_lab"])
    rule = visual_rule_for_code(
        target.get("code"),
        enable_rules=enable_rules,
        rule_strength=rule_strength,
        default_l_original_mix=default_l_original_mix,
        default_b_scale=default_b_scale,
        default_b_pos_cap=default_b_pos_cap,
        default_b_neg_cap=default_b_neg_cap,
    )

    eff_dL = dL * float(alpha) * float(l_scale)
    eff_da = da * float(alpha) * float(ab_scale)

    # b 单独按色系缩放和限幅，防止“过黄”。
    eff_db = db * float(alpha) * float(ab_scale) * float(rule["b_scale"])
    eff_db = clamp_signed_delta(eff_db, pos_cap=rule["b_pos_cap"], neg_cap=rule["b_neg_cap"])
    return eff_dL, eff_da, eff_db, rule


def save_visual_rules_csv(
    path: Path,
    fixed_targets: list[dict],
    *,
    alpha: float,
    ab_scale: float,
    l_scale: float,
    enable_rules: bool,
    rule_strength: float,
    default_l_original_mix: float,
    default_b_scale: float,
    default_b_pos_cap: float | None,
    default_b_neg_cap: float | None,
) -> None:
    """保存每个 W 色号实际吃到的视觉规则，方便你查 W053 / W033-W064。"""
    rows = []
    for t in fixed_targets:
        raw_dL, raw_da, raw_db = map(float, t["residual_lab"])
        eff_dL, eff_da, eff_db, rule = effective_lab_delta_for_target(
            t,
            alpha=alpha,
            ab_scale=ab_scale,
            l_scale=l_scale,
            enable_rules=enable_rules,
            rule_strength=rule_strength,
            default_l_original_mix=default_l_original_mix,
            default_b_scale=default_b_scale,
            default_b_pos_cap=default_b_pos_cap,
            default_b_neg_cap=default_b_neg_cap,
        )
        rows.append({
            "index": t.get("index"),
            "code": t.get("code"),
            "name": t.get("name"),
            "group": rule["group"],
            "raw_residual_L": raw_dL,
            "raw_residual_a": raw_da,
            "raw_residual_b": raw_db,
            "effective_delta_L": eff_dL,
            "effective_delta_a": eff_da,
            "effective_delta_b": eff_db,
            "l_original_mix": rule["l_original_mix"],
            "b_scale": rule["b_scale"],
            "b_pos_cap": rule["b_pos_cap"],
            "b_neg_cap": rule["b_neg_cap"],
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def feather_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    m = mask.astype(np.float32)

    if m.max() > 1:
        m = m / 255.0

    if feather > 0:
        k = max(3, int(feather) | 1)
        m = cv2.GaussianBlur(m, (k, k), 0)

    return np.clip(m, 0.0, 1.0)


def build_visual_mask(
    h: int,
    w: int,
    mode: str = "rectangle",
    feather: int = 31,
) -> np.ndarray:
    """
    视觉修正用的 mask。

    注意：
    测色 mask 和视觉 mask 分开。
    测色 mask 用原图分割出来的胶块有效区域；
    视觉 mask 默认用整个 ROI 羽化，这样预览图不会只改中间一小块。
    """
    if mode == "rectangle":
        mask = np.ones((h, w), dtype=np.float32)

    elif mode == "ellipse":
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (max(2, int(w * 0.48)), max(2, int(h * 0.48)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        mask = mask.astype(np.float32) / 255.0

    else:
        raise ValueError(f"未知 visual mask mode: {mode}")

    return feather_mask(mask, feather)


def protect_light_weight(crop_lab_cv: np.ndarray) -> np.ndarray:
    """
    可选：高光和极暗处少修一点。
    默认脚本不开，因为为了让 alpha sweep 的 ΔE 更容易解释。
    """
    L = crop_lab_cv[:, :, 0].astype(np.float32)

    dark = np.clip((L - 15.0) / 40.0, 0.0, 1.0)
    bright = np.clip((245.0 - L) / 40.0, 0.0, 1.0)

    weight = dark * bright
    return np.clip(weight, 0.20, 1.0)


def build_background_mask(
    bgr: np.ndarray,
    target_colors: list[dict],
    bg_min_L: float,
    bg_max_saturation: float,
    feather: int = 31,
) -> np.ndarray:
    """
    背景 mask。
    默认本脚本 bg_scale=0，不启用背景修正。
    需要肉眼更干净时，可以加 --bg-scale 0.2 或 0.3。
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    L = lab[:, :, 0].astype(np.float32) * 100.0 / 255.0
    S = hsv[:, :, 1].astype(np.float32)

    mask = ((L >= bg_min_L) & (S <= bg_max_saturation)).astype(np.uint8)

    for item in target_colors:
        roi = item.get("roi_xyxy")
        if not roi:
            continue

        x1, y1, x2, y2 = map(int, roi)

        x1 = max(0, min(mask.shape[1], x1))
        x2 = max(0, min(mask.shape[1], x2))
        y1 = max(0, min(mask.shape[0], y1))
        y2 = max(0, min(mask.shape[0], y2))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return feather_mask(mask, feather)


def build_fixed_targets(
    *,
    original_bgr: np.ndarray,
    corrected_bgr: np.ndarray,
    target_colors: list[dict],
    trim_percent: float,
    visual_mask_mode: str,
    feather: int,
) -> list[dict]:
    """
    核心步骤：

    1. 用原图建立固定测色 mask。
       这样和 main.py 的逻辑更接近。

    2. 在 corrected 图上，用同一个 mask 测当前 corrected Lab。

    3. residual = standard Lab - corrected measured Lab。

    4. 后续所有 alpha 都复用同一套 mask 和 residual。
    """
    fixed_targets: list[dict] = []

    for item in target_colors:
        roi = item.get("roi_xyxy")
        standard = item.get("standard") or {}
        std_lab = standard.get("lab")

        if not roi or not std_lab:
            continue

        roi_tuple = tuple(map(int, roi))
        x1, y1, x2, y2 = roi_tuple

        x1 = max(0, min(corrected_bgr.shape[1] - 1, x1))
        x2 = max(1, min(corrected_bgr.shape[1], x2))
        y1 = max(0, min(corrected_bgr.shape[0] - 1, y1))
        y2 = max(1, min(corrected_bgr.shape[0], y2))

        if x2 <= x1 or y2 <= y1:
            continue

        roi_tuple = (x1, y1, x2, y2)

        sample_mask = build_glue_block_mask(
            original_bgr,
            roi_tuple,
            debug_path=None,
        )

        corrected_rgb = get_glue_block_representative_rgb(
            corrected_bgr,
            roi_tuple,
            mask=sample_mask,
            trim_percent=trim_percent,
        )

        corrected_lab = rgb_to_lab(corrected_rgb.reshape(1, 3))[0]
        std_lab_arr = np.asarray(std_lab, dtype=np.float64)

        residual_lab = std_lab_arr - corrected_lab

        h = y2 - y1
        w = x2 - x1

        visual_mask = build_visual_mask(
            h=h,
            w=w,
            mode=visual_mask_mode,
            feather=feather,
        )

        x1, y1, x2, y2 = roi_tuple
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r = int(min(x2 - x1, y2 - y1) * 0.45)

        default_visual_roi = circle_to_bbox(cx, cy, r)
        default_visual_mask = build_circle_visual_mask(r, feather)

        fixed_targets.append(
            {
                "index": item.get("index"),
                "code": standard.get("code"),
                "name": standard.get("name"),
                "roi_xyxy": roi_tuple,  # 测色 ROI
                "visual_roi_xyxy": default_visual_roi,  # 视觉 ROI
                "visual_circle": {"cx": cx, "cy": cy, "r": r},
                "standard_lab": std_lab_arr,
                "corrected_lab": corrected_lab,
                "residual_lab": residual_lab,
                "sample_mask": sample_mask,
                "visual_mask": default_visual_mask,
                "valid_mask_pixels": int((sample_mask > 0).sum()),
            }
        )

    return fixed_targets

def draw_roi_hint(img: np.ndarray, roi_xyxy: tuple[int, int, int, int], text: str) -> np.ndarray:
    show = img.copy()
    x1, y1, x2, y2 = map(int, roi_xyxy)
    cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        show,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return show

def build_circle_visual_mask(radius: int, feather: int) -> np.ndarray:
    """
    根据半径生成圆形羽化 mask。
    返回大小为 (2r+1, 2r+1) 的 float32 mask，范围 0~1。
    """
    r = max(3, int(radius))
    size = 2 * r + 1

    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = r
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    mask = (dist <= r).astype(np.float32)

    if feather > 0:
        k = max(3, int(feather) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    return np.clip(mask, 0.0, 1.0)


def clamp_circle_to_image(cx: int, cy: int, r: int, img_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    保证圆不会越出图像边界。
    """
    h, w = img_shape[:2]

    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))

    max_r = min(cx, cy, w - 1 - cx, h - 1 - cy)
    r = int(np.clip(r, 3, max_r if max_r >= 3 else 3))

    return cx, cy, r


def circle_to_bbox(cx: int, cy: int, r: int) -> tuple[int, int, int, int]:
    return (cx - r, cy - r, cx + r + 1, cy + r + 1)


def draw_circle_hint(
    img: np.ndarray,
    measure_roi: tuple[int, int, int, int],
    text: str,
    circle: tuple[int, int, int] | None = None,
) -> np.ndarray:
    show = img.copy()

    x1, y1, x2, y2 = map(int, measure_roi)
    cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if circle is not None:
        cx, cy, r = circle
        cv2.circle(show, (cx, cy), r, (0, 0, 255), 2)
        cv2.circle(show, (cx, cy), 3, (0, 0, 255), -1)

    cv2.putText(
        show,
        text,
        (max(10, x1), max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        show,
        "drag to draw circle | Enter confirm | R redraw | Esc keep old",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return show


def select_one_circle_interactively(
    img: np.ndarray,
    measure_roi: tuple[int, int, int, int],
    title: str = "draw visual circle",
) -> tuple[int, int, int] | None:
    """
    鼠标按下定圆心，拖动定半径，松开完成一个圆。
    Enter 确认，R 重画，Esc 返回 None（表示沿用旧的）。
    """
    base = img.copy()
    state = {
        "drawing": False,
        "center": None,
        "current": None,
        "final": None,
    }

    def mouse(event, x, y, flags, param):
        nonlocal base, state

        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["center"] = (x, y)
            state["current"] = (x, y, 3)

        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            cx, cy = state["center"]
            r = int(round(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5))
            r = max(3, r)
            state["current"] = (cx, cy, r)

        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            state["drawing"] = False
            cx, cy = state["center"]
            r = int(round(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5))
            r = max(3, r)
            state["final"] = (cx, cy, r)
            state["current"] = state["final"]

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse)

    while True:
        temp = base.copy()

        if state["current"] is not None:
            cx, cy, r = state["current"]
            cv2.circle(temp, (cx, cy), r, (0, 0, 255), 2)
            cv2.circle(temp, (cx, cy), 3, (0, 0, 255), -1)

        cv2.imshow(title, temp)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            cv2.destroyWindow(title)
            return state["final"]

        elif key in (ord("r"), ord("R")):
            state["drawing"] = False
            state["center"] = None
            state["current"] = None
            state["final"] = None

        elif key == 27:  # Esc
            cv2.destroyWindow(title)
            return None


def select_visual_circles_interactively(
    *,
    preview_bgr: np.ndarray,
    fixed_targets: list[dict],
    feather: int,
) -> list[dict]:
    """
    手动画圆。
    只影响视觉叠加区域，不影响 measure ROI / sample_mask / ΔE。
    """
    updated = []

    print("\n开始手动画 visual circle（只影响预览图）")
    print("操作：鼠标按下定圆心，拖动定半径，松开完成；Enter确认，R重画，Esc沿用旧ROI\n")

    for i, target in enumerate(fixed_targets, start=1):
        code = target.get("code", "")
        name = target.get("name", "")
        measure_roi = target["roi_xyxy"]

        hint = f"{i}/{len(fixed_targets)}  {code} {name}"
        show = draw_circle_hint(preview_bgr, measure_roi, hint, circle=None)

        circle = select_one_circle_interactively(show, measure_roi, title="draw visual circle")

        if circle is None:
            # 沿用旧 ROI 的中心和半径
            x1, y1, x2, y2 = measure_roi
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r = int(min(x2 - x1, y2 - y1) * 0.45)
        else:
            cx, cy, r = circle

        cx, cy, r = clamp_circle_to_image(cx, cy, r, preview_bgr.shape)

        visual_roi = circle_to_bbox(cx, cy, r)
        visual_mask = build_circle_visual_mask(r, feather)

        new_target = dict(target)
        new_target["visual_circle"] = {"cx": cx, "cy": cy, "r": r}
        new_target["visual_roi_xyxy"] = visual_roi
        new_target["visual_mask"] = visual_mask
        updated.append(new_target)

        print(f"[{i:03d}] {code} {name} circle = (cx={cx}, cy={cy}, r={r})")

    return updated


def save_visual_circles_json(path: Path, fixed_targets: list[dict]) -> None:
    rows = []

    for t in fixed_targets:
        circle = t.get("visual_circle")
        if circle is None:
            x1, y1, x2, y2 = t.get("visual_roi_xyxy", t["roi_xyxy"])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r = int(min(x2 - x1, y2 - y1) * 0.45)
            circle = {"cx": cx, "cy": cy, "r": r}

        rows.append(
            {
                "index": t.get("index"),
                "code": t.get("code"),
                "name": t.get("name"),
                "measure_roi_xyxy": list(map(int, t["roi_xyxy"])),
                "visual_circle": {
                    "cx": int(circle["cx"]),
                    "cy": int(circle["cy"]),
                    "r": int(circle["r"]),
                },
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_visual_circles_json(
    path: Path,
    fixed_targets: list[dict],
    preview_shape: tuple[int, int, int],
    feather: int,
) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_code = {}

    for row in rows:
        key = str(row.get("code") or row.get("index"))
        by_code[key] = row

    updated = []

    for target in fixed_targets:
        key = str(target.get("code") or target.get("index"))
        row = by_code.get(key)

        if row and row.get("visual_circle"):
            c = row["visual_circle"]
            cx, cy, r = int(c["cx"]), int(c["cy"]), int(c["r"])
        else:
            x1, y1, x2, y2 = target["roi_xyxy"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            r = int(min(x2 - x1, y2 - y1) * 0.45)

        cx, cy, r = clamp_circle_to_image(cx, cy, r, preview_shape)

        visual_roi = circle_to_bbox(cx, cy, r)
        visual_mask = build_circle_visual_mask(r, feather)

        new_target = dict(target)
        new_target["visual_circle"] = {"cx": cx, "cy": cy, "r": r}
        new_target["visual_roi_xyxy"] = visual_roi
        new_target["visual_mask"] = visual_mask
        updated.append(new_target)

    return updated


def select_visual_rois_interactively(
    *,
    preview_bgr: np.ndarray,
    fixed_targets: list[dict],
    visual_mask_mode: str,
    feather: int,
) -> list[dict]:
    """
    交互式重新选择视觉 ROI。
    只影响 preview 叠加区域，不影响 measure ROI / ΔE。
    """
    updated = []

    print("\n开始手动选择 visual ROI（只影响预览图）")
    print("操作：框选更贴合胶块的区域，Enter确认，Esc跳过沿用原 ROI\n")

    for i, target in enumerate(fixed_targets, start=1):
        code = target.get("code", "")
        name = target.get("name", "")
        measure_roi = target["roi_xyxy"]

        hint = f"{i}/{len(fixed_targets)}  {code} {name}"
        show = draw_roi_hint(preview_bgr, measure_roi, hint)

        cv2.namedWindow("select visual roi", cv2.WINDOW_NORMAL)
        cv2.imshow("select visual roi", show)

        x, y, w, h = cv2.selectROI(
            "select visual roi",
            show,
            showCrosshair=False,
            fromCenter=False,
        )
        cv2.destroyWindow("select visual roi")

        if w <= 0 or h <= 0:
            # 跳过：沿用原 ROI
            visual_roi = measure_roi
        else:
            visual_roi = (int(x), int(y), int(x + w), int(y + h))

        vx1, vy1, vx2, vy2 = visual_roi
        vh = max(1, vy2 - vy1)
        vw = max(1, vx2 - vx1)

        visual_mask = build_visual_mask(
            h=vh,
            w=vw,
            mode=visual_mask_mode,
            feather=feather,
        )

        new_target = dict(target)
        new_target["visual_roi_xyxy"] = visual_roi
        new_target["visual_mask"] = visual_mask
        updated.append(new_target)

        print(f"[{i:03d}] {code} {name} visual_roi = {visual_roi}")

    return updated


def save_visual_rois_json(path: Path, fixed_targets: list[dict]) -> None:
    rows = []

    for t in fixed_targets:
        rows.append(
            {
                "index": t.get("index"),
                "code": t.get("code"),
                "name": t.get("name"),
                "measure_roi_xyxy": list(map(int, t["roi_xyxy"])),
                "visual_roi_xyxy": list(map(int, t.get("visual_roi_xyxy", t["roi_xyxy"]))),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_visual_rois_json(
    path: Path,
    fixed_targets: list[dict],
    visual_mask_mode: str,
    feather: int,
) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))

    by_code = {}
    for row in rows:
        key = str(row.get("code") or row.get("index"))
        by_code[key] = row

    updated = []

    for target in fixed_targets:
        key = str(target.get("code") or target.get("index"))
        row = by_code.get(key)

        if row and row.get("visual_roi_xyxy"):
            vx1, vy1, vx2, vy2 = map(int, row["visual_roi_xyxy"])
            visual_roi = (vx1, vy1, vx2, vy2)
        else:
            visual_roi = target["roi_xyxy"]

        vh = max(1, visual_roi[3] - visual_roi[1])
        vw = max(1, visual_roi[2] - visual_roi[0])

        visual_mask = build_visual_mask(
            h=vh,
            w=vw,
            mode=visual_mask_mode,
            feather=feather,
        )

        new_target = dict(target)
        new_target["visual_roi_xyxy"] = visual_roi
        new_target["visual_mask"] = visual_mask
        updated.append(new_target)

    return updated

def apply_residual_to_corrected(
    *,
    original_bgr: np.ndarray,
    corrected_bgr: np.ndarray,
    fixed_targets: list[dict],
    alpha: float,
    ab_scale: float,
    l_scale: float,
    protect_extreme_light: bool,
    bg_alpha: float,
    target_colors: list[dict],
    bg_min_L: float,
    bg_max_saturation: float,
    use_visual_series_rules: bool,
    rule_strength: float,
    glue_l_original_mix: float,
    default_b_scale: float,
    b_pos_cap: float | None,
    b_neg_cap: float | None,
) -> np.ndarray:
    """
    在 corrected 图上叠加 residual，但加入两类视觉保护：

    1. b 通道按色系缩放/限幅，防止 W033-W064、W053 等过黄。
    2. 胶块 L 通道可按色系混回 original 图，防止深色/灰白被 corrected 图提得过浅。

    注意：这只影响 preview 图，不改变测色 ROI，也不改变 report.json。
    """
    lab = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    original_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    if bg_alpha > 0:
        bg_mask = build_background_mask(
            corrected_bgr,
            target_colors=target_colors,
            bg_min_L=bg_min_L,
            bg_max_saturation=bg_max_saturation,
            feather=31,
        )
        lab[:, :, 1] = lab[:, :, 1] + bg_mask * bg_alpha * (128.0 - lab[:, :, 1])
        lab[:, :, 2] = lab[:, :, 2] + bg_mask * bg_alpha * (128.0 - lab[:, :, 2])

    for target in fixed_targets:
        x1, y1, x2, y2 = target.get("visual_roi_xyxy", target["roi_xyxy"])

        crop = lab[y1:y2, x1:x2]
        orig_crop = original_lab[y1:y2, x1:x2]

        eff_dL, eff_da, eff_db, rule = effective_lab_delta_for_target(
            target,
            alpha=float(alpha),
            ab_scale=float(ab_scale),
            l_scale=float(l_scale),
            enable_rules=bool(use_visual_series_rules),
            rule_strength=float(rule_strength),
            default_l_original_mix=float(glue_l_original_mix),
            default_b_scale=float(default_b_scale),
            default_b_pos_cap=b_pos_cap,
            default_b_neg_cap=b_neg_cap,
        )

        # 标准 Lab L 残差要换算到 OpenCV Lab 的 0~255 尺度；a/b 残差不用加 128，因为这是“差值”。
        dL_cv = eff_dL * 255.0 / 100.0
        da_cv = eff_da
        db_cv = eff_db

        m = target["visual_mask"].astype(np.float32)
        if m.max() > 1:
            m = m / 255.0

        if protect_extreme_light:
            m = m * protect_light_weight(crop)

        # 先做残差修正。
        crop[:, :, 0] = crop[:, :, 0] + m * dL_cv
        crop[:, :, 1] = crop[:, :, 1] + m * da_cv
        crop[:, :, 2] = crop[:, :, 2] + m * db_cv

        # 再把 L 通道按规则混回原图，解决“深色/灰白被校得过浅”。
        # mix=0: L 保持 corrected/residual；mix=1: L 完全用 original。
        l_mix = float(np.clip(rule["l_original_mix"], 0.0, 1.0))
        if l_mix > 0:
            wm = m * l_mix
            crop[:, :, 0] = crop[:, :, 0] * (1.0 - wm) + orig_crop[:, :, 0] * wm

        lab[y1:y2, x1:x2] = crop

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def evaluate_candidate(
    candidate_bgr: np.ndarray,
    fixed_targets: list[dict],
    trim_percent: float,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    delta_es: list[float] = []

    for target in fixed_targets:
        roi = target["roi_xyxy"]
        sample_mask = target["sample_mask"]
        std_lab = target["standard_lab"]

        rgb = get_glue_block_representative_rgb(
            candidate_bgr,
            roi,
            mask=sample_mask,
            trim_percent=trim_percent,
        )

        measured_lab = rgb_to_lab(rgb.reshape(1, 3))[0]

        de = float(
            delta_e_2000(
                measured_lab.reshape(1, 3),
                std_lab.reshape(1, 3),
            )[0]
        )

        delta_es.append(de)

        residual_after = std_lab - measured_lab

        rows.append(
            {
                "index": target["index"],
                "code": target["code"],
                "name": target["name"],
                "roi_x1": roi[0],
                "roi_y1": roi[1],
                "roi_x2": roi[2],
                "roi_y2": roi[3],
                "valid_mask_pixels": target["valid_mask_pixels"],
                "standard_L": float(std_lab[0]),
                "standard_a": float(std_lab[1]),
                "standard_b": float(std_lab[2]),
                "measured_L": float(measured_lab[0]),
                "measured_a": float(measured_lab[1]),
                "measured_b": float(measured_lab[2]),
                "deltaE2000": de,
                "residual_L_after": float(residual_after[0]),
                "residual_a_after": float(residual_after[1]),
                "residual_b_after": float(residual_after[2]),
            }
        )

    return rows, stat_pack(delta_es)


def save_per_target_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "index",
        "code",
        "name",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "valid_mask_pixels",
        "standard_L",
        "standard_a",
        "standard_b",
        "measured_L",
        "measured_a",
        "measured_b",
        "deltaE2000",
        "residual_L_after",
        "residual_a_after",
        "residual_b_after",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "alpha",
        "alpha_ab",
        "alpha_L",
        "bg_alpha",
        "mean_deltaE",
        "median_deltaE",
        "p95_deltaE",
        "max_deltaE",
        "std_deltaE",
        "preview_file",
        "per_target_csv",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_metric_plot(path: Path, summary_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    alphas = [r["alpha"] for r in summary_rows]
    means = [r["mean_deltaE"] for r in summary_rows]
    p95s = [r["p95_deltaE"] for r in summary_rows]
    maxs = [r["max_deltaE"] for r in summary_rows]

    plt.figure(figsize=(10, 5))
    plt.plot(alphas, means, marker="o", label="mean ΔE")
    plt.plot(alphas, p95s, marker="s", label="p95 ΔE")
    plt.plot(alphas, maxs, marker="^", label="max ΔE")
    plt.xlabel("alpha")
    plt.ylabel("ΔE2000")
    plt.title("Corrected-base residual alpha sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    h, w = img.shape[:2]
    top = 52
    out = np.full((h + top, w, 3), 245, dtype=np.uint8)
    out[top:, :] = img

    cv2.putText(
        out,
        text[:120],
        (12, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return out


def resize_keep(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(
        img,
        (width, max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def make_sheet(
    images: list[np.ndarray],
    labels: list[str],
    cols: int,
    thumb_width: int,
) -> np.ndarray:
    thumbs = [
        put_label(resize_keep(img, thumb_width), label)
        for img, label in zip(images, labels)
    ]

    gap = 18

    tw = max(t.shape[1] for t in thumbs)
    th = max(t.shape[0] for t in thumbs)

    rows = int(np.ceil(len(thumbs) / cols))

    sheet = np.full(
        (
            rows * th + (rows + 1) * gap,
            cols * tw + (cols + 1) * gap,
            3,
        ),
        250,
        dtype=np.uint8,
    )

    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)

        y = gap + r * (th + gap)
        x = gap + c * (tw + gap)

        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb

    return sheet

def make_full_visual_mask(
    image_shape: tuple[int, int, int],
    fixed_targets: list[dict],
) -> np.ndarray:
    """
    把 128 个胶块的 visual_mask 合成一张整图 mask。
    mask=1 的地方用修正后的胶块；
    mask=0 的地方用原图背景。
    """
    h, w = image_shape[:2]
    full_mask = np.zeros((h, w), dtype=np.float32)

    for target in fixed_targets:
        x1, y1, x2, y2 = target.get("visual_roi_xyxy", target["roi_xyxy"])
        m = target["visual_mask"].astype(np.float32)

        if m.max() > 1:
            m = m / 255.0

        mh, mw = m.shape[:2]
        rh = y2 - y1
        rw = x2 - x1

        if mh != rh or mw != rw:
            m = cv2.resize(m, (rw, rh), interpolation=cv2.INTER_LINEAR)

        full_mask[y1:y2, x1:x2] = np.maximum(
            full_mask[y1:y2, x1:x2],
            m,
        )

    return np.clip(full_mask, 0.0, 1.0)


def composite_with_original_background(
    *,
    original_bgr: np.ndarray,
    corrected_bgr: np.ndarray,
    glue_bgr: np.ndarray,
    fixed_targets: list[dict],
    background_mode: str,
    background_mix: float,
) -> np.ndarray:
    """
    胶块区域用 glue_bgr；
    背景区域根据 background_mode 选择。

    background_mode:
        original  = 背景完全用原图，最自然
        blend     = 背景 = 原图*(1-mix) + corrected*mix
        corrected = 背景也用校正图
    """
    if background_mode == "original":
        bg = original_bgr.astype(np.float32)

    elif background_mode == "blend":
        mix = float(np.clip(background_mix, 0.0, 1.0))
        bg = (
            original_bgr.astype(np.float32) * (1.0 - mix)
            + corrected_bgr.astype(np.float32) * mix
        )

    elif background_mode == "corrected":
        bg = corrected_bgr.astype(np.float32)

    else:
        raise ValueError(f"未知 background_mode: {background_mode}")

    mask = make_full_visual_mask(original_bgr.shape, fixed_targets)
    mask3 = mask[:, :, None]

    out = bg * (1.0 - mask3) + glue_bgr.astype(np.float32) * mask3
    return np.clip(out, 0, 255).astype(np.uint8)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep alpha on corrected image using Lab residual correction."
    )

    parser.add_argument(
        "--report",
        required=True,
        help="main.py 输出的 report.json，例如 output_128/report.json",
    )

    parser.add_argument(
        "--photo",
        default=None,
        help="原图路径。用于建立固定测色 mask。不填则从 report 里读取。",
    )

    parser.add_argument(
        "--corrected",
        default=None,
        help="校正图路径。不填则默认使用 report 同目录下的 02_corrected.png。",
    )

    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.1,
        help="alpha 步长，默认 0.1。",
    )

    parser.add_argument(
        "--alpha-list",
        default=None,
        help='手动指定 alpha 列表，例如 "0,0.1,0.2,0.3,0.5"。指定后忽略 alpha-step。',
    )

    parser.add_argument(
        "--ab-scale",
        type=float,
        default=1.0,
        help="a/b 残差叠加强度倍率。默认 1.0，即 alpha_ab=alpha。",
    )

    parser.add_argument(
        "--l-scale",
        type=float,
        default=0.0,
        help="L 残差叠加强度倍率。默认 0，先不动亮度。若 ΔE 主要来自偏暗，可试 0.05 或 0.1。",
    )

    parser.add_argument(
        "--bg-scale",
        type=float,
        default=0.0,
        help="背景中性化强度倍率。默认 0，不动背景。需要视觉更干净可试 0.2 或 0.3。",
    )

    parser.add_argument(
        "--visual-mask",
        choices=["rectangle", "ellipse"],
        default="rectangle",
        help="视觉修正 mask。默认 rectangle。",
    )

    parser.add_argument(
        "--feather",
        type=int,
        default=31,
        help="ROI 视觉修正边缘羽化。默认 31。",
    )

    parser.add_argument(
        "--protect-extreme-light",
        action="store_true",
        help="开启后高光/极暗区域少修。更自然，但 ΔE sweep 可能不够线性。",
    )

    parser.add_argument(
        "--bg-min-L",
        type=float,
        default=45.0,
        help="背景 mask 最低 L。仅 bg-scale > 0 时使用。",
    )

    parser.add_argument(
        "--bg-max-saturation",
        type=float,
        default=85.0,
        help="背景 mask 最高饱和度。仅 bg-scale > 0 时使用。",
    )

    parser.add_argument(
        "--trim-percent",
        type=float,
        default=10.0,
        help="ROI 代表色 trimmed mean 百分比，默认 10。",
    )

    parser.add_argument(
        "--thumb-width",
        type=int,
        default=360,
        help="总览图中每张预览图宽度。",
    )

    parser.add_argument(
        "--background-mode",
        choices=["original", "blend", "corrected"],
        default="original",
        help="最终图背景模式。original=背景用原图；blend=原图和校正图混合；corrected=背景也用校正图。",
    )

    parser.add_argument(
        "--background-mix",
        type=float,
        default=0.15,
        help="background-mode=blend 时使用。0=全原图背景，1=全 corrected 背景。",
    )
    parser.add_argument(
        "--manual-visual-roi",
        action="store_true",
        help="手动重新选择视觉叠加用 ROI。只影响 preview 图，不影响 ΔE 测量。",
    )

    parser.add_argument(
        "--visual-roi-file",
        default=None,
        help="保存/复用手动视觉 ROI 的 json 文件。默认保存在输出目录 visual_rois_manual.json",
    )
    parser.add_argument(
        "--manual-visual-circle",
        action="store_true",
        help="手动画圆形视觉区域。只影响 preview 图，不影响 ΔE 测量。",
    )

    parser.add_argument(
        "--visual-circle-file",
        default=None,
        help="保存/复用手动画圆结果的 json 文件。默认保存在输出目录 visual_circles_manual.json",
    )


    parser.add_argument(
        "--use-visual-series-rules",
        action="store_true",
        help="启用按 W 编号/色系的视觉保护规则：限制 W033-W064 过黄、深色过浅、W053 暖白异常。",
    )

    parser.add_argument(
        "--rule-strength",
        type=float,
        default=1.0,
        help="视觉规则强度，0=不用规则，1=完整规则。建议先试 0.7 或 1.0。",
    )

    parser.add_argument(
        "--glue-l-original-mix",
        type=float,
        default=0.0,
        help="全局胶块 L 通道回原图比例。0=用 corrected L，1=用 original L。启用色系规则时各组会自动覆盖/插值。",
    )

    parser.add_argument(
        "--b-scale",
        type=float,
        default=1.0,
        help="全局 b 残差倍率。小于 1 会整体减少变黄/变蓝幅度。",
    )

    parser.add_argument(
        "--b-pos-cap",
        type=float,
        default=None,
        help="全局限制 b 正向残差，即最多变黄多少 Lab b。例：--b-pos-cap 2。",
    )

    parser.add_argument(
        "--b-neg-cap",
        type=float,
        default=None,
        help="全局限制 b 负向残差，即最多变蓝多少 Lab b。例：--b-neg-cap 3。",
    )

    args = parser.parse_args()

    report_path = Path(args.report)

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    target_colors = report.get("target_colors") or []

    if not target_colors:
        raise RuntimeError("report.json 里没有 target_colors，无法 sweep。")

    photo_path = resolve_path(
        args.photo or report.get("input", {}).get("photo"),
        report_path,
    )

    corrected_path = resolve_path(
        args.corrected,
        report_path,
    )

    if corrected_path is None:
        corrected_path = report_path.parent / "02_corrected.png"

    if photo_path is None or not photo_path.exists():
        raise FileNotFoundError(f"找不到原图：{photo_path}")

    if corrected_path is None or not corrected_path.exists():
        raise FileNotFoundError(
            f"找不到 corrected 图：{corrected_path}。"
            f"请确认 main.py 已经跑完，并生成 02_corrected.png。"
        )

    print("原图：", photo_path)
    print("corrected 底图：", corrected_path)

    original_bgr = imread_unicode(photo_path)
    corrected_bgr = imread_unicode(corrected_path)

    if original_bgr.shape[:2] != corrected_bgr.shape[:2]:
        raise RuntimeError(
            "原图和 corrected 图尺寸不一致，不能复用同一套 ROI/mask。"
        )

    out_dir = report_path.parent / "corrected_residual_alpha_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n建立固定 ROI/mask，并计算 corrected 图 residual ...")

    fixed_targets = build_fixed_targets(
        original_bgr=original_bgr,
        corrected_bgr=corrected_bgr,
        target_colors=target_colors,
        trim_percent=args.trim_percent,
        visual_mask_mode=args.visual_mask,
        feather=args.feather,
    )
    visual_roi_file = Path(args.visual_roi_file) if args.visual_roi_file else (out_dir / "visual_rois_manual.json")
    visual_circle_file = Path(args.visual_circle_file) if args.visual_circle_file else (out_dir / "visual_circles_manual.json")

    # 重要：visual circle 和 visual ROI 只能二选一。
    # 之前的 bug 是：先加载/保存了手动画圆，后面又自动加载 visual_rois_manual.json，
    # 于是圆形 mask 被旧的方框 ROI 覆盖，最终看起来仍然是大方框渲染。
    if args.manual_visual_circle:
        fixed_targets = select_visual_circles_interactively(
            preview_bgr=original_bgr,
            fixed_targets=fixed_targets,
            feather=args.feather,
        )
        save_visual_circles_json(visual_circle_file, fixed_targets)
        print("已保存手动画圆结果：", visual_circle_file)
        print("当前视觉区域模式：manual_circle_saved")

    elif args.manual_visual_roi:
        fixed_targets = select_visual_rois_interactively(
            preview_bgr=original_bgr,
            fixed_targets=fixed_targets,
            visual_mask_mode=args.visual_mask,
            feather=args.feather,
        )
        save_visual_rois_json(visual_roi_file, fixed_targets)
        print("已保存手动 visual ROI：", visual_roi_file)
        print("当前视觉区域模式：manual_roi_saved")

    elif visual_circle_file.exists():
        fixed_targets = load_visual_circles_json(
            visual_circle_file,
            fixed_targets=fixed_targets,
            preview_shape=original_bgr.shape,
            feather=args.feather,
        )
        print("已加载已有画圆结果：", visual_circle_file)
        print("当前视觉区域模式：manual_circle_loaded")

    elif visual_roi_file.exists():
        fixed_targets = load_visual_rois_json(
            visual_roi_file,
            fixed_targets=fixed_targets,
            visual_mask_mode=args.visual_mask,
            feather=args.feather,
        )
        print("已加载已有 visual ROI：", visual_roi_file)
        print("当前视觉区域模式：manual_roi_loaded")

    else:
        print("未找到手动画圆/ROI文件，使用默认自动圆形视觉区域。")
        print("当前视觉区域模式：default_auto_circle")
    if not fixed_targets:
        raise RuntimeError("没有有效 fixed_targets。")

    print(f"有效目标数量：{len(fixed_targets)}")

    alpha_list = parse_alpha_list(args.alpha_list, args.alpha_step)

    print(f"\n开始 sweep alpha，共 {len(alpha_list)} 个：{alpha_list}")
    print(f"ab_scale={args.ab_scale}, l_scale={args.l_scale}, bg_scale={args.bg_scale}")
    print(
        f"visual_series_rules={args.use_visual_series_rules}, "
        f"rule_strength={args.rule_strength}, "
        f"glue_l_original_mix={args.glue_l_original_mix}, "
        f"b_scale={args.b_scale}, b_pos_cap={args.b_pos_cap}, b_neg_cap={args.b_neg_cap}"
    )

    summary_rows: list[dict] = []
    sheet_images: list[np.ndarray] = []
    sheet_labels: list[str] = []

    for alpha in alpha_list:
        bg_alpha = float(alpha) * float(args.bg_scale)

        glue_candidate = apply_residual_to_corrected(
            original_bgr=original_bgr,
            corrected_bgr=corrected_bgr,
            fixed_targets=fixed_targets,
            alpha=float(alpha),
            ab_scale=args.ab_scale,
            l_scale=args.l_scale,
            protect_extreme_light=args.protect_extreme_light,
            bg_alpha=bg_alpha,
            target_colors=target_colors,
            bg_min_L=args.bg_min_L,
            bg_max_saturation=args.bg_max_saturation,
            use_visual_series_rules=args.use_visual_series_rules,
            rule_strength=args.rule_strength,
            glue_l_original_mix=args.glue_l_original_mix,
            default_b_scale=args.b_scale,
            b_pos_cap=args.b_pos_cap,
            b_neg_cap=args.b_neg_cap,
        )

        candidate = composite_with_original_background(
            original_bgr=original_bgr,
            corrected_bgr=corrected_bgr,
            glue_bgr=glue_candidate,
            fixed_targets=fixed_targets,
            background_mode=args.background_mode,
            background_mix=args.background_mix,
        )

        preview_path = out_dir / f"preview_alpha_{alpha:.2f}.png"
        imwrite_unicode(preview_path, candidate)

        rows, stats = evaluate_candidate(
            candidate_bgr=candidate,
            fixed_targets=fixed_targets,
            trim_percent=args.trim_percent,
        )

        per_target_csv = out_dir / f"targets_alpha_{alpha:.2f}.csv"
        save_per_target_csv(per_target_csv, rows)

        rules_csv = out_dir / f"visual_rules_alpha_{alpha:.2f}.csv"
        save_visual_rules_csv(
            rules_csv,
            fixed_targets,
            alpha=float(alpha),
            ab_scale=args.ab_scale,
            l_scale=args.l_scale,
            enable_rules=args.use_visual_series_rules,
            rule_strength=args.rule_strength,
            default_l_original_mix=args.glue_l_original_mix,
            default_b_scale=args.b_scale,
            default_b_pos_cap=args.b_pos_cap,
            default_b_neg_cap=args.b_neg_cap,
        )

        summary_row = {
            "alpha": float(alpha),
            "alpha_ab": float(alpha) * float(args.ab_scale),
            "alpha_L": float(alpha) * float(args.l_scale),
            "bg_alpha": bg_alpha,
            "mean_deltaE": stats["mean"],
            "median_deltaE": stats["median"],
            "p95_deltaE": stats["p95"],
            "max_deltaE": stats["max"],
            "std_deltaE": stats["std"],
            "preview_file": str(preview_path),
            "per_target_csv": str(per_target_csv),
        }

        summary_rows.append(summary_row)

        label = (
            f"a={alpha:.2f} "
            f"mean={stats['mean']:.2f} "
            f"p95={stats['p95']:.2f} "
            f"max={stats['max']:.2f}"
        )

        sheet_images.append(candidate)
        sheet_labels.append(label)

        print(
            f"alpha={alpha:.2f} | "
            f"mean={stats['mean']:.3f} | "
            f"p95={stats['p95']:.3f} | "
            f"max={stats['max']:.3f}"
        )

    save_summary_csv(out_dir / "alpha_sweep_summary.csv", summary_rows)
    save_metric_plot(out_dir / "alpha_sweep_metrics.png", summary_rows)

    sheet = make_sheet(
        images=sheet_images,
        labels=sheet_labels,
        cols=3,
        thumb_width=args.thumb_width,
    )

    sheet_path = out_dir / "alpha_sweep_contact_sheet.png"
    imwrite_unicode(sheet_path, sheet)

    sorted_by_mean = sorted(
        summary_rows,
        key=lambda r: (
            r["mean_deltaE"],
            r["p95_deltaE"],
            r["max_deltaE"],
        ),
    )

    result = {
        "report": str(report_path),
        "photo": str(photo_path),
        "corrected": str(corrected_path),
        "output_dir": str(out_dir),
        "settings": {
            "alpha_list": alpha_list,
            "ab_scale": args.ab_scale,
            "l_scale": args.l_scale,
            "bg_scale": args.bg_scale,
            "visual_mask": args.visual_mask,
            "feather": args.feather,
            "protect_extreme_light": args.protect_extreme_light,
            "trim_percent": args.trim_percent,
            "use_visual_series_rules": args.use_visual_series_rules,
            "rule_strength": args.rule_strength,
            "glue_l_original_mix": args.glue_l_original_mix,
            "b_scale": args.b_scale,
            "b_pos_cap": args.b_pos_cap,
            "b_neg_cap": args.b_neg_cap,
        },
        "summary_rows": summary_rows,
        "recommended_by_metrics": sorted_by_mean[:5],
        "note": (
            "本脚本是在 02_corrected.png 上做 residual alpha sweep。"
            "alpha=0 应接近 main.py 的校正后 ΔE。"
            "最终 alpha 不建议只看 mean ΔE，需结合 contact_sheet 肉眼判断。"
        ),
    }

    (out_dir / "alpha_sweep_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n==== 完成 ====")
    print("输出目录：", out_dir)
    print("总览图：", sheet_path)
    print("指标曲线：", out_dir / "alpha_sweep_metrics.png")
    print("汇总表：", out_dir / "alpha_sweep_summary.csv")
    print("\n按 mean/p95/max 排序的前 5 个：")

    for i, row in enumerate(sorted_by_mean[:5], start=1):
        print(
            f"{i}. alpha={row['alpha']:.2f}, "
            f"mean={row['mean_deltaE']:.3f}, "
            f"p95={row['p95_deltaE']:.3f}, "
            f"max={row['max_deltaE']:.3f}"
        )


if __name__ == "__main__":
    main()