# -*- coding: utf-8 -*-
"""
用途：
    render_from_combined_mapping.py

    作用：
        对“没有标准 CSV 的胶块大板”，
        利用已经训练好的 combined visual_mapping_T，
        把 corrected 图自动渲染成一张“更接近肉眼库”的初始 visual preview。

    它解决的问题：
        manual_visual_dataset_builder.py 第一次跑完后，只会得到：
            02_corrected.png
            corrected_samples.csv
            visual_circles_manual.json
        这一步只是“采样 corrected_lab”，不会明显改颜色。

        本脚本补上中间这一步：
            corrected_lab --(combined visual_mapping_T)--> predicted visual_lab
        然后把这个映射结果渲染回每个胶块 circle 区域，输出：
            03_auto_visual_preview.png

    推荐流程：
        1) 先跑 manual_visual_dataset_builder.py 第一次
        2) 再跑本脚本，生成自动视觉预览图
        3) 如果需要，再对这张图做轻微人工微调
        4) 再回到 builder，用 --final-preview 采样 visual_training_samples.csv

输入：
    --corrected         builder 输出的 02_corrected.png
    --samples-csv       builder 输出的 corrected_samples.csv
    --circle-file       builder 输出的 visual_circles_manual.json
    --mapping           output_combined/visual_mapping_T_poly2/visual_mapping_T.json
    --out               输出自动视觉图
可选：
    --background-mode   corrected / original
    --original-photo    当 background-mode=original 时需要提供
    --feather           圆边缘羽化
    --display-l-offset
    --display-a-offset
    --display-b-offset
    --display-chroma-scale
    --alpha             0~1，预测改动强度
    --keep-texture      保持圆内原纹理，只平移 Lab；默认开启
    --flat-fill         用预测的纯色直接填圆，不保留纹理

典型运行：
    python render_from_combined_mapping.py ^
      --corrected output_no_std_dataset/02_corrected.png ^
      --samples-csv output_no_std_dataset/corrected_samples.csv ^
      --circle-file output_no_std_dataset/visual_circles_manual.json ^
      --mapping output_combined/visual_mapping_T_poly2/visual_mapping_T.json ^
      --out output_no_std_dataset/03_auto_visual_preview.png

如果想背景保留原图：
    python render_from_combined_mapping.py ^
      --corrected output_no_std_dataset/02_corrected.png ^
      --samples-csv output_no_std_dataset/corrected_samples.csv ^
      --circle-file output_no_std_dataset/visual_circles_manual.json ^
      --mapping output_combined/visual_mapping_T_poly2/visual_mapping_T.json ^
      --background-mode original ^
      --original-photo data/no_std/board.jpg ^
      --out output_no_std_dataset/03_auto_visual_preview.png

如果想再微暖一点、少一点灰：
    python render_from_combined_mapping.py ^
      --corrected output_no_std_dataset/02_corrected.png ^
      --samples-csv output_no_std_dataset/corrected_samples.csv ^
      --circle-file output_no_std_dataset/visual_circles_manual.json ^
      --mapping output_combined/visual_mapping_T_poly2/visual_mapping_T.json ^
      --out output_no_std_dataset/03_auto_visual_preview_tweak.png ^
      --display-b-offset 1.0 ^
      --display-chroma-scale 1.05
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Unicode 路径读写
# ============================================================

def imread_unicode(path: str | Path):
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return img


def imwrite_unicode(path: str | Path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")
    buf.tofile(str(path))


# ============================================================
# 颜色空间：sRGB / XYZ / Lab
# ============================================================

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


def srgb_to_linear(rgb_255):
    rgb = np.asarray(rgb_255, dtype=np.float64) / 255.0
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(rgb_lin):
    rgb_lin = np.clip(np.asarray(rgb_lin, dtype=np.float64), 0.0, 1.0)
    srgb = np.where(
        rgb_lin <= 0.0031308,
        rgb_lin * 12.92,
        1.055 * np.power(rgb_lin, 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb * 255.0, 0, 255)


def rgb_to_lab(rgb_255):
    rgb_lin = srgb_to_linear(rgb_255)
    xyz = rgb_lin @ SRGB_TO_XYZ.T
    xyz_scaled = xyz / D65

    eps = 216 / 24389
    kappa = 24389 / 27

    f = np.where(xyz_scaled > eps, np.cbrt(xyz_scaled), (kappa * xyz_scaled + 16) / 116)

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

    def invf(t):
        t3 = t ** 3
        return np.where(t3 > eps, t3, (116 * t - 16) / kappa)

    x = D65[0] * invf(fx)
    y = D65[1] * invf(fy)
    z = D65[2] * invf(fz)

    xyz = np.stack([x, y, z], axis=-1)
    rgb_lin = xyz @ XYZ_TO_SRGB.T
    return linear_to_srgb(rgb_lin)


# ============================================================
# 读取 circle
# ============================================================

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


def load_circles(circle_file: Path, samples_df: pd.DataFrame):
    if not circle_file.exists():
        raise FileNotFoundError(f"找不到 circle 文件：{circle_file}")

    data = json.loads(circle_file.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "circles" in data:
        data = data["circles"]

    circles = []

    # list：按顺序
    if isinstance(data, list):
        for i in range(len(samples_df)):
            c = normalize_circle(data[i]) if i < len(data) else None
            circles.append(c)
        return circles

    # dict：按 code / index
    if isinstance(data, dict):
        for _, row in samples_df.iterrows():
            code = str(row.get("code", "")).strip()
            idx = str(row.get("index", "")).strip()

            hit = None
            for key in [code, code.upper(), idx, str(int(float(idx))) if idx else ""]:
                if key and key in data:
                    hit = normalize_circle(data[key])
                    break
            circles.append(hit)
        return circles

    raise RuntimeError("无法解析 circle 文件。")


# ============================================================
# 读取 / 应用 mapping_T
# ============================================================

def build_lab_features(lab_arr: np.ndarray, feature_mode: str, feature_names: list[str] | None = None):
    """
    构造 Lab 特征。

    关键修复：
        你的 visual_mapping_T.json 里 feature_names 的顺序是：
            1, L_corr, a_corr, b_corr,
            L_corr^2, a_corr^2, b_corr^2,
            L_corr*a_corr, L_corr*b_corr, a_corr*b_corr

        也就是常数项 1 在最前面。

        之前脚本错用了：
            L, a, b, L^2, a^2, b^2, L*a, L*b, a*b, 1

        常数项位置错了，会直接把系数乘乱，导致 pred_visual_b 爆到几千甚至上万。
    """
    X = np.asarray(lab_arr, dtype=np.float64).reshape(-1, 3)
    L = X[:, 0:1]
    a = X[:, 1:2]
    b = X[:, 2:3]
    ones = np.ones_like(L)

    # 如果 mapping json 提供了 feature_names，就严格按它的顺序构造
    if feature_names:
        cols = []
        for name in feature_names:
            key = str(name).strip()

            if key in ["1", "bias", "const", "constant"]:
                cols.append(ones)
            elif key in ["L_corr", "L", "corrected_L"]:
                cols.append(L)
            elif key in ["a_corr", "a", "corrected_a"]:
                cols.append(a)
            elif key in ["b_corr", "b", "corrected_b"]:
                cols.append(b)
            elif key in ["L_corr^2", "L^2", "corrected_L^2"]:
                cols.append(L * L)
            elif key in ["a_corr^2", "a^2", "corrected_a^2"]:
                cols.append(a * a)
            elif key in ["b_corr^2", "b^2", "corrected_b^2"]:
                cols.append(b * b)
            elif key in ["L_corr*a_corr", "L*a", "corrected_L*corrected_a"]:
                cols.append(L * a)
            elif key in ["L_corr*b_corr", "L*b", "corrected_L*corrected_b"]:
                cols.append(L * b)
            elif key in ["a_corr*b_corr", "a*b", "corrected_a*corrected_b"]:
                cols.append(a * b)
            else:
                raise RuntimeError(f"无法识别 mapping feature_name：{key}")

        return np.concatenate(cols, axis=1)

    # 没有 feature_names 时，按 train_visual_mapping_T.py 当前约定：bias 在前
    if feature_mode in ["linear", "linear_bias", "affine"]:
        return np.concatenate([ones, L, a, b], axis=1)

    if feature_mode == "poly2":
        return np.concatenate([ones, L, a, b, L * L, a * a, b * b, L * a, L * b, a * b], axis=1)

    if feature_mode == "root_poly2":
        def signed_root(v):
            return np.sign(v) * np.sqrt(np.abs(v) + 1e-12)

        return np.concatenate(
            [
                ones, L, a, b,
                signed_root(L * a),
                signed_root(L * b),
                signed_root(a * b),
            ],
            axis=1,
        )

    raise ValueError(f"不支持的 feature_mode：{feature_mode}")



def load_mapping(mapping_path: Path):
    """
    兼容不同版本 train_visual_mapping_T.py 导出的 visual_mapping_T.json。

    你这次报错的原因：
        json 里某个候选字段是 dict，不是纯二维系数矩阵。
        旧版脚本直接 np.asarray(data[key])，所以遇到 dict 就炸。

    本函数会更谨慎地解析：
        1. 先读 feature_mode
        2. 跳过不是 list/tuple/np.ndarray 的字段
        3. 支持 coefficients 是 dict 的情况
        4. 支持 mapping 里只有 a/b 模型，L 使用 identity 的情况
        5. 支持字段名：
            coef / coefficients / W / weights / matrix
            coef_L/coef_a/coef_b
            coeff_L/coeff_a/coeff_b
            model_L/model_a/model_b
            L/a/b 子字段
    """
    data = json.loads(mapping_path.read_text(encoding="utf-8"))

    feature_mode = (
        data.get("feature_mode")
        or data.get("model_type")
        or data.get("feature_type")
        or data.get("features")
        or "poly2"
    )

    # 某些版本把参数包在 model/mapping 里
    containers = [data]
    for k in ["model", "mapping", "params", "visual_mapping"]:
        if isinstance(data.get(k), dict):
            containers.append(data[k])

    def as_1d(v):
        if isinstance(v, dict):
            # 常见：{"coef":[...]} / {"coeff":[...]} / {"weights":[...]}
            for kk in ["coef", "coeff", "coefficients", "weights", "w"]:
                if kk in v and not isinstance(v[kk], dict):
                    return np.asarray(v[kk], dtype=np.float64).reshape(-1)
            return None
        if isinstance(v, (list, tuple, np.ndarray)):
            return np.asarray(v, dtype=np.float64).reshape(-1)
        return None

    def try_matrix(v):
        if isinstance(v, dict):
            # dict 里可能包了一层 matrix
            for kk in ["coef", "coeff", "coefficients", "W", "weights", "matrix"]:
                if kk in v and not isinstance(v[kk], dict):
                    arr = np.asarray(v[kk], dtype=np.float64)
                    if arr.ndim == 2:
                        return arr
            return None

        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=np.float64)
            if arr.ndim == 2:
                return arr

        return None

    W = None

    # 1. 直接二维矩阵字段
    for container in containers:
        for key in ["coef", "coeff", "coefficients", "W", "weights", "matrix"]:
            if key in container:
                cand = try_matrix(container[key])
                if cand is not None:
                    W = cand
                    break
        if W is not None:
            break

    # 2. 三个输出分别存：coef_L / coef_a / coef_b
    if W is None:
        key_triplets = [
            ("coef_L", "coef_a", "coef_b"),
            ("coeff_L", "coeff_a", "coeff_b"),
            ("L_coef", "a_coef", "b_coef"),
            ("L_coeff", "a_coeff", "b_coeff"),
            ("wL", "wa", "wb"),
            ("coef_l", "coef_a", "coef_b"),
        ]

        for container in containers:
            for k1, k2, k3 in key_triplets:
                if all(k in container for k in [k1, k2, k3]):
                    c1 = as_1d(container[k1])
                    c2 = as_1d(container[k2])
                    c3 = as_1d(container[k3])
                    if c1 is not None and c2 is not None and c3 is not None:
                        W = np.stack([c1, c2, c3], axis=1)
                        break
            if W is not None:
                break

    # 3. outputs / coefficients 是 dict，里面有 L/a/b
    if W is None:
        for container in containers:
            for dict_key in ["outputs", "output_coefficients", "coefficients", "coef", "models"]:
                obj = container.get(dict_key)
                if isinstance(obj, dict):
                    # L 可能叫 L/l/lightness；a/b 同理
                    L_obj = obj.get("L", obj.get("l", obj.get("lightness")))
                    a_obj = obj.get("a", obj.get("A"))
                    b_obj = obj.get("b", obj.get("B"))

                    cL = as_1d(L_obj)
                    ca = as_1d(a_obj)
                    cb = as_1d(b_obj)

                    if ca is not None and cb is not None:
                        # 如果没有 L 系数，允许 L identity，后面单独处理
                        if cL is None:
                            cL = None
                        else:
                            cL = cL.reshape(-1)

                        ca = ca.reshape(-1)
                        cb = cb.reshape(-1)

                        if cL is not None:
                            W = np.stack([cL, ca, cb], axis=1)
                        else:
                            W = {"L_identity": True, "a": ca, "b": cb}
                        break
            if W is not None:
                break

    # 4. 某些版本只有 a_model / b_model，L 使用 identity
    if W is None:
        for container in containers:
            a_obj = (
                container.get("a_model")
                or container.get("model_a")
                or container.get("coef_a")
                or container.get("coeff_a")
            )
            b_obj = (
                container.get("b_model")
                or container.get("model_b")
                or container.get("coef_b")
                or container.get("coeff_b")
            )

            ca = as_1d(a_obj)
            cb = as_1d(b_obj)
            if ca is not None and cb is not None:
                W = {"L_identity": True, "a": ca, "b": cb}
                break

    if W is None:
        print("\nvisual_mapping_T.json 顶层字段：")
        for k, v in data.items():
            print("  ", k, type(v).__name__)
        raise RuntimeError(
            "无法从 mapping json 中解析系数。请把 visual_mapping_T.json 发我，"
            "或者把上面打印的字段截图给我。"
        )

    # 统一返回
    if isinstance(W, dict) and W.get("L_identity"):
        return {
            "feature_mode": feature_mode,
            "feature_names": data.get("feature_names"),
            "W": None,
            "L_identity": True,
            "coef_a": W["a"],
            "coef_b": W["b"],
            "raw": data,
        }

    W = np.asarray(W, dtype=np.float64)

    if W.ndim != 2:
        raise RuntimeError(f"mapping 系数矩阵维度不对：{W.shape}")

    # 统一成 [n_features, 3]
    if W.shape[1] == 3:
        pass
    elif W.shape[0] == 3:
        W = W.T
    else:
        raise RuntimeError(f"mapping 系数矩阵既不像 [n_features,3] 也不像 [3,n_features]：{W.shape}")

    return {
        "feature_mode": feature_mode,
        "feature_names": data.get("feature_names"),
        "W": W,
        "L_identity": False,
        "raw": data,
    }


def predict_visual_lab(corrected_lab: np.ndarray, mapping_info: dict):
    X = build_lab_features(
        corrected_lab,
        mapping_info["feature_mode"],
        feature_names=mapping_info.get("feature_names"),
    )

    # 情况 1：完整 L/a/b 矩阵
    if not mapping_info.get("L_identity", False):
        W = mapping_info["W"]

        if X.shape[1] != W.shape[0]:
            raise RuntimeError(
                f"feature 维度和 mapping 系数不匹配：features={X.shape[1]}, W={W.shape}。"
                f"mapping feature_mode={mapping_info['feature_mode']}"
            )

        pred = X @ W
        return pred.reshape(-1, 3)

    # 情况 2：L 直接沿用 corrected_L，只预测 a/b
    ca = np.asarray(mapping_info["coef_a"], dtype=np.float64).reshape(-1)
    cb = np.asarray(mapping_info["coef_b"], dtype=np.float64).reshape(-1)

    if X.shape[1] != len(ca) or X.shape[1] != len(cb):
        raise RuntimeError(
            f"feature 维度和 a/b 系数不匹配：features={X.shape[1]}, "
            f"coef_a={len(ca)}, coef_b={len(cb)}。"
            f"mapping feature_mode={mapping_info['feature_mode']}"
        )

    corrected_lab = np.asarray(corrected_lab, dtype=np.float64).reshape(-1, 3)

    pred_L = corrected_lab[:, 0]
    pred_a = X @ ca
    pred_b = X @ cb

    return np.stack([pred_L, pred_a, pred_b], axis=1)


# ============================================================
# 渲染
# ============================================================

def apply_display_tweak(lab: np.ndarray, l_offset=0.0, a_offset=0.0, b_offset=0.0, chroma_scale=1.0):
    out = np.asarray(lab, dtype=np.float64).copy()

    # 转成 LCh 更稳一些
    L = out[..., 0]
    a = out[..., 1]
    b = out[..., 2]
    C = np.sqrt(a * a + b * b)
    H = np.arctan2(b, a)

    C2 = C * float(chroma_scale)
    a2 = C2 * np.cos(H)
    b2 = C2 * np.sin(H)

    out[..., 0] = np.clip(L + float(l_offset), 0, 100)
    out[..., 1] = a2 + float(a_offset)
    out[..., 2] = b2 + float(b_offset)
    return out



def circle_alpha_mask(h, w, cx, cy, r, feather):
    """
    生成圆形 alpha mask。
    圆中心以内完全生效，边缘按 feather 羽化。
    """
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if feather <= 0:
        return (dist <= r).astype(np.float64)

    alpha = (r - dist) / max(float(feather), 1e-6)
    return np.clip(alpha, 0.0, 1.0)


def render_one_circle(base_rgb, corrected_rgb, circle, corrected_mean_lab, target_mean_lab, feather, alpha_strength=1.0, keep_texture=True):
    """
    如果 keep_texture=True：
        对圆内每个像素做 “整体平移 delta_lab”，保留纹理阴影
    否则：
        用 target_mean_lab 对整个圆做纯色填充
    """
    H, W = base_rgb.shape[:2]
    cx, cy, r = circle

    x1 = max(0, int(math.floor(cx - r - feather - 2)))
    y1 = max(0, int(math.floor(cy - r - feather - 2)))
    x2 = min(W, int(math.ceil(cx + r + feather + 2)))
    y2 = min(H, int(math.ceil(cy + r + feather + 2)))

    if x2 <= x1 or y2 <= y1:
        return base_rgb

    local_cx = cx - x1
    local_cy = cy - y1
    alpha = circle_alpha_mask(y2 - y1, x2 - x1, local_cx, local_cy, r, feather)
    if alpha.max() <= 0:
        return base_rgb

    base_patch = base_rgb[y1:y2, x1:x2, :].astype(np.float64)
    corr_patch = corrected_rgb[y1:y2, x1:x2, :].astype(np.float64)

    corr_lab = rgb_to_lab(corr_patch)

    if keep_texture:
        delta = (np.asarray(target_mean_lab, dtype=np.float64) - np.asarray(corrected_mean_lab, dtype=np.float64)) * float(alpha_strength)
        out_lab = corr_lab + delta.reshape(1, 1, 3)
    else:
        tgt = np.asarray(target_mean_lab, dtype=np.float64)
        cur = np.asarray(corrected_mean_lab, dtype=np.float64)
        tgt2 = cur + (tgt - cur) * float(alpha_strength)
        out_lab = np.zeros_like(corr_lab) + tgt2.reshape(1, 1, 3)

    out_rgb = lab_to_rgb(out_lab)
    alpha3 = alpha[..., None]

    mixed = base_patch * (1 - alpha3) + out_rgb * alpha3

    out = base_rgb.copy()
    out[y1:y2, x1:x2, :] = np.clip(mixed, 0, 255)
    return out


# ============================================================
# 主程序
# ============================================================


def _split_tokens(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    for sep in [',', ';', '|', '\t', '\n']:
        s = s.replace(sep, ' ')
    return [tok.strip() for tok in s.split(' ') if tok.strip()]


def _to_float_or_default(v, default):
    if v is None:
        return default
    try:
        if isinstance(v, str) and not v.strip():
            return default
        if pd.isna(v):
            return default
    except Exception:
        pass
    return float(v)


def load_override_rows(path: Path | None):
    """
    读取局部修正表。

    支持列：
        code / codes
        index / indices
        L_offset / l_offset
        a_offset
        b_offset
        chroma_scale
        alpha
        remark

    匹配规则：
        按 CSV 行顺序从上到下匹配，第一条命中即生效。
        因此“更靠前的规则优先”。
    """
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f'找不到 override csv：{path}')

    df = pd.read_csv(path, encoding='utf-8-sig')
    rows = []
    for _, row in df.iterrows():
        code_tokens = []
        index_tokens = []
        if 'code' in row.index:
            code_tokens += _split_tokens(row.get('code'))
        if 'codes' in row.index:
            code_tokens += _split_tokens(row.get('codes'))
        if 'index' in row.index:
            index_tokens += _split_tokens(row.get('index'))
        if 'indices' in row.index:
            index_tokens += _split_tokens(row.get('indices'))

        rows.append({
            'codes': set(code_tokens),
            'indices': set(index_tokens),
            'l_offset': _to_float_or_default(row.get('L_offset', row.get('l_offset', 0.0)), 0.0),
            'a_offset': _to_float_or_default(row.get('a_offset', 0.0), 0.0),
            'b_offset': _to_float_or_default(row.get('b_offset', 0.0), 0.0),
            'chroma_scale': _to_float_or_default(row.get('chroma_scale', 1.0), 1.0),
            'alpha': _to_float_or_default(row.get('alpha', 1.0), 1.0),
            'remark': '' if ('remark' not in row.index or pd.isna(row.get('remark'))) else str(row.get('remark')),
        })
    return rows


def find_override(rule_rows, code: str, index_value) -> dict | None:
    if not rule_rows:
        return None
    code = str(code).strip()
    idx_str = '' if index_value is None else str(index_value).strip()
    try:
        idx_int_str = str(int(float(index_value))) if str(index_value).strip() != '' else ''
    except Exception:
        idx_int_str = idx_str

    for rule in rule_rows:
        if rule['codes'] and code in rule['codes']:
            return rule
        if rule['indices'] and (idx_str in rule['indices'] or idx_int_str in rule['indices']):
            return rule
    return None


def main():
    parser = argparse.ArgumentParser(description="利用 combined mapping_T，把 corrected 图自动渲染成初始肉眼图。")

    parser.add_argument("--corrected", required=True, type=Path, help="02_corrected.png")
    parser.add_argument("--samples-csv", required=True, type=Path, help="corrected_samples.csv")
    parser.add_argument("--circle-file", required=True, type=Path, help="visual_circles_manual.json")
    parser.add_argument("--mapping", required=True, type=Path, help="combined visual_mapping_T.json")
    parser.add_argument("--out", required=True, type=Path, help="输出自动视觉图")

    parser.add_argument("--background-mode", choices=["corrected", "original"], default="corrected", help="底图选择")
    parser.add_argument("--original-photo", default=None, type=Path, help="background-mode=original 时需要")

    parser.add_argument("--feather", type=float, default=6.0, help="圆边缘羽化")
    parser.add_argument("--alpha", type=float, default=1.0, help="映射改动强度 0~1")
    parser.add_argument("--display-l-offset", type=float, default=0.0)
    parser.add_argument("--display-a-offset", type=float, default=0.0)
    parser.add_argument("--display-b-offset", type=float, default=0.0)
    parser.add_argument("--display-chroma-scale", type=float, default=1.0)

    parser.add_argument("--flat-fill", action="store_true", help="不保留纹理，直接纯色填圆")
    parser.add_argument("--override-csv", default=None, type=Path, help="局部人工修正表，支持 code/index 和 L/a/b/chroma/alpha 覆盖")
    parser.add_argument("--out-pred-csv", default=None, type=Path, help="输出每个胶块预测结果 csv")
    args = parser.parse_args()

    if args.background_mode == "original" and args.original_photo is None:
        raise RuntimeError("background-mode=original 时，必须提供 --original-photo。")

    corrected_bgr = imread_unicode(args.corrected)
    corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

    if args.background_mode == "corrected":
        base_rgb = corrected_rgb.copy()
    else:
        base_bgr = imread_unicode(args.original_photo)
        if base_bgr.shape[:2] != corrected_bgr.shape[:2]:
            raise RuntimeError("original-photo 和 corrected 图尺寸不一致。")
        base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

    samples = pd.read_csv(args.samples_csv, encoding="utf-8-sig")
    if "code" not in samples.columns:
        raise RuntimeError("samples csv 中找不到 code 列。")

    circles = load_circles(args.circle_file, samples)
    mapping_info = load_mapping(args.mapping)
    override_rows = load_override_rows(args.override_csv)

    final_rgb = base_rgb.copy()
    pred_rows = []

    for i, row in samples.iterrows():
        circle = circles[i] if i < len(circles) else None
        if circle is None:
            continue

        code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()

        # corrected_lab：优先用 csv，避免重复采样误差
        cols_ok = all(c in row.index for c in ["corrected_L", "corrected_a", "corrected_b"])
        if cols_ok and pd.notna(row["corrected_L"]) and str(row["corrected_L"]) != "":
            corr_mean_lab = np.array(
                [float(row["corrected_L"]), float(row["corrected_a"]), float(row["corrected_b"])],
                dtype=np.float64,
            )
        else:
            # 回退：从 corrected 图重采样
            cx, cy, r = circle
            x1 = max(0, int(math.floor(cx - r)))
            y1 = max(0, int(math.floor(cy - r)))
            x2 = min(corrected_rgb.shape[1], int(math.ceil(cx + r)))
            y2 = min(corrected_rgb.shape[0], int(math.ceil(cy + r)))
            patch = corrected_rgb[y1:y2, x1:x2, :]
            yy, xx = np.mgrid[0:(y2 - y1), 0:(x2 - x1)]
            mask = np.sqrt((xx - (cx - x1)) ** 2 + (yy - (cy - y1)) ** 2) <= r
            mean_rgb = patch[mask].reshape(-1, 3).mean(axis=0)
            corr_mean_lab = rgb_to_lab(mean_rgb.reshape(1, 1, 3))[0, 0]

        pred_lab_raw = predict_visual_lab(corr_mean_lab.reshape(1, 3), mapping_info)[0]
        pred_lab = apply_display_tweak(
            pred_lab_raw.reshape(1, 3),
            l_offset=args.display_l_offset,
            a_offset=args.display_a_offset,
            b_offset=args.display_b_offset,
            chroma_scale=args.display_chroma_scale,
        )[0]

        override = find_override(override_rows, code, row.get("index", i + 1))
        local_l_offset = 0.0
        local_a_offset = 0.0
        local_b_offset = 0.0
        local_chroma_scale = 1.0
        local_alpha = 1.0
        local_remark = ""
        if override is not None:
            local_l_offset = float(override.get("l_offset", 0.0))
            local_a_offset = float(override.get("a_offset", 0.0))
            local_b_offset = float(override.get("b_offset", 0.0))
            local_chroma_scale = float(override.get("chroma_scale", 1.0))
            local_alpha = float(override.get("alpha", 1.0))
            local_remark = str(override.get("remark", ""))
            pred_lab = apply_display_tweak(
                pred_lab.reshape(1, 3),
                l_offset=local_l_offset,
                a_offset=local_a_offset,
                b_offset=local_b_offset,
                chroma_scale=local_chroma_scale,
            )[0]

        render_alpha = float(args.alpha) * float(local_alpha)

        final_rgb = render_one_circle(
            base_rgb=final_rgb,
            corrected_rgb=corrected_rgb,
            circle=circle,
            corrected_mean_lab=corr_mean_lab,
            target_mean_lab=pred_lab,
            feather=float(args.feather),
            alpha_strength=render_alpha,
            keep_texture=not args.flat_fill,
        )

        pred_rows.append(
            {
                "index": row.get("index", i + 1),
                "code": code,
                "name": name,
                "circle_cx": circle[0],
                "circle_cy": circle[1],
                "circle_r": circle[2],
                "corrected_L": corr_mean_lab[0],
                "corrected_a": corr_mean_lab[1],
                "corrected_b": corr_mean_lab[2],
                "pred_visual_raw_L": pred_lab_raw[0],
                "pred_visual_raw_a": pred_lab_raw[1],
                "pred_visual_raw_b": pred_lab_raw[2],
                "pred_visual_L": pred_lab[0],
                "pred_visual_a": pred_lab[1],
                "pred_visual_b": pred_lab[2],
                "override_applied": override is not None,
                "override_l_offset": local_l_offset,
                "override_a_offset": local_a_offset,
                "override_b_offset": local_b_offset,
                "override_chroma_scale": local_chroma_scale,
                "override_alpha": local_alpha,
                "override_remark": local_remark,
            }
        )

    out_bgr = cv2.cvtColor(np.clip(final_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    imwrite_unicode(args.out, out_bgr)

    pred_csv = args.out_pred_csv or args.out.with_suffix(".predicted_visual_lab.csv")
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    summary = {
        "corrected": str(args.corrected),
        "samples_csv": str(args.samples_csv),
        "circle_file": str(args.circle_file),
        "mapping": str(args.mapping),
        "out": str(args.out),
        "pred_csv": str(pred_csv),
        "background_mode": args.background_mode,
        "original_photo": str(args.original_photo) if args.original_photo else "",
        "feature_mode": mapping_info["feature_mode"],
        "feature_names": mapping_info.get("feature_names"),
        "W_shape": list(mapping_info["W"].shape) if mapping_info.get("W") is not None else "L_identity_ab_only",
        "feather": args.feather,
        "alpha": args.alpha,
        "display_l_offset": args.display_l_offset,
        "display_a_offset": args.display_a_offset,
        "display_b_offset": args.display_b_offset,
        "display_chroma_scale": args.display_chroma_scale,
        "override_csv": str(args.override_csv) if args.override_csv else "",
        "override_rule_count": len(override_rows),
        "keep_texture": not args.flat_fill,
        "count": len(pred_rows),
        "note": "如果自动视觉图偏灰/偏浅，可继续调 display-b-offset / display-chroma-scale / display-l-offset。",
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print("输出图：", args.out)
    print("预测 Lab：", pred_csv)
    print("summary：", summary_path)
    print("已渲染胶块数：", len(pred_rows))
    print("mapping feature_mode：", mapping_info["feature_mode"])
    print("mapping W shape：", mapping_info["W"].shape if mapping_info.get("W") is not None else "L_identity_ab_only")


if __name__ == "__main__":
    main()
