import os
import glob
import json
import re
import cv2
import numpy as np
import pandas as pd
from skimage import color


# =========================
# 配置区
# =========================
REPORT_GLOB = "output_*/report.json"   # 扫描所有输出目录
SAVE_ROOT = "best_model_color_blocks"  # 保存纯色图的总目录
WIDTH = 800
HEIGHT = 800


# =========================
# 工具函数
# =========================
def safe_name(text: str) -> str:
    """把文件名里不安全的字符替换掉"""
    if text is None:
        return "unknown"
    return re.sub(r'[\\/:*?"<>| ]+', "_", str(text))


def lab_to_bgr_block(lab_value, width=800, height=800):
    """
    把单个 Lab 值转换成纯色图(BGR, uint8)，用于 cv2.imwrite 保存
    lab_value: [L, a, b]
    """
    lab_value = np.asarray(lab_value, dtype=np.float64)

    lab_img = np.zeros((height, width, 3), dtype=np.float64)
    lab_img[:, :, 0] = lab_value[0]
    lab_img[:, :, 1] = lab_value[1]
    lab_img[:, :, 2] = lab_value[2]

    # skimage 输出 RGB，范围 0~1
    rgb_img = color.lab2rgb(lab_img)

    # 转成 0~255
    rgb_255 = np.clip(rgb_img * 255, 0, 255).astype(np.uint8)

    # OpenCV 保存用 BGR
    bgr_255 = cv2.cvtColor(rgb_255, cv2.COLOR_RGB2BGR)
    return bgr_255


def save_lab_block(lab_value, save_path, width=800, height=800):
    """保存一个 Lab 对应的纯色块图"""
    img = lab_to_bgr_block(lab_value, width=width, height=height)
    cv2.imwrite(save_path, img)


def get_report_score(report: dict):
    """
    给 report 计算一个分数，越小越好。
    优先:
        report["target_validation"]["after_mean"]
    否则:
        所有 target_colors 的 deltaE_after 平均值
    """
    # 方案1：优先尝试 report 内可能已有的总体指标

    candidate_scores = []

    if isinstance(report.get("target_validation"), dict):
        tv = report["target_validation"]
        if tv.get("after_mean") is not None:
            candidate_scores.append(float(tv["after_mean"]))

    if isinstance(report.get("summary"), dict):
        sm = report["summary"]
        if sm.get("target_after_mean") is not None:
            candidate_scores.append(float(sm["target_after_mean"]))
        if sm.get("chart_after_mean") is not None:
            candidate_scores.append(float(sm["chart_after_mean"]))

    if report.get("target_after_mean") is not None:
        candidate_scores.append(float(report["target_after_mean"]))

    if report.get("chart_after_mean") is not None:
        candidate_scores.append(float(report["chart_after_mean"]))

    if candidate_scores:
        return candidate_scores[0]

    values = []
    for item in report.get("target_colors", []):
        de = item.get("delta_e_2000_to_standard", {})
        after = de.get("after")
        if after is not None:
            values.append(float(after))

    if values:
        return float(np.mean(values))

    # 完全没有可用信息，给一个极大值
    return float("inf")


def extract_model_info(report: dict):
    """提取模型信息字符串"""
    model_type = None
    ridge_alpha = None

    if isinstance(report.get("model"), dict):
        model_type = report["model"].get("type")
        ridge_alpha = report["model"].get("ridge_alpha")

    return model_type, ridge_alpha


# =========================
# 第一步：扫描所有 report.json，找最好的模型
# =========================
report_paths = glob.glob(REPORT_GLOB)

if not report_paths:
    raise FileNotFoundError(f"没有找到任何 report.json，检查路径模式是否正确：{REPORT_GLOB}")

report_rows = []
best_report = None
best_report_path = None
best_score = float("inf")

for report_path in report_paths:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    score = get_report_score(report)
    model_type, ridge_alpha = extract_model_info(report)

    report_rows.append({
        "report_path": report_path,
        "output_dir": os.path.dirname(report_path),
        "model": model_type,
        "ridge_alpha": ridge_alpha,
        "score_smaller_is_better": score,
    })

    if score < best_score:
        best_score = score
        best_report = report
        best_report_path = report_path

report_df = pd.DataFrame(report_rows).sort_values("score_smaller_is_better")
print("\n===== 所有模型评分（越小越好） =====")
print(report_df.to_string(index=False))

if best_report is None:
    raise RuntimeError("没有成功解析出最优模型。")

best_output_dir = os.path.dirname(best_report_path)
best_model_type, best_ridge_alpha = extract_model_info(best_report)

print("\n===== 自动选出的最佳模型 =====")
print(f"report 路径: {best_report_path}")
print(f"output_dir : {best_output_dir}")
print(f"model      : {best_model_type}")
print(f"ridge_alpha: {best_ridge_alpha}")
print(f"score      : {best_score:.6f}")


# =========================
# 第二步：从最佳模型里提取 12 个颜色
# =========================
target_colors = best_report.get("target_colors", [])
if not target_colors:
    raise RuntimeError("最佳 report.json 中没有 target_colors。")

os.makedirs(SAVE_ROOT, exist_ok=True)

summary_rows = []

for idx, item in enumerate(target_colors, start=1):
    std = item.get("standard", {})
    code = std.get("code", f"color_{idx:02d}")
    name = std.get("name", "")
    std_lab = std.get("lab", None)

    before_lab = item.get("before_lab", None)
    after_lab = item.get("after_lab", None)

    de_info = item.get("delta_e_2000_to_standard", {})
    de_before = de_info.get("before", None)
    de_after = de_info.get("after", None)
    improvement = de_info.get("improvement", None)

    # 为每个颜色建一个文件夹
    folder_name = f"{idx:02d}_{safe_name(code)}"
    if name:
        folder_name += f"_{safe_name(name)}"

    color_dir = os.path.join(SAVE_ROOT, folder_name)
    os.makedirs(color_dir, exist_ok=True)

    # 三种图路径
    before_path = os.path.join(color_dir, "before.png")
    after_path = os.path.join(color_dir, "after.png")
    standard_path = os.path.join(color_dir, "standard.png")

    # 保存纯色图
    if before_lab is not None:
        save_lab_block(before_lab, before_path, width=WIDTH, height=HEIGHT)

    if after_lab is not None:
        save_lab_block(after_lab, after_path, width=WIDTH, height=HEIGHT)

    if std_lab is not None:
        save_lab_block(std_lab, standard_path, width=WIDTH, height=HEIGHT)

    # 同时把 Lab / RGB 也记下来，便于检查
    def lab_to_rgb_triplet(lab_value):
        if lab_value is None:
            return [None, None, None]
        img = np.zeros((1, 1, 3), dtype=np.float64)
        img[0, 0, :] = np.asarray(lab_value, dtype=np.float64)
        rgb = color.lab2rgb(img)[0, 0]
        rgb_255 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return [int(rgb_255[0]), int(rgb_255[1]), int(rgb_255[2])]

    before_rgb = lab_to_rgb_triplet(before_lab)
    after_rgb = lab_to_rgb_triplet(after_lab)
    std_rgb = lab_to_rgb_triplet(std_lab)

    summary_rows.append({
        "index": idx,
        "code": code,
        "name": name,

        "before_L": before_lab[0] if before_lab is not None else None,
        "before_a": before_lab[1] if before_lab is not None else None,
        "before_b": before_lab[2] if before_lab is not None else None,
        "before_R": before_rgb[0],
        "before_G": before_rgb[1],
        "before_B": before_rgb[2],

        "after_L": after_lab[0] if after_lab is not None else None,
        "after_a": after_lab[1] if after_lab is not None else None,
        "after_b": after_lab[2] if after_lab is not None else None,
        "after_R": after_rgb[0],
        "after_G": after_rgb[1],
        "after_B": after_rgb[2],

        "standard_L": std_lab[0] if std_lab is not None else None,
        "standard_a": std_lab[1] if std_lab is not None else None,
        "standard_b": std_lab[2] if std_lab is not None else None,
        "standard_R": std_rgb[0],
        "standard_G": std_rgb[1],
        "standard_B": std_rgb[2],

        "deltaE_before": de_before,
        "deltaE_after": de_after,
        "improvement": improvement,

        "before_img": before_path if before_lab is not None else None,
        "after_img": after_path if after_lab is not None else None,
        "standard_img": standard_path if std_lab is not None else None,
    })


# =========================
# 第三步：导出汇总结果
# =========================
summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(SAVE_ROOT, "best_model_12colors_summary.csv")
summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

print("\n===== 12 个颜色的汇总结果 =====")
print(summary_df[[
    "index", "code", "name",
    "deltaE_before", "deltaE_after", "improvement"
]].to_string(index=False))

print(f"\n纯色图已保存到：{SAVE_ROOT}")
print(f"汇总 CSV 已保存：{summary_csv_path}")

# =========================
# 检查 after 是否系统性偏暗
# =========================
summary_df["delta_L_before"] = summary_df["before_L"] - summary_df["standard_L"]
summary_df["delta_L_after"] = summary_df["after_L"] - summary_df["standard_L"]

print("\n===== 亮度 L 偏差检查 =====")
print(summary_df[[
    "index", "code", "name",
    "before_L", "after_L", "standard_L",
    "delta_L_before", "delta_L_after",
    "deltaE_before", "deltaE_after"
]].to_string(index=False))

print("\n===== L 偏差统计 =====")
print(f"before 平均 ΔL = {summary_df['delta_L_before'].mean():.3f}")
print(f"after  平均 ΔL = {summary_df['delta_L_after'].mean():.3f}")
print(f"before ΔL 中位数 = {summary_df['delta_L_before'].median():.3f}")
print(f"after  ΔL 中位数 = {summary_df['delta_L_after'].median():.3f}")