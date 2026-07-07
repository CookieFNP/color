# 输出 图像 图表 CSV JSON等

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def add_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_side_by_side(path: str | Path, left_bgr: np.ndarray, right_bgr: np.ndarray, left_text: str, right_text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h = min(left_bgr.shape[0], right_bgr.shape[0])
    left_resized = cv2.resize(left_bgr, (max(1, int(left_bgr.shape[1] * h / left_bgr.shape[0])), h))
    right_resized = cv2.resize(right_bgr, (max(1, int(right_bgr.shape[1] * h / right_bgr.shape[0])), h))

    combined = np.concatenate([add_label(left_resized, left_text), add_label(right_resized, right_text)], axis=1)
    cv2.imwrite(str(path), combined)


def save_delta_e_plot(path: str | Path, before_de: np.ndarray, after_de: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(1, len(before_de) + 1)
    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, before_de, width=0.4, label="Before correction")
    plt.bar(x + 0.2, after_de, width=0.4, label="After correction")
    plt.xlabel("ColorChecker patch index")
    plt.ylabel("Delta E 2000")
    plt.title("Color difference before and after correction")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_sample_csv(
    path: str | Path,
    captured_rgb: np.ndarray,
    corrected_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    before_de: np.ndarray,
    after_de: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "patch_index",
                "captured_R",
                "captured_G",
                "captured_B",
                "corrected_R",
                "corrected_G",
                "corrected_B",
                "reference_R",
                "reference_G",
                "reference_B",
                "deltaE_before",
                "deltaE_after",
            ]
        )

        for i in range(len(reference_rgb)):
            writer.writerow(
                [
                    i + 1,
                    *captured_rgb[i].round(3).tolist(),
                    *corrected_rgb[i].round(3).tolist(),
                    *reference_rgb[i].round(3).tolist(),
                    round(float(before_de[i]), 4),
                    round(float(after_de[i]), 4),
                ]
            )


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_target_validation_csv(path: str | Path, target_results: list[dict]) -> None:
    # 保存胶块目标颜色验证结果
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "index",
        "input_label",
        "standard_code",
        "standard_name",
        "standard_L",
        "standard_a",
        "standard_b",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "valid_mask_pixels",
        "before_R",
        "before_G",
        "before_B",
        "after_R",
        "after_G",
        "after_B",
        "before_L",
        "before_a",
        "before_b",
        "after_L",
        "after_a",
        "after_b",
        "before_dL",
        "before_da",
        "before_db",
        "after_dL",
        "after_da",
        "after_db",
        "deltaE_before_to_standard",
        "deltaE_after_to_standard",
        "deltaE_improvement",
        "pass_after_threshold",
        "pred_before_code",
        "pred_before_name",
        "pred_before_deltaE",
        "pred_after_code",
        "pred_after_name",
        "pred_after_deltaE",
        "classification_correct_after",
        "target_mask_debug",
        "target_before_after",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for item in target_results:
            std = item.get("standard") or {}
            de = item.get("delta_e_2000_to_standard") or {}
            pred_before = (item.get("nearest_before") or [{}])[0]
            pred_after = (item.get("nearest_after") or [{}])[0]
            roi = item.get("roi_xyxy") or [None, None, None, None]

            writer.writerow({
                "index": item.get("index"),
                "input_label": item.get("input_label"),
                "standard_code": std.get("code"),
                "standard_name": std.get("name"),
                "standard_L": (std.get("lab") or [None, None, None])[0],
                "standard_a": (std.get("lab") or [None, None, None])[1],
                "standard_b": (std.get("lab") or [None, None, None])[2],
                "roi_x1": roi[0],
                "roi_y1": roi[1],
                "roi_x2": roi[2],
                "roi_y2": roi[3],
                "valid_mask_pixels": item.get("valid_mask_pixels"),
                "before_R": item.get("before_rgb", [None, None, None])[0],
                "before_G": item.get("before_rgb", [None, None, None])[1],
                "before_B": item.get("before_rgb", [None, None, None])[2],
                "after_R": item.get("after_rgb", [None, None, None])[0],
                "after_G": item.get("after_rgb", [None, None, None])[1],
                "after_B": item.get("after_rgb", [None, None, None])[2],
                "before_L": item.get("before_lab", [None, None, None])[0],
                "before_a": item.get("before_lab", [None, None, None])[1],
                "before_b": item.get("before_lab", [None, None, None])[2],
                "after_L": item.get("after_lab", [None, None, None])[0],
                "after_a": item.get("after_lab", [None, None, None])[1],
                "after_b": item.get("after_lab", [None, None, None])[2],
                "before_dL": item.get("delta_lab_to_standard", {}).get("before", [None, None, None])[0],
                "before_da": item.get("delta_lab_to_standard", {}).get("before", [None, None, None])[1],
                "before_db": item.get("delta_lab_to_standard", {}).get("before", [None, None, None])[2],
                "after_dL": item.get("delta_lab_to_standard", {}).get("after", [None, None, None])[0],
                "after_da": item.get("delta_lab_to_standard", {}).get("after", [None, None, None])[1],
                "after_db": item.get("delta_lab_to_standard", {}).get("after", [None, None, None])[2],
                "deltaE_before_to_standard": de.get("before"),
                "deltaE_after_to_standard": de.get("after"),
                "deltaE_improvement": de.get("improvement"),
                "pass_after_threshold": item.get("pass_after_threshold"),
                "pred_before_code": pred_before.get("code"),
                "pred_before_name": pred_before.get("name"),
                "pred_before_deltaE": pred_before.get("delta_e_2000"),
                "pred_after_code": pred_after.get("code"),
                "pred_after_name": pred_after.get("name"),
                "pred_after_deltaE": pred_after.get("delta_e_2000"),
                "classification_correct_after": item.get("classification_correct_after"),
                "target_mask_debug": item.get("outputs", {}).get("target_mask_debug"),
                "target_before_after": item.get("outputs", {}).get("target_before_after"),
            })



def save_alpha_sweep_csv(path: str | Path, alpha_sweep: list[dict]) -> None:
    # 保存校正强度 alpha 对目标胶块 ΔE 的影响。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "alpha",
        "target_mean_deltaE",
        "target_median_deltaE",
        "target_max_deltaE",
        "target_p95_deltaE",
        "harm_count",
        "harm_rate",
        "classification_acc",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in alpha_sweep:
            writer.writerow({key: row.get(key) for key in headers})


def save_validation_bar_plot(path: str | Path, target_results: list[dict]) -> None:
    # 保存所有胶块样本校正前后的 ΔE 对比图
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [item for item in target_results if item.get("delta_e_2000_to_standard")]
    if not rows:
        return

    labels = []
    before = []
    after = []
    for item in rows:
        std = item.get("standard") or {}
        labels.append(std.get("code") or str(item.get("index")))
        before.append(item["delta_e_2000_to_standard"]["before"])
        after.append(item["delta_e_2000_to_standard"]["after"])

    x = np.arange(len(rows))
    plt.figure(figsize=(max(8, len(rows) * 0.75), 5))
    plt.bar(x - 0.2, before, width=0.4, label="Before correction")
    plt.bar(x + 0.2, after, width=0.4, label="After correction")
    plt.xlabel("Target class")
    plt.ylabel("Delta E 2000 to standard Lab")
    plt.title("Target glue block validation")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
