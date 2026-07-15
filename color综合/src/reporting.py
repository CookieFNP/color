from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import imwrite_unicode, write_json


def add_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(out, text[:80], (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def save_side_by_side(path: str | Path, left_bgr: np.ndarray, right_bgr: np.ndarray, left_text: str, right_text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = min(left_bgr.shape[0], right_bgr.shape[0])
    if h <= 0:
        return
    left = cv2.resize(left_bgr, (max(1, int(left_bgr.shape[1] * h / left_bgr.shape[0])), h))
    right = cv2.resize(right_bgr, (max(1, int(right_bgr.shape[1] * h / right_bgr.shape[0])), h))
    both = np.concatenate([add_label(left, left_text), add_label(right, right_text)], axis=1)
    imwrite_unicode(path, both)


def save_delta_e_plot(path: str | Path, before_de: np.ndarray, after_de: np.ndarray, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(before_de) + 1)
    plt.figure(figsize=(14, 5))
    plt.bar(x - 0.2, before_de, width=0.4, label="before")
    plt.bar(x + 0.2, after_de, width=0.4, label="after")
    plt.xlabel("index")
    plt.ylabel("ΔE2000")
    plt.title(title)
    if len(x) <= 40:
        plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_target_validation_csv(path: str | Path, target_results: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "index", "input_code", "input_name",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2", "valid_mask_pixels",
        "standard_L", "standard_a", "standard_b",
        "before_L", "before_a", "before_b",
        "after_L", "after_a", "after_b",
        "before_deltaE", "after_deltaE", "improvement",
        "top1_code", "top1_name", "top1_deltaE",
        "top2_code", "top2_name", "top2_deltaE",
        "top3_code", "top3_name", "top3_deltaE",
        "top1_correct", "top3_correct", "confidence", "margin",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in target_results:
            nearest = r.get("nearest_after") or []
            row = {
                "index": r["index"],
                "input_code": r["standard"]["code"],
                "input_name": r["standard"]["name"],
                "roi_x1": r["roi_xyxy"][0],
                "roi_y1": r["roi_xyxy"][1],
                "roi_x2": r["roi_xyxy"][2],
                "roi_y2": r["roi_xyxy"][3],
                "valid_mask_pixels": r["valid_mask_pixels"],
                "standard_L": r["standard"]["lab"][0],
                "standard_a": r["standard"]["lab"][1],
                "standard_b": r["standard"]["lab"][2],
                "before_L": r["before_lab"][0],
                "before_a": r["before_lab"][1],
                "before_b": r["before_lab"][2],
                "after_L": r["after_lab"][0],
                "after_a": r["after_lab"][1],
                "after_b": r["after_lab"][2],
                "before_deltaE": r["delta_e_2000_to_standard"]["before"],
                "after_deltaE": r["delta_e_2000_to_standard"]["after"],
                "improvement": r["delta_e_2000_to_standard"]["improvement"],
                "top1_correct": r["classification_correct_after"],
                "top3_correct": r["top3_correct_after"],
                "confidence": r["confidence"]["level"],
                "margin": r["confidence"].get("margin"),
            }
            for k in range(3):
                if len(nearest) > k:
                    row[f"top{k+1}_code"] = nearest[k]["code"]
                    row[f"top{k+1}_name"] = nearest[k]["name"]
                    row[f"top{k+1}_deltaE"] = nearest[k]["delta_e_2000"]
            writer.writerow(row)


def stat_pack(x: list[float] | np.ndarray) -> dict:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "p95": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }
