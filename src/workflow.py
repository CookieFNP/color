# 主流程
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

from .calibration import apply_correction_to_image, fit_correction_model
from .chart import extract_chart_means, warp_chart_from_photo
from .color_math import delta_e_2000, rgb_to_lab
from .glue_mask import (
    build_glue_block_mask,
    draw_roi_and_mask,
    get_glue_block_representative_rgb,
)
from .interaction import select_four_points, select_roi
from .reporting import (
    save_delta_e_plot,
    save_json,
    save_sample_csv,
    save_side_by_side,
    save_target_validation_csv,
    save_validation_bar_plot,
)
from .standards import (
    nearest_standards,
    parse_standard_sequence,
    resolve_standard,
    standards_as_rows,
)


# 固定参数
ROWS = 4
COLS = 6
CENTER_RATIO = 0.50

MODEL = "linear_bias"
RIDGE_ALPHA = 1e-6

TRIM_PERCENT = 10.0
VALIDATION_THRESHOLD = 5.0

TARGET_SEQUENCE = "all"


# 目标胶块 ROI 获取

def _get_target_specs(photo_bgr: np.ndarray) -> list[dict]:
    """
    按内置标准顺序依次框选 12 个胶块。
    默认顺序：
    W015 -> W016 -> W031 -> W032 -> W047 -> W048
    -> W063 -> W064 -> W079 -> W080 -> W095 -> W096
    """
    labels = parse_standard_sequence(TARGET_SEQUENCE)
    specs: list[dict] = []

    print("\n批量验证顺序：")
    print("  " + " -> ".join(labels))

    for idx, label in enumerate(labels, start=1):
        standard = resolve_standard(label)
        readable = f"{standard.code} {standard.name}" if standard else label

        print(f"\n===== 选择第 {idx}/{len(labels)} 个胶块：{readable} =====")

        roi = select_roi(
            photo_bgr,
            window_name=f"target {idx:02d} {readable}",
            prompt=f"框选 {readable} 胶块主体 | Enter确认 | R重选 | Esc取消",
        )

        if roi is None:
            raise RuntimeError(f"未选择第 {idx} 个胶块 ROI。")

        specs.append(
            {
                "index": idx,
                "roi": roi,
                "input_label": label,
            }
        )

    return specs


# 单个胶块处理

def _process_one_target(
    *,
    index: int,
    input_label: str,
    target_roi: tuple[int, int, int, int],
    photo_bgr: np.ndarray,
    corrected_photo_bgr: np.ndarray,
    out_dir: Path,
) -> dict:
    """
    处理单个胶块：
    1. 构建目标 mask
    2. 提取校正前 RGB / Lab
    3. 提取校正后 RGB / Lab
    4. 与数据库标准 Lab 计算 ΔE00
    5. 输出 mask 和前后对比图
    """
    target_dir = out_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"target_{index:02d}"

    standard = resolve_standard(input_label)
    if standard is None:
        raise ValueError(f"未在标准库中找到类别：{input_label}")

    target_mask = build_glue_block_mask(
        photo_bgr,
        target_roi,
        debug_path=target_dir / f"{prefix}_mask_debug.png",
    )

    target_before_rgb = get_glue_block_representative_rgb(
        photo_bgr,
        target_roi,
        mask=target_mask,
        trim_percent=TRIM_PERCENT,
    )

    target_after_rgb = get_glue_block_representative_rgb(
        corrected_photo_bgr,
        target_roi,
        mask=target_mask,
        trim_percent=TRIM_PERCENT,
    )

    target_before_lab = rgb_to_lab(target_before_rgb[None, :])[0]
    target_after_lab = rgb_to_lab(target_after_rgb[None, :])[0]

    reference_lab = np.asarray(standard.lab, dtype=np.float32)

    before_de = float(
        delta_e_2000(
            target_before_lab[None, :],
            reference_lab[None, :],
        )[0]
    )

    after_de = float(
        delta_e_2000(
            target_after_lab[None, :],
            reference_lab[None, :],
        )[0]
    )

    nearest_before = nearest_standards(target_before_lab, top_k=3)
    nearest_after = nearest_standards(target_after_lab, top_k=3)

    classification_correct_after = (
        nearest_after[0]["code"] == standard.code
        if nearest_after
        else None
    )

    x1, y1, x2, y2 = target_roi
    target_before_crop = photo_bgr[y1:y2, x1:x2]
    target_after_crop = corrected_photo_bgr[y1:y2, x1:x2]

    target_before_after_path = target_dir / f"{prefix}_before_after.png"

    save_side_by_side(
        target_before_after_path,
        target_before_crop,
        target_after_crop,
        f"Before {standard.code}",
        f"After {standard.code}",
    )

    return {
        "index": index,
        "input_label": input_label,
        "standard": {
            **standard.as_dict(),
            "source": "built-in standard database",
        },
        "roi_xyxy": list(map(int, target_roi)),
        "sampling_method": "glue block mask + highlight/shadow filtering + trimmed mean",
        "trim_percent": TRIM_PERCENT,
        "valid_mask_pixels": int(np.sum(target_mask > 0)),
        "before_rgb": target_before_rgb.round(3).tolist(),
        "after_rgb": target_after_rgb.round(3).tolist(),
        "before_lab": target_before_lab.round(3).tolist(),
        "after_lab": target_after_lab.round(3).tolist(),
        "delta_e_2000_to_standard": {
            "reference_lab": reference_lab.round(4).tolist(),
            "before": before_de,
            "after": after_de,
            "improvement": before_de - after_de,
        },
        "validation_threshold": VALIDATION_THRESHOLD,
        "pass_after_threshold": after_de <= VALIDATION_THRESHOLD,
        "nearest_before": nearest_before,
        "nearest_after": nearest_after,
        "classification_correct_after": classification_correct_after,
        "outputs": {
            "target_mask_debug": str(target_dir / f"{prefix}_mask_debug.png"),
            "target_before_after": str(target_before_after_path),
        },
        "_mask": target_mask,
    }


# 主流程
def run(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    photo_bgr = cv2.imread(args.photo)
    standard_bgr = cv2.imread(args.standard)

    if photo_bgr is None:
        raise FileNotFoundError(f"读取实拍图失败：{args.photo}")

    if standard_bgr is None:
        raise FileNotFoundError(f"读取标准色卡失败：{args.standard}")

    standard_h, standard_w = standard_bgr.shape[:2]

    print("\n请依次点击实拍图中色卡四角：左上、右上、右下、左下。")
    corners = select_four_points(photo_bgr)

    captured_chart_bgr, perspective_matrix = warp_chart_from_photo(
        photo_bgr,
        corners,
        output_size=(standard_w, standard_h),
    )

    cv2.imwrite(str(out_dir / "01_captured_chart_warped.png"), captured_chart_bgr)
    cv2.imwrite(str(out_dir / "02_standard_chart.png"), standard_bgr)

    captured_rgb = extract_chart_means(
        captured_chart_bgr,
        rows=ROWS,
        cols=COLS,
        center_ratio=CENTER_RATIO,
    )

    reference_rgb = extract_chart_means(
        standard_bgr,
        rows=ROWS,
        cols=COLS,
        center_ratio=CENTER_RATIO,
    )

    W = fit_correction_model(
        captured_rgb,
        reference_rgb,
        model=MODEL,
        ridge_alpha=RIDGE_ALPHA,
    )

    corrected_photo_bgr = apply_correction_to_image(
        photo_bgr,
        W,
        model=MODEL,
    )

    corrected_chart_bgr = apply_correction_to_image(
        captured_chart_bgr,
        W,
        model=MODEL,
    )

    cv2.imwrite(str(out_dir / "03_corrected_photo.png"), corrected_photo_bgr)
    cv2.imwrite(str(out_dir / "04_corrected_chart_warped.png"), corrected_chart_bgr)

    corrected_chart_rgb = extract_chart_means(
        corrected_chart_bgr,
        rows=ROWS,
        cols=COLS,
        center_ratio=CENTER_RATIO,
    )

    lab_reference = rgb_to_lab(reference_rgb)
    lab_captured = rgb_to_lab(captured_rgb)
    lab_corrected = rgb_to_lab(corrected_chart_rgb)

    delta_e_before = delta_e_2000(lab_captured, lab_reference)
    delta_e_after = delta_e_2000(lab_corrected, lab_reference)

    target_specs = _get_target_specs(photo_bgr)

    target_results_with_masks = []
    annotated_before = photo_bgr.copy()
    annotated_after = corrected_photo_bgr.copy()

    for spec in target_specs:
        result = _process_one_target(
            index=spec["index"],
            input_label=spec["input_label"],
            target_roi=spec["roi"],
            photo_bgr=photo_bgr,
            corrected_photo_bgr=corrected_photo_bgr,
            out_dir=out_dir,
        )

        target_results_with_masks.append(result)

        mask = result["_mask"]
        label = result["input_label"]
        idx = result["index"]

        annotated_before = draw_roi_and_mask(
            annotated_before,
            tuple(result["roi_xyxy"]),
            mask,
            f"{idx}:{label} before",
        )

        annotated_after = draw_roi_and_mask(
            annotated_after,
            tuple(result["roi_xyxy"]),
            mask,
            f"{idx}:{label} after",
        )

    save_side_by_side(
        out_dir / "05_photo_before_after.png",
        annotated_before,
        annotated_after,
        "Before correction",
        "After correction",
    )

    save_side_by_side(
        out_dir / "06_chart_before_after.png",
        captured_chart_bgr,
        corrected_chart_bgr,
        "Captured chart",
        "Corrected chart",
    )

    save_delta_e_plot(
        out_dir / "08_chart_calibration_delta_e.png",
        delta_e_before,
        delta_e_after,
    )

    save_sample_csv(
        out_dir / "09_chart_samples.csv",
        captured_rgb,
        corrected_chart_rgb,
        reference_rgb,
        delta_e_before,
        delta_e_after,
    )

    target_results = []

    for item in target_results_with_masks:
        copied = dict(item)
        copied.pop("_mask", None)
        target_results.append(copied)

    save_target_validation_csv(
        out_dir / "11_target_validation.csv",
        target_results,
    )

    save_validation_bar_plot(
        out_dir / "12_target_validation_summary.png",
        target_results,
    )

    report = {
        "input": {
            "photo": args.photo,
            "standard_chart": args.standard,
        },
        "built_in_standard_lab_database": standards_as_rows(),
        "chart": {
            "rows": ROWS,
            "cols": COLS,
            "center_ratio": CENTER_RATIO,
            "corners_order": "top-left, top-right, bottom-right, bottom-left",
            "corners": corners.round(3).tolist(),
            "perspective_matrix": perspective_matrix.round(8).tolist(),
        },
        "model": {
            "type": MODEL,
            "ridge_alpha": RIDGE_ALPHA,
            "weights": W.tolist(),
        },
        "chart_delta_e_2000": {
            "before_each_patch": delta_e_before.round(4).tolist(),
            "after_each_patch": delta_e_after.round(4).tolist(),
            "before_mean": float(np.mean(delta_e_before)),
            "after_mean": float(np.mean(delta_e_after)),
            "before_median": float(np.median(delta_e_before)),
            "after_median": float(np.median(delta_e_after)),
            "before_max": float(np.max(delta_e_before)),
            "after_max": float(np.max(delta_e_after)),
            "improvement_mean": float(np.mean(delta_e_before) - np.mean(delta_e_after)),
        },
        "target_colors": target_results,
        "outputs": {
            "captured_chart_warped": str(out_dir / "01_captured_chart_warped.png"),
            "standard_chart": str(out_dir / "02_standard_chart.png"),
            "corrected_photo": str(out_dir / "03_corrected_photo.png"),
            "corrected_chart_warped": str(out_dir / "04_corrected_chart_warped.png"),
            "photo_before_after": str(out_dir / "05_photo_before_after.png"),
            "chart_before_after": str(out_dir / "06_chart_before_after.png"),
            "chart_delta_e_plot": str(out_dir / "08_chart_calibration_delta_e.png"),
            "chart_samples_csv": str(out_dir / "09_chart_samples.csv"),
            "target_validation_csv": str(out_dir / "11_target_validation.csv"),
            "target_validation_summary": str(out_dir / "12_target_validation_summary.png"),
            "report": str(out_dir / "report.json"),
        },
    }

    save_json(out_dir / "report.json", report)
    return report


# 结果输出

def print_summary(report: dict) -> None:
    chart_de = report["chart_delta_e_2000"]
    targets = report.get("target_colors") or []

    print("\n处理完成。")

    print("\n色卡校正模型评估：")
    print(f"  校正前 mean ΔE00: {chart_de['before_mean']:.4f}")
    print(f"  校正后 mean ΔE00: {chart_de['after_mean']:.4f}")
    print(f"  mean ΔE00 改善: {chart_de['improvement_mean']:.4f}")

    print("\n目标胶块验证结果：")

    for item in targets:
        std = item.get("standard") or {}
        de = item.get("delta_e_2000_to_standard")
        pred_after = (item.get("nearest_after") or [{}])[0]

        print(
            f"  [{item['index']}] {item['input_label']} "
            f"-> 标准: {std.get('code', 'N/A')} {std.get('name', '')}"
        )

        print("      校正前 Lab:", np.asarray(item["before_lab"]).round(3))
        print("      校正后 Lab:", np.asarray(item["after_lab"]).round(3))

        if de:
            print(
                f"      ΔE00 校正前/后: "
                f"{de['before']:.4f} -> {de['after']:.4f}，"
                f"改善 {de['improvement']:.4f}"
            )

        print(f"      阈值通过: {item.get('pass_after_threshold')}")

        print(
            f"      最近分类 after: "
            f"{pred_after.get('code', 'N/A')} {pred_after.get('name', '')} "
            f"(ΔE00={pred_after.get('delta_e_2000', float('nan')):.4f})"
        )

        print(f"      分类是否正确: {item.get('classification_correct_after')}")

    print("\n主要输出：")

    for key in [
        "corrected_photo",
        "photo_before_after",
        "chart_before_after",
        "chart_delta_e_plot",
        "chart_samples_csv",
        "target_validation_csv",
        "target_validation_summary",
        "report",
    ]:
        print(" ", report["outputs"][key])