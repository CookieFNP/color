# 主流程
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

from .calibration import (
    apply_correction_to_image,
    fit_correction_model,
    build_chart_sample_weights,
)
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
    save_alpha_sweep_csv,
)
from .standards import (
    nearest_standards,
    parse_standard_sequence,
    resolve_standard,
    standards_as_rows,
)
from .white_balance import (
    apply_rgb_gains_bgr,
    chart_gray_gains_from_samples,
    gains_to_report,
    gray_world_gains_from_bgr,
)


# 默认参数。命令行参数会覆盖这些默认值。
ROWS = 4
COLS = 6
CENTER_RATIO = 0.50

MODEL = "linear_bias"
RIDGE_ALPHA = 1e-6

CHART_SAMPLE_METHOD = "mean"
CHART_TRIM_PERCENT = 10.0

TRIM_PERCENT = 10.0
VALIDATION_THRESHOLD = 5.0

TARGET_SEQUENCE = "all"

WHITE_BALANCE = "none"
CORRECTION_STRENGTH = 1.0
ALPHA_SWEEP = "0,0.25,0.5,0.75,1"

# 加权 root polynomial 参数
# none       ：普通 root_poly2，24 个色卡点权重相同
# gray       ：提高灰阶块权重，强化白平衡/明度约束
# light      ：提高浅色块权重，强化浅黄/米色/浅灰约束
# gray_light ：灰阶 + 浅色同时加权，当前胶块场景建议优先试
CHART_WEIGHT_MODE = "none"
GRAY_WEIGHT = 4.0
LIGHT_WEIGHT = 2.5
LIGHT_L_THRESHOLD = 70.0


def _arg(args, name: str, default):
    return getattr(args, name, default)


def _parse_alpha_sweep(text: str | None) -> list[float]:
    """解析命令行传入的 alpha 列表。"""
    if text is None:
        return []

    raw = str(text).strip()
    if raw.lower() in {"", "none", "off", "false", "no"}:
        return []

    values: list[float] = []
    for part in raw.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))

    # 去重并限制在 [0, 1]
    values = sorted({round(float(np.clip(v, 0.0, 1.0)), 6) for v in values})
    return values


def _blend_bgr(before_bgr: np.ndarray, after_bgr: np.ndarray, alpha: float) -> np.ndarray:
    """
    校正强度混合。

    alpha=0：完全原图
    alpha=1：完整校正
    0<alpha<1：部分校正，用于验证强光下是否存在过校正。
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    mixed = before_bgr.astype(np.float32) * (1.0 - alpha) + after_bgr.astype(np.float32) * alpha
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _stat_pack(values: list[float] | np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "median": None, "max": None, "p95": None, "std": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
        "std": float(np.std(values)),
    }


def _summarize_target_results(target_results: list[dict]) -> dict:
    """汇总目标胶块 before/after ΔE、harm_rate、分类正确率。"""
    before_values = []
    after_values = []
    correct_flags = []

    for item in target_results:
        de = item.get("delta_e_2000_to_standard") or {}
        if de.get("before") is not None:
            before_values.append(float(de["before"]))
        if de.get("after") is not None:
            after_values.append(float(de["after"]))
        if item.get("classification_correct_after") is not None:
            correct_flags.append(bool(item.get("classification_correct_after")))

    before = np.asarray(before_values, dtype=np.float64)
    after = np.asarray(after_values, dtype=np.float64)

    harm_count = int(np.sum(after > before)) if before.size == after.size and before.size > 0 else 0
    total = int(after.size)

    return {
        "count": total,
        "before_deltaE": _stat_pack(before),
        "after_deltaE": _stat_pack(after),
        "mean_improvement": float(np.mean(before - after)) if before.size == after.size and before.size > 0 else None,
        "harm_count": harm_count,
        "harm_rate": float(harm_count / total) if total else None,
        "classification_acc": float(sum(correct_flags) / len(correct_flags)) if correct_flags else None,
    }


def _summarize_delta_lab(target_results: list[dict]) -> dict:
    """统计 Lab 三个通道的系统性偏差，判断问题主要来自 L/a/b 哪个方向。"""
    before = []
    after = []

    for item in target_results:
        diff = item.get("delta_lab_to_standard") or {}
        if diff.get("before") is not None:
            before.append(diff["before"])
        if diff.get("after") is not None:
            after.append(diff["after"])

    def pack(arr: list[list[float]]) -> dict:
        if not arr:
            return {}
        x = np.asarray(arr, dtype=np.float64)
        return {
            "mean_dL": float(np.mean(x[:, 0])),
            "mean_da": float(np.mean(x[:, 1])),
            "mean_db": float(np.mean(x[:, 2])),
            "mean_abs_dL": float(np.mean(np.abs(x[:, 0]))),
            "mean_abs_da": float(np.mean(np.abs(x[:, 1]))),
            "mean_abs_db": float(np.mean(np.abs(x[:, 2]))),
        }

    return {"before": pack(before), "after": pack(after)}


def _alpha_sweep_from_target_results(target_results: list[dict], alpha_values: list[float]) -> list[dict]:
    """
    离线评估不同校正强度 alpha。

    这里不重新处理整张图，而是在 Lab 空间近似模拟：
        mixed_lab = before_lab + alpha * (full_after_lab - before_lab)
    用来判断强光下是不是“完整校正过头”。
    """
    rows: list[dict] = []

    if not alpha_values:
        return rows

    for alpha in alpha_values:
        delta_e_values = []
        before_delta_e_values = []
        correct_flags = []

        for item in target_results:
            std = item.get("standard") or {}
            std_code = std.get("code")
            ref_lab = np.asarray(std.get("lab"), dtype=np.float32)
            before_lab = np.asarray(item.get("before_lab"), dtype=np.float32)
            full_after_lab = np.asarray(item.get("full_after_lab", item.get("after_lab")), dtype=np.float32)

            if ref_lab.size != 3 or before_lab.size != 3 or full_after_lab.size != 3:
                continue

            mixed_lab = before_lab + float(alpha) * (full_after_lab - before_lab)
            de = float(delta_e_2000(mixed_lab[None, :], ref_lab[None, :])[0])
            before_de = float(delta_e_2000(before_lab[None, :], ref_lab[None, :])[0])
            pred = nearest_standards(mixed_lab, top_k=1)[0]

            delta_e_values.append(de)
            before_delta_e_values.append(before_de)
            correct_flags.append(pred["code"] == std_code)

        values = np.asarray(delta_e_values, dtype=np.float64)
        before_values = np.asarray(before_delta_e_values, dtype=np.float64)
        total = int(values.size)
        harm_count = int(np.sum(values > before_values)) if total else 0

        rows.append({
            "alpha": float(alpha),
            "target_mean_deltaE": float(np.mean(values)) if total else None,
            "target_median_deltaE": float(np.median(values)) if total else None,
            "target_max_deltaE": float(np.max(values)) if total else None,
            "target_p95_deltaE": float(np.percentile(values, 95)) if total else None,
            "harm_count": harm_count,
            "harm_rate": float(harm_count / total) if total else None,
            "classification_acc": float(sum(correct_flags) / len(correct_flags)) if correct_flags else None,
        })

    return rows


def _apply_white_balance(
    *,
    method: str,
    photo_bgr: np.ndarray,
    captured_chart_bgr: np.ndarray,
    captured_rgb_raw: np.ndarray,
    reference_rgb: np.ndarray,
    out_dir: Path,
    chart_sample_method: str,
    chart_trim_percent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    根据 method 对整张图和色卡图做白平衡，返回：
        photo_for_model, chart_for_model, captured_rgb_for_model, white_balance_report
    """
    method = (method or "none").strip().lower()

    if method == "none":
        return photo_bgr, captured_chart_bgr, captured_rgb_raw, {
            "method": "none",
            "applied": False,
            "gains_rgb": {"R_gain": 1.0, "G_gain": 1.0, "B_gain": 1.0},
        }

    if method == "gray_world":
        gains = gray_world_gains_from_bgr(photo_bgr)
    elif method == "chart_gray":
        gains = chart_gray_gains_from_samples(captured_rgb_raw, reference_rgb)
    else:
        raise ValueError(f"未知 white_balance 方法：{method}")

    wb_photo_bgr = apply_rgb_gains_bgr(photo_bgr, gains)
    wb_chart_bgr = apply_rgb_gains_bgr(captured_chart_bgr, gains)

    cv2.imwrite(str(out_dir / "01b_white_balanced_photo.png"), wb_photo_bgr)
    cv2.imwrite(str(out_dir / "01c_white_balanced_chart_warped.png"), wb_chart_bgr)

    captured_rgb_wb = extract_chart_means(
        wb_chart_bgr,
        rows=ROWS,
        cols=COLS,
        center_ratio=CENTER_RATIO,
        sample_method=chart_sample_method,
        trim_percent=chart_trim_percent,
    )

    return wb_photo_bgr, wb_chart_bgr, captured_rgb_wb, {
        "method": method,
        "applied": True,
        "gains_rgb": gains_to_report(gains),
        "outputs": {
            "white_balanced_photo": str(out_dir / "01b_white_balanced_photo.png"),
            "white_balanced_chart_warped": str(out_dir / "01c_white_balanced_chart_warped.png"),
        },
    }


# 目标胶块 ROI 获取

def _get_target_specs(photo_bgr: np.ndarray, target_sequence: str = TARGET_SEQUENCE) -> list[dict]:
    """
    按内置标准顺序依次框选胶块。
    默认顺序：
    W015 -> W016 -> W031 -> W032 -> W047 -> W048
    -> W063 -> W064 -> W079 -> W080 -> W095 -> W096
    """
    labels = parse_standard_sequence(target_sequence)
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
    trim_percent: float,
    validation_threshold: float,
    full_corrected_photo_bgr: np.ndarray | None = None,
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
        trim_percent=trim_percent,
    )

    target_after_rgb = get_glue_block_representative_rgb(
        corrected_photo_bgr,
        target_roi,
        mask=target_mask,
        trim_percent=trim_percent,
    )

    if full_corrected_photo_bgr is None:
        full_corrected_photo_bgr = corrected_photo_bgr

    target_full_after_rgb = get_glue_block_representative_rgb(
        full_corrected_photo_bgr,
        target_roi,
        mask=target_mask,
        trim_percent=trim_percent,
    )

    target_before_lab = rgb_to_lab(target_before_rgb[None, :])[0]
    target_after_lab = rgb_to_lab(target_after_rgb[None, :])[0]
    target_full_after_lab = rgb_to_lab(target_full_after_rgb[None, :])[0]

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

    full_after_de = float(
        delta_e_2000(
            target_full_after_lab[None, :],
            reference_lab[None, :],
        )[0]
    )

    before_delta_lab = target_before_lab - reference_lab
    after_delta_lab = target_after_lab - reference_lab
    full_after_delta_lab = target_full_after_lab - reference_lab

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
        "trim_percent": float(trim_percent),
        "valid_mask_pixels": int(np.sum(target_mask > 0)),
        "before_rgb": target_before_rgb.round(3).tolist(),
        "after_rgb": target_after_rgb.round(3).tolist(),
        "full_after_rgb": target_full_after_rgb.round(3).tolist(),
        "before_lab": target_before_lab.round(3).tolist(),
        "after_lab": target_after_lab.round(3).tolist(),
        "full_after_lab": target_full_after_lab.round(3).tolist(),
        "delta_lab_to_standard": {
            "before": before_delta_lab.round(3).tolist(),
            "after": after_delta_lab.round(3).tolist(),
            "full_after": full_after_delta_lab.round(3).tolist(),
        },
        "delta_e_2000_to_standard": {
            "reference_lab": reference_lab.round(4).tolist(),
            "before": before_de,
            "after": after_de,
            "improvement": before_de - after_de,
        },
        "delta_e_2000_full_correction": {
            "after_full": full_after_de,
            "improvement_full": before_de - full_after_de,
        },
        "validation_threshold": float(validation_threshold),
        "pass_after_threshold": after_de <= validation_threshold,
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

    model = _arg(args, "model", MODEL)
    ridge_alpha = float(_arg(args, "ridge_alpha", RIDGE_ALPHA))
    chart_sample_method = _arg(args, "chart_sample_method", CHART_SAMPLE_METHOD)
    chart_trim_percent = float(_arg(args, "chart_trim_percent", CHART_TRIM_PERCENT))
    target_trim_percent = float(_arg(args, "target_trim_percent", TRIM_PERCENT))
    validation_threshold = float(_arg(args, "threshold", VALIDATION_THRESHOLD))
    target_sequence = _arg(args, "target_sequence", TARGET_SEQUENCE)
    white_balance_method = _arg(args, "white_balance", WHITE_BALANCE)
    correction_strength = float(np.clip(_arg(args, "correction_strength", CORRECTION_STRENGTH), 0.0, 1.0))
    alpha_values = _parse_alpha_sweep(_arg(args, "alpha_sweep", ALPHA_SWEEP))

    chart_weight_mode = str(_arg(args, "chart_weight_mode", CHART_WEIGHT_MODE)).strip().lower()
    gray_weight = float(_arg(args, "gray_weight", GRAY_WEIGHT))
    light_weight = float(_arg(args, "light_weight", LIGHT_WEIGHT))
    light_l_threshold = float(_arg(args, "light_l_threshold", LIGHT_L_THRESHOLD))

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

    captured_rgb_raw = extract_chart_means(
        captured_chart_bgr,
        rows=ROWS,
        cols=COLS,
        center_ratio=CENTER_RATIO,
        sample_method=chart_sample_method,
        trim_percent=chart_trim_percent,
    )

    reference_rgb = extract_chart_means(
        standard_bgr,
        rows=ROWS,
        cols=COLS,
        center_ratio=CENTER_RATIO,
        sample_method=chart_sample_method,
        trim_percent=chart_trim_percent,
    )

    photo_for_model_bgr, chart_for_model_bgr, captured_rgb, white_balance_report = _apply_white_balance(
        method=white_balance_method,
        photo_bgr=photo_bgr,
        captured_chart_bgr=captured_chart_bgr,
        captured_rgb_raw=captured_rgb_raw,
        reference_rgb=reference_rgb,
        out_dir=out_dir,
        chart_sample_method=chart_sample_method,
        chart_trim_percent=chart_trim_percent,
    )

    chart_sample_weights = None

    if chart_weight_mode != "none":
        chart_sample_weights = build_chart_sample_weights(
            reference_rgb,
            mode=chart_weight_mode,
            gray_weight=gray_weight,
            light_weight=light_weight,
            light_l_threshold=light_l_threshold,
            normalize=True,
        )

    W = fit_correction_model(
        captured_rgb,
        reference_rgb,
        model=model,
        ridge_alpha=ridge_alpha,
        sample_weights=chart_sample_weights,
    )

    full_corrected_photo_bgr = apply_correction_to_image(
        photo_for_model_bgr,
        W,
        model=model,
    )

    full_corrected_chart_bgr = apply_correction_to_image(
        chart_for_model_bgr,
        W,
        model=model,
    )

    corrected_photo_bgr = _blend_bgr(photo_bgr, full_corrected_photo_bgr, correction_strength)
    corrected_chart_bgr = _blend_bgr(captured_chart_bgr, full_corrected_chart_bgr, correction_strength)

    cv2.imwrite(str(out_dir / "03_corrected_photo.png"), corrected_photo_bgr)
    cv2.imwrite(str(out_dir / "03b_full_corrected_photo.png"), full_corrected_photo_bgr)
    cv2.imwrite(str(out_dir / "04_corrected_chart_warped.png"), corrected_chart_bgr)
    cv2.imwrite(str(out_dir / "04b_full_corrected_chart_warped.png"), full_corrected_chart_bgr)

    corrected_chart_rgb = extract_chart_means(
        corrected_chart_bgr,
        rows=ROWS,
        cols=COLS,
        center_ratio=CENTER_RATIO,
        sample_method=chart_sample_method,
        trim_percent=chart_trim_percent,
    )

    lab_reference = rgb_to_lab(reference_rgb)
    lab_captured = rgb_to_lab(captured_rgb)
    lab_corrected = rgb_to_lab(corrected_chart_rgb)

    delta_e_before = delta_e_2000(lab_captured, lab_reference)
    delta_e_after = delta_e_2000(lab_corrected, lab_reference)

    target_specs = _get_target_specs(photo_bgr, target_sequence=target_sequence)

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
            full_corrected_photo_bgr=full_corrected_photo_bgr,
            out_dir=out_dir,
            trim_percent=target_trim_percent,
            validation_threshold=validation_threshold,
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

    target_summary = _summarize_target_results(target_results)
    delta_lab_summary = _summarize_delta_lab(target_results)
    alpha_sweep = _alpha_sweep_from_target_results(target_results, alpha_values)

    if alpha_sweep:
        save_alpha_sweep_csv(out_dir / "13_alpha_sweep_summary.csv", alpha_sweep)

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
            "sample_method": chart_sample_method,
            "trim_percent": chart_trim_percent,
            "corners_order": "top-left, top-right, bottom-right, bottom-left",
            "corners": corners.round(3).tolist(),
            "perspective_matrix": perspective_matrix.round(8).tolist(),
        },
        "model": {
            "type": model,
            "ridge_alpha": ridge_alpha,
            "weights": W.tolist(),
            "chart_weight_mode": chart_weight_mode,
            "chart_sample_weights": (
                None
                if chart_sample_weights is None
                else np.asarray(chart_sample_weights, dtype=np.float32).round(6).tolist()
            ),
            "chart_weight_params": {
                "gray_weight": gray_weight,
                "light_weight": light_weight,
                "light_l_threshold": light_l_threshold,
                "note": (
                    "Weighted root polynomial. "
                    "Weights are normalized to mean 1 before fitting. "
                    "Only ColorChecker patches are weighted; target glue standards are not used for fitting."
                ),
            },
        },
        "white_balance": white_balance_report,
        "correction_control": {
            "correction_strength": correction_strength,
            "alpha_sweep_values": alpha_values,
            "note": "correction_strength=1 means full correction; 0 means original image; alpha_sweep is evaluated in Lab space from before_lab to full_after_lab.",
        },
        "chart_delta_e_2000": {
            "before_each_patch": delta_e_before.round(4).tolist(),
            "after_each_patch": delta_e_after.round(4).tolist(),
            "before_mean": float(np.mean(delta_e_before)),
            "after_mean": float(np.mean(delta_e_after)),
            "before_median": float(np.median(delta_e_before)),
            "after_median": float(np.median(delta_e_after)),
            "before_p95": float(np.percentile(delta_e_before, 95)),
            "after_p95": float(np.percentile(delta_e_after, 95)),
            "before_max": float(np.max(delta_e_before)),
            "after_max": float(np.max(delta_e_after)),
            "improvement_mean": float(np.mean(delta_e_before) - np.mean(delta_e_after)),
        },
        "target_summary": target_summary,
        "delta_lab_summary": delta_lab_summary,
        "alpha_sweep": alpha_sweep,
        "target_colors": target_results,
        "outputs": {
            "captured_chart_warped": str(out_dir / "01_captured_chart_warped.png"),
            "standard_chart": str(out_dir / "02_standard_chart.png"),
            "corrected_photo": str(out_dir / "03_corrected_photo.png"),
            "full_corrected_photo": str(out_dir / "03b_full_corrected_photo.png"),
            "corrected_chart_warped": str(out_dir / "04_corrected_chart_warped.png"),
            "full_corrected_chart_warped": str(out_dir / "04b_full_corrected_chart_warped.png"),
            "photo_before_after": str(out_dir / "05_photo_before_after.png"),
            "chart_before_after": str(out_dir / "06_chart_before_after.png"),
            "chart_delta_e_plot": str(out_dir / "08_chart_calibration_delta_e.png"),
            "chart_samples_csv": str(out_dir / "09_chart_samples.csv"),
            "target_validation_csv": str(out_dir / "11_target_validation.csv"),
            "target_validation_summary": str(out_dir / "12_target_validation_summary.png"),
            "alpha_sweep_summary_csv": str(out_dir / "13_alpha_sweep_summary.csv"),
            "report": str(out_dir / "report.json"),
        },
    }

    save_json(out_dir / "report.json", report)
    return report


# 结果输出

def print_summary(report: dict) -> None:
    chart_de = report["chart_delta_e_2000"]
    targets = report.get("target_colors") or []
    model = report.get("model") or {}
    chart = report.get("chart") or {}
    white_balance = report.get("white_balance") or {}
    correction_control = report.get("correction_control") or {}
    target_summary = report.get("target_summary") or {}
    delta_lab_summary = report.get("delta_lab_summary") or {}
    alpha_sweep = report.get("alpha_sweep") or []

    print("\n处理完成。")

    print("\n色卡校正模型评估：")
    print(
        f"  model: {model.get('type')} | ridge_alpha: {model.get('ridge_alpha')} "
        f"| chart_weight_mode: {model.get('chart_weight_mode', 'none')}"
    )
    if model.get("chart_sample_weights") is not None:
        print(f"  chart_sample_weights: {model.get('chart_sample_weights')}")
    print(f"  chart sampling: {chart.get('sample_method')} | trim_percent: {chart.get('trim_percent')}")
    print(f"  white_balance: {white_balance.get('method')} | gains: {white_balance.get('gains_rgb')}")
    print(f"  correction_strength alpha: {correction_control.get('correction_strength')}")
    print(f"  校正前 mean ΔE00: {chart_de['before_mean']:.4f}")
    print(f"  校正后 mean ΔE00: {chart_de['after_mean']:.4f}")
    print(f"  校正前/后 P95 ΔE00: {chart_de.get('before_p95', float('nan')):.4f} -> {chart_de.get('after_p95', float('nan')):.4f}")
    print(f"  校正前/后 max ΔE00: {chart_de['before_max']:.4f} -> {chart_de['after_max']:.4f}")
    print(f"  mean ΔE00 改善: {chart_de['improvement_mean']:.4f}")

    print("\n目标胶块汇总：")
    if target_summary:
        before_stat = target_summary.get("before_deltaE") or {}
        after_stat = target_summary.get("after_deltaE") or {}
        print(f"  mean ΔE00: {before_stat.get('mean', float('nan')):.4f} -> {after_stat.get('mean', float('nan')):.4f}")
        print(f"  max  ΔE00: {before_stat.get('max', float('nan')):.4f} -> {after_stat.get('max', float('nan')):.4f}")
        print(f"  harm_rate: {target_summary.get('harm_rate', float('nan')):.4f} | classification_acc: {target_summary.get('classification_acc', float('nan')):.4f}")

    if delta_lab_summary:
        b = delta_lab_summary.get("before") or {}
        a = delta_lab_summary.get("after") or {}
        print("  平均 Lab 偏差 before: ", {k: round(v, 3) for k, v in b.items() if k.startswith("mean_d")})
        print("  平均 Lab 偏差 after : ", {k: round(v, 3) for k, v in a.items() if k.startswith("mean_d")})

    if alpha_sweep:
        best = min(alpha_sweep, key=lambda row: row.get("target_mean_deltaE") if row.get("target_mean_deltaE") is not None else 1e9)
        print(f"  alpha_sweep 最低 mean ΔE00: alpha={best.get('alpha')} | mean={best.get('target_mean_deltaE'):.4f} | harm_rate={best.get('harm_rate'):.4f}")

    print("\n目标胶块逐项结果：")

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
        if item.get("delta_lab_to_standard"):
            print("      ΔLab before:", np.asarray(item["delta_lab_to_standard"].get("before")).round(3))
            print("      ΔLab after :", np.asarray(item["delta_lab_to_standard"].get("after")).round(3))

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
        "full_corrected_photo",
        "photo_before_after",
        "chart_before_after",
        "chart_delta_e_plot",
        "chart_samples_csv",
        "target_validation_csv",
        "target_validation_summary",
        "alpha_sweep_summary_csv",
        "report",
    ]:
        print(" ", report["outputs"][key])
