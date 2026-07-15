from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .calibration import apply_correction_to_image, fit_correction_model, predict_rgb
from .chart import extract_chart_means, warp_chart_from_photo
from .color_math import delta_e_2000, rgb_to_lab
from .glue_mask import build_glue_block_mask, get_glue_block_representative_rgb
from .interaction import select_four_points, select_roi
from .io_utils import imread_unicode, imwrite_unicode, read_json, write_json
from .reporting import save_delta_e_plot, save_side_by_side, save_target_validation_csv, stat_pack
from .standards import (
    confidence_from_nearest,
    load_standard_database,
    nearest_standards,
    parse_standard_sequence,
    resolve_standard,
    standards_as_rows,
)

ROWS = 4
COLS = 6
CENTER_RATIO = 0.50
CHART_OUTPUT_SIZE = (600, 400)  # width, height
TRIM_PERCENT = 10.0
VALIDATION_THRESHOLD = 5.0


def _load_or_select_chart_corners(photo_bgr: np.ndarray, out_dir: Path, force: bool) -> np.ndarray:
    path = out_dir / "chart_corners.json"
    if path.exists() and not force:
        obj = read_json(path)
        corners = np.asarray(obj["corners"], dtype=np.float32)
        if corners.shape == (4, 2):
            print(f"复用色卡四角：{path}")
            return corners

    corners = select_four_points(photo_bgr)
    write_json(path, {"corners": corners.tolist(), "order": "top-left, top-right, bottom-right, bottom-left"})
    print(f"已保存色卡四角：{path}")
    return corners


def _select_rois_128(photo_bgr: np.ndarray, labels: list[str], out_dir: Path, force: bool) -> list[dict]:
    path = out_dir / "rois_128.json"
    if path.exists() and not force:
        obj = read_json(path)
        rois = obj.get("rois") or []
        if len(rois) == len(labels):
            print(f"复用 {len(rois)} 个 ROI：{path}")
            return rois
        print(f"ROI 数量不一致：文件 {len(rois)} 个，当前需要 {len(labels)} 个，将重新框选。")

    specs: list[dict] = []
    print("\n开始按 data.csv 顺序框选胶块。")
    print("建议：一次认真框完，程序会保存 rois_128.json，下次可复用。")

    for idx, code in enumerate(labels, start=1):
        standard = resolve_standard(code)
        if standard is None:
            raise ValueError(f"标准色不存在：{code}")
        readable = f"{standard.code} {standard.name}"
        roi = select_roi(
            photo_bgr,
            window_name=f"ROI {idx:03d}/{len(labels):03d} {readable}",
            prompt=f"{idx}/{len(labels)} 框选 {readable} | Enter确认 | R重选 | Esc取消",
        )
        specs.append({"index": idx, "input_label": standard.code, "roi_xyxy": list(map(int, roi))})

        # 每 5 个保存一次，防止中途崩溃全丢
        if idx % 5 == 0 or idx == len(labels):
            write_json(path, {"source": "manual", "count": len(specs), "rois": specs})
            print(f"已临时保存 ROI：{idx}/{len(labels)}")

    return specs


def _process_one_target(
    *,
    spec: dict,
    photo_bgr: np.ndarray,
    corrected_photo_bgr: np.ndarray,
    out_dir: Path,
    top_k: int,
) -> dict:
    index = int(spec["index"])
    input_label = str(spec["input_label"])
    roi = tuple(map(int, spec["roi_xyxy"]))

    standard = resolve_standard(input_label)
    if standard is None:
        raise ValueError(f"未找到标准颜色：{input_label}")

    target_dir = out_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"target_{index:03d}_{standard.code}"

    mask = build_glue_block_mask(
        photo_bgr,
        roi,
        debug_path=target_dir / f"{prefix}_mask_debug.png",
    )

    before_rgb = get_glue_block_representative_rgb(photo_bgr, roi, mask=mask, trim_percent=TRIM_PERCENT)
    after_rgb = get_glue_block_representative_rgb(corrected_photo_bgr, roi, mask=mask, trim_percent=TRIM_PERCENT)

    before_lab = rgb_to_lab(before_rgb[None, :])[0]
    after_lab = rgb_to_lab(after_rgb[None, :])[0]
    ref_lab = np.asarray(standard.lab, dtype=np.float64)

    before_de = float(delta_e_2000(before_lab[None, :], ref_lab[None, :])[0])
    after_de = float(delta_e_2000(after_lab[None, :], ref_lab[None, :])[0])

    nearest_before = nearest_standards(before_lab, top_k=top_k)
    nearest_after = nearest_standards(after_lab, top_k=top_k)
    confidence = confidence_from_nearest(nearest_after)

    top1_correct = bool(nearest_after and nearest_after[0]["code"] == standard.code)
    top3_correct = bool(any(x["code"] == standard.code for x in nearest_after[:3]))

    x1, y1, x2, y2 = roi
    before_crop = photo_bgr[y1:y2, x1:x2]
    after_crop = corrected_photo_bgr[y1:y2, x1:x2]
    save_side_by_side(
        target_dir / f"{prefix}_before_after.png",
        before_crop,
        after_crop,
        f"Before {standard.code}",
        f"After {standard.code}",
    )

    return {
        "index": index,
        "input_label": input_label,
        "standard": {**standard.as_dict(), "source": "data.csv"},
        "roi_xyxy": list(map(int, roi)),
        "sampling_method": "manual ROI + glue mask + highlight/shadow filtering + trimmed mean",
        "trim_percent": TRIM_PERCENT,
        "valid_mask_pixels": int((mask > 0).sum()),
        "before_rgb": before_rgb.round(4).tolist(),
        "after_rgb": after_rgb.round(4).tolist(),
        "before_lab": before_lab.round(4).tolist(),
        "after_lab": after_lab.round(4).tolist(),
        "delta_e_2000_to_standard": {
            "reference_lab": ref_lab.round(4).tolist(),
            "before": before_de,
            "after": after_de,
            "improvement": before_de - after_de,
        },
        "validation_threshold": VALIDATION_THRESHOLD,
        "pass_after_threshold": after_de <= VALIDATION_THRESHOLD,
        "nearest_before": nearest_before,
        "nearest_after": nearest_after,
        "classification_correct_after": top1_correct,
        "top3_correct_after": top3_correct,
        "confidence": confidence,
        "outputs": {
            "target_mask_debug": str(target_dir / f"{prefix}_mask_debug.png"),
            "target_before_after": str(target_dir / f"{prefix}_before_after.png"),
        },
    }


def _summarize_targets(results: list[dict]) -> dict:
    before_de = [r["delta_e_2000_to_standard"]["before"] for r in results]
    after_de = [r["delta_e_2000_to_standard"]["after"] for r in results]
    top1 = [bool(r["classification_correct_after"]) for r in results]
    top3 = [bool(r["top3_correct_after"]) for r in results]
    passed = [bool(r["pass_after_threshold"]) for r in results]
    harmed = [r["delta_e_2000_to_standard"]["after"] > r["delta_e_2000_to_standard"]["before"] for r in results]

    return {
        "count": len(results),
        "before_delta_e": stat_pack(before_de),
        "after_delta_e": stat_pack(after_de),
        "mean_improvement": float(np.mean(np.asarray(before_de) - np.asarray(after_de))) if results else None,
        "top1_accuracy": float(np.mean(top1)) if results else None,
        "top3_accuracy": float(np.mean(top3)) if results else None,
        "pass_rate_delta_e_le_5": float(np.mean(passed)) if results else None,
        "harm_count": int(np.sum(harmed)) if results else 0,
        "harm_rate": float(np.mean(harmed)) if results else None,
        "confidence_counts": {
            level: int(sum(1 for r in results if r["confidence"]["level"] == level))
            for level in sorted({r["confidence"]["level"] for r in results})
        } if results else {},
    }


def run(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    standard_rows = load_standard_database(args.data)
    target_labels = parse_standard_sequence(args.target_sequence)

    photo_path = Path(args.photo)
    standard_chart_path = Path(args.standard)
    if not standard_chart_path.exists():
        raise FileNotFoundError(f"未找到标准 ColorChecker 图片：{standard_chart_path}")

    photo_bgr = imread_unicode(photo_path)
    standard_chart_bgr = imread_unicode(standard_chart_path)

    corners = _load_or_select_chart_corners(photo_bgr, out_dir, force=args.force_select_chart)

    warped_chart, matrix = warp_chart_from_photo(photo_bgr, corners, CHART_OUTPUT_SIZE)
    ref_chart, _ = warp_chart_from_photo(
        standard_chart_bgr,
        np.array(
            [
                [0, 0],
                [standard_chart_bgr.shape[1] - 1, 0],
                [standard_chart_bgr.shape[1] - 1, standard_chart_bgr.shape[0] - 1],
                [0, standard_chart_bgr.shape[0] - 1],
            ],
            dtype=np.float32,
        ),
        CHART_OUTPUT_SIZE,
    )

    captured_rgb = extract_chart_means(warped_chart, rows=ROWS, cols=COLS, center_ratio=CENTER_RATIO, trim_percent=TRIM_PERCENT)
    reference_rgb = extract_chart_means(ref_chart, rows=ROWS, cols=COLS, center_ratio=CENTER_RATIO, trim_percent=TRIM_PERCENT)

    W = fit_correction_model(captured_rgb, reference_rgb, model=args.model, ridge_alpha=args.ridge_alpha)
    corrected_photo_bgr = apply_correction_to_image(
        photo_bgr,
        W,
        model=args.model,
        correction_strength=args.correction_strength,
    )

    corrected_chart_rgb = predict_rgb(captured_rgb, W, model=args.model)
    if args.correction_strength < 1.0:
        # chart 指标也按 correction_strength 混合，和整图保持一致
        mixed = (
            (1 - args.correction_strength) * captured_rgb.astype(np.float64)
            + args.correction_strength * corrected_chart_rgb.astype(np.float64)
        )
        corrected_chart_rgb = np.clip(np.round(mixed), 0, 255).astype(np.uint8)

    captured_lab = rgb_to_lab(captured_rgb)
    reference_lab = rgb_to_lab(reference_rgb)
    corrected_chart_lab = rgb_to_lab(corrected_chart_rgb)

    chart_before_de = delta_e_2000(captured_lab, reference_lab)
    chart_after_de = delta_e_2000(corrected_chart_lab, reference_lab)

    imwrite_unicode(out_dir / "01_original.png", photo_bgr)
    imwrite_unicode(out_dir / "02_corrected.png", corrected_photo_bgr)
    imwrite_unicode(out_dir / "03_warped_chart.png", warped_chart)
    save_side_by_side(out_dir / "04_original_vs_corrected.png", photo_bgr, corrected_photo_bgr, "Original", "Corrected")
    save_delta_e_plot(out_dir / "05_chart_delta_e.png", chart_before_de, chart_after_de, "ColorChecker ΔE2000")

    roi_specs = _select_rois_128(photo_bgr, target_labels, out_dir, force=args.force_select_rois)

    target_results = []
    for spec in roi_specs:
        print(f"\n处理 {spec['index']}/{len(roi_specs)} {spec['input_label']}")
        result = _process_one_target(
            spec=spec,
            photo_bgr=photo_bgr,
            corrected_photo_bgr=corrected_photo_bgr,
            out_dir=out_dir,
            top_k=args.top_k,
        )
        target_results.append(result)
        de = result["delta_e_2000_to_standard"]["after"]
        top1 = result["nearest_after"][0]["code"] if result["nearest_after"] else "None"
        print(f"  after ΔE={de:.3f}, Top1={top1}, confidence={result['confidence']['level']}")

    save_target_validation_csv(out_dir / "06_target_validation.csv", target_results)

    target_before = np.asarray([r["delta_e_2000_to_standard"]["before"] for r in target_results], dtype=np.float64)
    target_after = np.asarray([r["delta_e_2000_to_standard"]["after"] for r in target_results], dtype=np.float64)
    if len(target_results) > 0:
        save_delta_e_plot(out_dir / "07_target_delta_e.png", target_before, target_after, "128 Glue Blocks ΔE2000")

    report = {
        "input": {
            "photo": str(photo_path),
            "standard_chart": str(standard_chart_path),
            "data_csv": str(Path(args.data)),
        },
        "standard_lab_database": {
            "count": len(standard_rows),
            "rows": standards_as_rows(),
        },
        "chart": {
            "rows": ROWS,
            "cols": COLS,
            "center_ratio": CENTER_RATIO,
            "trim_percent": TRIM_PERCENT,
            "corners_order": "top-left, top-right, bottom-right, bottom-left",
            "corners": corners.tolist(),
            "perspective_matrix": matrix.round(8).tolist(),
        },
        "model": {
            "type": args.model,
            "ridge_alpha": args.ridge_alpha,
            "correction_strength": args.correction_strength,
            "weights": W.round(10).tolist(),
        },
        "chart_delta_e_2000": {
            "before_each_patch": chart_before_de.round(4).tolist(),
            "after_each_patch": chart_after_de.round(4).tolist(),
            "before": stat_pack(chart_before_de),
            "after": stat_pack(chart_after_de),
            "improvement_mean": float(np.mean(chart_before_de) - np.mean(chart_after_de)),
        },
        "target_colors": target_results,
        "target_summary": _summarize_targets(target_results),
        "outputs": {
            "original": str(out_dir / "01_original.png"),
            "corrected": str(out_dir / "02_corrected.png"),
            "original_vs_corrected": str(out_dir / "04_original_vs_corrected.png"),
            "target_validation_csv": str(out_dir / "06_target_validation.csv"),
            "report_json": str(out_dir / "report.json"),
        },
    }

    write_json(out_dir / "report.json", report)
    return report


def print_summary(report: dict) -> None:
    chart = report["chart_delta_e_2000"]
    target = report["target_summary"]
    print("\n================ Summary ================")
    print(f"标准库数量：{report['standard_lab_database']['count']}")
    print(f"模型：{report['model']['type']}, ridge={report['model']['ridge_alpha']}, strength={report['model']['correction_strength']}")
    print(f"ColorChecker mean ΔE: {chart['before']['mean']:.3f} -> {chart['after']['mean']:.3f}")
    if target["count"]:
        print(f"目标数量：{target['count']}")
        print(f"Target mean ΔE: {target['before_delta_e']['mean']:.3f} -> {target['after_delta_e']['mean']:.3f}")
        print(f"Top1 accuracy: {target['top1_accuracy']:.3f}")
        print(f"Top3 accuracy: {target['top3_accuracy']:.3f}")
        print(f"Pass rate ΔE<=5: {target['pass_rate_delta_e_le_5']:.3f}")
        print(f"Harm rate: {target['harm_rate']:.3f}")
        print(f"Confidence counts: {target['confidence_counts']}")
    print(f"输出目录：{report['outputs']['report_json']}")
