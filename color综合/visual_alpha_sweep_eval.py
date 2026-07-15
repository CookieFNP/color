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
from visual_preview_grid import make_candidate, make_sheet


def frange_0_1(step: float = 0.1) -> list[float]:
    vals = []
    x = 0.0
    while x < 1.0 + 1e-9:
        vals.append(round(x, 6))
        x += step
    return vals


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


def evaluate_candidate_image(
    candidate_bgr: np.ndarray,
    target_colors: list[dict],
    trim_percent: float = 10.0,
) -> tuple[list[dict], dict]:
    """
    对 candidate 图重新测 128 个胶块的 representative Lab，
    再与各自真实 standard Lab 比较，得到 ΔE。
    """
    per_target_rows: list[dict] = []
    delta_es: list[float] = []

    for item in target_colors:
        roi = item.get("roi_xyxy")
        standard = item.get("standard") or {}
        std_lab = standard.get("lab")
        code = standard.get("code")
        name = standard.get("name")

        if not roi or not std_lab:
            continue

        roi_tuple = tuple(map(int, roi))

        mask = build_glue_block_mask(candidate_bgr, roi_tuple)
        rep_rgb = get_glue_block_representative_rgb(
            candidate_bgr,
            roi_tuple,
            mask=mask,
            trim_percent=trim_percent,
        )
        rep_lab = rgb_to_lab(rep_rgb.reshape(1, 3))[0]

        std_lab_arr = np.asarray(std_lab, dtype=np.float64).reshape(1, 3)
        rep_lab_arr = np.asarray(rep_lab, dtype=np.float64).reshape(1, 3)

        de = float(delta_e_2000(rep_lab_arr, std_lab_arr)[0])
        delta_es.append(de)

        per_target_rows.append(
            {
                "index": item.get("index"),
                "code": code,
                "name": name,
                "roi_x1": roi_tuple[0],
                "roi_y1": roi_tuple[1],
                "roi_x2": roi_tuple[2],
                "roi_y2": roi_tuple[3],
                "valid_mask_pixels": int((mask > 0).sum()),
                "standard_L": float(std_lab[0]),
                "standard_a": float(std_lab[1]),
                "standard_b": float(std_lab[2]),
                "measured_L": float(rep_lab[0]),
                "measured_a": float(rep_lab[1]),
                "measured_b": float(rep_lab[2]),
                "deltaE2000": de,
            }
        )

    summary = stat_pack(delta_es)
    return per_target_rows, summary


def save_per_target_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "index", "code", "name",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2", "valid_mask_pixels",
        "standard_L", "standard_a", "standard_b",
        "measured_L", "measured_a", "measured_b",
        "deltaE2000",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "alpha",
        "glue_ab_alpha",
        "bg_alpha",
        "l_alpha",
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
        for r in rows:
            writer.writerow(r)


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
    plt.title("Alpha sweep metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def sort_key_for_recommend(row: dict):
    """
    先看 mean，再看 p95，再看 max。
    你可以后面按自己的偏好再改。
    """
    return (
        row["mean_deltaE"],
        row["p95_deltaE"],
        row["max_deltaE"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep alpha for visual preview and evaluate 128-block ΔE.")
    parser.add_argument("--report", required=True, help="main.py 输出的 report.json")
    parser.add_argument("--photo", default=None, help="原图路径，不填则从 report 里读取")
    parser.add_argument("--alpha-step", type=float, default=0.1, help="alpha 步长，默认 0.1")
    parser.add_argument("--bg-scale", type=float, default=0.6, help="背景 alpha = alpha * bg_scale，默认 0.6")
    parser.add_argument("--l-alpha", type=float, default=0.0, help="亮度修正强度，建议默认 0")
    parser.add_argument("--feather", type=int, default=31)
    parser.add_argument("--bg-min-L", type=float, default=45.0)
    parser.add_argument("--bg-max-saturation", type=float, default=85.0)
    parser.add_argument("--thumb-width", type=int, default=350)
    parser.add_argument("--trim-percent", type=float, default=10.0)
    args = parser.parse_args()

    report_path = Path(args.report)
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    photo_path = Path(args.photo) if args.photo else Path(report["input"]["photo"])
    bgr = imread_unicode(photo_path)
    target_colors = report.get("target_colors") or []

    if not target_colors:
        raise RuntimeError("report.json 里没有 target_colors，无法做 alpha sweep。")

    out_dir = report_path.parent / "visual_alpha_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_list = frange_0_1(args.alpha_step)

    summary_rows: list[dict] = []
    sheet_images: list[np.ndarray] = []
    sheet_labels: list[str] = []

    print(f"开始 sweep alpha，共 {len(alpha_list)} 个：{alpha_list}")

    for alpha in alpha_list:
        glue_ab_alpha = float(alpha)
        bg_alpha = float(min(1.0, alpha * args.bg_scale))

        candidate = make_candidate(
            bgr=bgr,
            target_colors=target_colors,
            glue_ab_alpha=glue_ab_alpha,
            bg_alpha=bg_alpha,
            l_alpha=args.l_alpha,
            feather=args.feather,
            bg_min_L=args.bg_min_L,
            bg_max_saturation=args.bg_max_saturation,
        )

        preview_name = f"preview_alpha_{alpha:.2f}.png"
        preview_path = out_dir / preview_name
        imwrite_unicode(preview_path, candidate)

        per_target_rows, stats = evaluate_candidate_image(
            candidate_bgr=candidate,
            target_colors=target_colors,
            trim_percent=args.trim_percent,
        )

        per_target_csv = out_dir / f"targets_alpha_{alpha:.2f}.csv"
        save_per_target_csv(per_target_csv, per_target_rows)

        summary_row = {
            "alpha": alpha,
            "glue_ab_alpha": glue_ab_alpha,
            "bg_alpha": bg_alpha,
            "l_alpha": args.l_alpha,
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
            f"a={alpha:.2f}  "
            f"mean={stats['mean']:.2f}  "
            f"p95={stats['p95']:.2f}  "
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

    summary_rows_sorted = sorted(summary_rows, key=sort_key_for_recommend)

    save_summary_csv(out_dir / "alpha_sweep_summary.csv", summary_rows)
    save_metric_plot(out_dir / "alpha_sweep_metrics.png", summary_rows)

    sheet = make_sheet(
        images=sheet_images,
        labels=sheet_labels,
        cols=3,
        thumb_width=args.thumb_width,
    )
    imwrite_unicode(out_dir / "alpha_sweep_contact_sheet.png", sheet)

    result_json = {
        "report": str(report_path),
        "photo": str(photo_path),
        "settings": {
            "alpha_step": args.alpha_step,
            "bg_scale": args.bg_scale,
            "l_alpha": args.l_alpha,
            "feather": args.feather,
            "bg_min_L": args.bg_min_L,
            "bg_max_saturation": args.bg_max_saturation,
            "trim_percent": args.trim_percent,
        },
        "summary_rows": summary_rows,
        "recommended_by_metrics": summary_rows_sorted[:3],
        "note": (
            "recommended_by_metrics 只是按 mean/p95/max 数值排序的前 3 个 alpha，"
            "最终仍建议结合 alpha_sweep_contact_sheet.png 肉眼挑选。"
        ),
    }
    (out_dir / "alpha_sweep_result.json").write_text(
        json.dumps(result_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n==== 完成 ====")
    print("总览图：", out_dir / "alpha_sweep_contact_sheet.png")
    print("指标曲线：", out_dir / "alpha_sweep_metrics.png")
    print("汇总表：", out_dir / "alpha_sweep_summary.csv")
    print("推荐（按数值前 3）：")
    for i, row in enumerate(summary_rows_sorted[:3], start=1):
        print(
            f"{i}. alpha={row['alpha']:.2f}, "
            f"mean={row['mean_deltaE']:.3f}, "
            f"p95={row['p95_deltaE']:.3f}, "
            f"max={row['max_deltaE']:.3f}"
        )


if __name__ == "__main__":
    main()