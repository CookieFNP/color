from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def stat_pack(values):
    values = [safe_float(v) for v in values if safe_float(v) is not None]
    if not values:
        return {
            "mean": None,
            "median": None,
            "max": None,
            "p95": None,
            "std": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
    }


def get_nested(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def lab_diff(lab, std_lab):
    if not lab or not std_lab:
        return [None, None, None]

    try:
        lab = np.asarray(lab, dtype=np.float64)
        std_lab = np.asarray(std_lab, dtype=np.float64)
        if lab.size != 3 or std_lab.size != 3:
            return [None, None, None]
        diff = lab - std_lab
        return [float(diff[0]), float(diff[1]), float(diff[2])]
    except Exception:
        return [None, None, None]


def summarize_delta_lab(targets):
    before_diffs = []
    after_diffs = []

    for item in targets:
        std_lab = get_nested(item, "standard", "lab")
        before_lab = item.get("before_lab")
        after_lab = item.get("after_lab")

        b = lab_diff(before_lab, std_lab)
        a = lab_diff(after_lab, std_lab)

        if all(v is not None for v in b):
            before_diffs.append(b)
        if all(v is not None for v in a):
            after_diffs.append(a)

    def pack(rows):
        if not rows:
            return {
                "mean_dL": None,
                "mean_da": None,
                "mean_db": None,
                "mean_abs_dL": None,
                "mean_abs_da": None,
                "mean_abs_db": None,
            }

        arr = np.asarray(rows, dtype=np.float64)
        return {
            "mean_dL": float(np.mean(arr[:, 0])),
            "mean_da": float(np.mean(arr[:, 1])),
            "mean_db": float(np.mean(arr[:, 2])),
            "mean_abs_dL": float(np.mean(np.abs(arr[:, 0]))),
            "mean_abs_da": float(np.mean(np.abs(arr[:, 1]))),
            "mean_abs_db": float(np.mean(np.abs(arr[:, 2]))),
        }

    return {
        "before": pack(before_diffs),
        "after": pack(after_diffs),
    }


def summarize_targets(targets):
    before_values = []
    after_values = []
    correct_flags = []
    pass_flags = []

    for item in targets:
        de = item.get("delta_e_2000_to_standard") or {}

        before = safe_float(de.get("before"))
        after = safe_float(de.get("after"))

        if before is not None:
            before_values.append(before)
        if after is not None:
            after_values.append(after)

        if item.get("classification_correct_after") is not None:
            correct_flags.append(bool(item.get("classification_correct_after")))

        if item.get("pass_after_threshold") is not None:
            pass_flags.append(bool(item.get("pass_after_threshold")))

    before_arr = np.asarray(before_values, dtype=np.float64)
    after_arr = np.asarray(after_values, dtype=np.float64)

    if len(before_arr) == len(after_arr) and len(after_arr) > 0:
        harm_count = int(np.sum(after_arr > before_arr))
        mean_improvement = float(np.mean(before_arr - after_arr))
    else:
        harm_count = None
        mean_improvement = None

    total = len(after_arr)

    return {
        "count": total,
        "before_deltaE": stat_pack(before_values),
        "after_deltaE": stat_pack(after_values),
        "mean_improvement": mean_improvement,
        "harm_count": harm_count,
        "harm_rate": float(harm_count / total) if total and harm_count is not None else None,
        "classification_acc": float(sum(correct_flags) / len(correct_flags)) if correct_flags else None,
        "pass_rate_after_threshold": float(sum(pass_flags) / len(pass_flags)) if pass_flags else None,
    }


def infer_error_note(before_delta_lab, after_delta_lab, target_before_mean, target_after_mean):
    """
    简单给一条文字判断，方便快速看问题方向。
    """
    b = before_delta_lab or {}
    a = after_delta_lab or {}

    after_abs = {
        "L": safe_float(a.get("mean_abs_dL"), 0),
        "a": safe_float(a.get("mean_abs_da"), 0),
        "b": safe_float(a.get("mean_abs_db"), 0),
    }

    dominant = max(after_abs, key=after_abs.get)

    before_db = safe_float(b.get("mean_db"))
    after_db = safe_float(a.get("mean_db"))

    db_sign_flip = (
        before_db is not None
        and after_db is not None
        and before_db * after_db < 0
    )

    worsened = (
        target_before_mean is not None
        and target_after_mean is not None
        and target_after_mean > target_before_mean
    )

    if worsened and db_sign_flip:
        return "校正后整体变差，且 b 轴发生正负翻转，疑似黄蓝方向/白平衡过补偿"

    if worsened:
        return f"校正后整体变差，主要残差方向可能是 {dominant} 轴"

    return f"校正后整体改善，主要残差方向是 {dominant} 轴"


def read_one_report(report_path: Path):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    output_dir = report_path.parent.name

    model = report.get("model") or {}
    chart = report.get("chart_delta_e_2000") or {}
    wb = report.get("white_balance") or {}
    gains = wb.get("gains_rgb") or {}

    correction_control = report.get("correction_control") or {}

    targets = report.get("target_colors") or []

    target_summary = report.get("target_summary")
    if not isinstance(target_summary, dict) or not target_summary:
        target_summary = summarize_targets(targets)

    delta_lab_summary = report.get("delta_lab_summary")
    if not isinstance(delta_lab_summary, dict) or not delta_lab_summary:
        delta_lab_summary = summarize_delta_lab(targets)

    before_delta = target_summary.get("before_deltaE") or {}
    after_delta = target_summary.get("after_deltaE") or {}

    target_before_mean = safe_float(before_delta.get("mean"))
    target_after_mean = safe_float(after_delta.get("mean"))

    error_note = infer_error_note(
        delta_lab_summary.get("before") or {},
        delta_lab_summary.get("after") or {},
        target_before_mean,
        target_after_mean,
    )

    alpha_sweep = report.get("alpha_sweep") or []
    best_alpha = None
    best_alpha_mean = None
    best_alpha_harm_rate = None
    best_alpha_acc = None

    valid_alpha_rows = [
        row for row in alpha_sweep
        if safe_float(row.get("target_mean_deltaE")) is not None
    ]

    if valid_alpha_rows:
        best = min(valid_alpha_rows, key=lambda x: safe_float(x.get("target_mean_deltaE"), 1e9))
        best_alpha = safe_float(best.get("alpha"))
        best_alpha_mean = safe_float(best.get("target_mean_deltaE"))
        best_alpha_harm_rate = safe_float(best.get("harm_rate"))
        best_alpha_acc = safe_float(best.get("classification_acc"))

    summary_row = {
        "output_dir": output_dir,
        "report_path": str(report_path),
        "photo": get_nested(report, "input", "photo"),
        "standard_chart": get_nested(report, "input", "standard_chart"),

        "model": model.get("type"),
        "ridge_alpha": safe_float(model.get("ridge_alpha")),

        "white_balance_method": wb.get("method"),
        "white_balance_applied": wb.get("applied"),
        "R_gain": safe_float(gains.get("R_gain")),
        "G_gain": safe_float(gains.get("G_gain")),
        "B_gain": safe_float(gains.get("B_gain")),

        "correction_strength": safe_float(correction_control.get("correction_strength")),

        "chart_before_mean_deltaE": safe_float(chart.get("before_mean")),
        "chart_after_mean_deltaE": safe_float(chart.get("after_mean")),
        "chart_improvement_mean": safe_float(chart.get("improvement_mean")),
        "chart_after_median_deltaE": safe_float(chart.get("after_median")),
        "chart_after_max_deltaE": safe_float(chart.get("after_max")),
        "chart_after_p95_deltaE": safe_float(chart.get("after_p95")),

        "target_count": target_summary.get("count"),
        "target_before_mean_deltaE": target_before_mean,
        "target_after_mean_deltaE": target_after_mean,
        "target_mean_improvement": safe_float(target_summary.get("mean_improvement")),
        "target_after_median_deltaE": safe_float(after_delta.get("median")),
        "target_after_max_deltaE": safe_float(after_delta.get("max")),
        "target_after_p95_deltaE": safe_float(after_delta.get("p95")),
        "target_after_std_deltaE": safe_float(after_delta.get("std")),
        "harm_count": target_summary.get("harm_count"),
        "harm_rate": safe_float(target_summary.get("harm_rate")),
        "classification_acc": safe_float(target_summary.get("classification_acc")),
        "pass_rate_after_threshold": safe_float(target_summary.get("pass_rate_after_threshold")),

        "before_mean_dL": safe_float(get_nested(delta_lab_summary, "before", "mean_dL")),
        "before_mean_da": safe_float(get_nested(delta_lab_summary, "before", "mean_da")),
        "before_mean_db": safe_float(get_nested(delta_lab_summary, "before", "mean_db")),
        "after_mean_dL": safe_float(get_nested(delta_lab_summary, "after", "mean_dL")),
        "after_mean_da": safe_float(get_nested(delta_lab_summary, "after", "mean_da")),
        "after_mean_db": safe_float(get_nested(delta_lab_summary, "after", "mean_db")),
        "after_mean_abs_dL": safe_float(get_nested(delta_lab_summary, "after", "mean_abs_dL")),
        "after_mean_abs_da": safe_float(get_nested(delta_lab_summary, "after", "mean_abs_da")),
        "after_mean_abs_db": safe_float(get_nested(delta_lab_summary, "after", "mean_abs_db")),

        "best_sweep_alpha": best_alpha,
        "best_sweep_target_mean_deltaE": best_alpha_mean,
        "best_sweep_harm_rate": best_alpha_harm_rate,
        "best_sweep_classification_acc": best_alpha_acc,

        "error_note": error_note,
    }

    target_rows = []
    for item in targets:
        std = item.get("standard") or {}
        std_lab = std.get("lab")

        before_lab = item.get("before_lab")
        after_lab = item.get("after_lab")

        before_diff = lab_diff(before_lab, std_lab)
        after_diff = lab_diff(after_lab, std_lab)

        de = item.get("delta_e_2000_to_standard") or {}

        nearest_after = item.get("nearest_after") or []
        best_nearest = nearest_after[0] if len(nearest_after) >= 1 else {}
        second_nearest = nearest_after[1] if len(nearest_after) >= 2 else {}

        best_de = safe_float(best_nearest.get("delta_e_2000"))
        second_de = safe_float(second_nearest.get("delta_e_2000"))
        gap = second_de - best_de if best_de is not None and second_de is not None else None

        if best_de is not None and gap is not None:
            if best_de <= 3 and gap >= 1:
                confidence = "high"
            elif best_de <= 5:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = None

        before_de = safe_float(de.get("before"))
        after_de = safe_float(de.get("after"))

        target_rows.append({
            "output_dir": output_dir,
            "photo": get_nested(report, "input", "photo"),
            "model": model.get("type"),
            "ridge_alpha": safe_float(model.get("ridge_alpha")),
            "white_balance_method": wb.get("method"),
            "correction_strength": safe_float(correction_control.get("correction_strength")),

            "target_index": item.get("index"),
            "input_label": item.get("input_label"),
            "standard_code": std.get("code"),
            "standard_name": std.get("name"),

            "before_deltaE": before_de,
            "after_deltaE": after_de,
            "improvement": before_de - after_de if before_de is not None and after_de is not None else None,
            "harm": after_de > before_de if before_de is not None and after_de is not None else None,
            "pass_after_threshold": item.get("pass_after_threshold"),
            "classification_correct_after": item.get("classification_correct_after"),

            "standard_L": std_lab[0] if std_lab else None,
            "standard_a": std_lab[1] if std_lab else None,
            "standard_b": std_lab[2] if std_lab else None,

            "before_L": before_lab[0] if before_lab else None,
            "before_a": before_lab[1] if before_lab else None,
            "before_b": before_lab[2] if before_lab else None,

            "after_L": after_lab[0] if after_lab else None,
            "after_a": after_lab[1] if after_lab else None,
            "after_b": after_lab[2] if after_lab else None,

            "before_dL": before_diff[0],
            "before_da": before_diff[1],
            "before_db": before_diff[2],
            "after_dL": after_diff[0],
            "after_da": after_diff[1],
            "after_db": after_diff[2],

            "nearest_after_code": best_nearest.get("code"),
            "nearest_after_name": best_nearest.get("name"),
            "nearest_after_deltaE": best_de,
            "second_after_code": second_nearest.get("code"),
            "second_after_deltaE": second_de,
            "nearest_gap": gap,
            "match_confidence": confidence,
        })

    alpha_rows = []
    for row in alpha_sweep:
        alpha_rows.append({
            "output_dir": output_dir,
            "photo": get_nested(report, "input", "photo"),
            "model": model.get("type"),
            "ridge_alpha": safe_float(model.get("ridge_alpha")),
            "white_balance_method": wb.get("method"),
            "correction_strength_used_in_report": safe_float(correction_control.get("correction_strength")),
            "alpha": safe_float(row.get("alpha")),
            "target_mean_deltaE": safe_float(row.get("target_mean_deltaE")),
            "target_median_deltaE": safe_float(row.get("target_median_deltaE")),
            "target_max_deltaE": safe_float(row.get("target_max_deltaE")),
            "target_p95_deltaE": safe_float(row.get("target_p95_deltaE")),
            "harm_count": row.get("harm_count"),
            "harm_rate": safe_float(row.get("harm_rate")),
            "classification_acc": safe_float(row.get("classification_acc")),
        })

    return summary_row, target_rows, alpha_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="项目根目录，默认当前目录")
    parser.add_argument("--pattern", default="output_strong*/report.json", help="report.json 匹配规则")
    parser.add_argument("--out", default="collected_reports", help="输出汇总文件夹")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report_paths = sorted(
        Path(p).resolve()
        for p in glob.glob(str(root / args.pattern), recursive=True)
    )

    # 去重
    report_paths = sorted(set(report_paths))

    if not report_paths:
        print(f"没有找到 report.json。当前搜索路径：{root / args.pattern}")
        return

    summary_rows = []
    all_target_rows = []
    all_alpha_rows = []

    for path in report_paths:
        try:
            summary_row, target_rows, alpha_rows = read_one_report(path)
            summary_rows.append(summary_row)
            all_target_rows.extend(target_rows)
            all_alpha_rows.extend(alpha_rows)
        except Exception as e:
            print(f"跳过 {path}，原因：{e}")

    summary_df = pd.DataFrame(summary_rows)
    target_df = pd.DataFrame(all_target_rows)
    alpha_df = pd.DataFrame(all_alpha_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["target_after_mean_deltaE", "harm_rate", "target_after_p95_deltaE"],
            na_position="last"
        )

    if not target_df.empty:
        target_df = target_df.sort_values(
            by=["output_dir", "target_index"],
            na_position="last"
        )

    if not alpha_df.empty:
        alpha_df = alpha_df.sort_values(
            by=["output_dir", "target_mean_deltaE", "harm_rate"],
            na_position="last"
        )

    summary_csv = out_dir / "summary_reports.csv"
    target_csv = out_dir / "summary_targets.csv"
    alpha_csv = out_dir / "summary_alpha_sweep.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    target_df.to_csv(target_csv, index=False, encoding="utf-8-sig")
    alpha_df.to_csv(alpha_csv, index=False, encoding="utf-8-sig")

    print("\n已读取 report 数量：", len(report_paths))
    print("已保存：")
    print(" ", summary_csv)
    print(" ", target_csv)
    print(" ", alpha_csv)

    print("\n=== 总览排序：优先看 target_after_mean_deltaE / harm_rate / after_mean_db ===")
    show_cols = [
        "output_dir",
        "model",
        "white_balance_method",
        "correction_strength",
        "target_before_mean_deltaE",
        "target_after_mean_deltaE",
        "target_mean_improvement",
        "harm_rate",
        "classification_acc",
        "after_mean_dL",
        "after_mean_da",
        "after_mean_db",
        "best_sweep_alpha",
        "best_sweep_target_mean_deltaE",
        "best_sweep_harm_rate",
        "error_note",
    ]

    show_cols = [c for c in show_cols if c in summary_df.columns]
    print(summary_df[show_cols].to_string(index=False))

    if not alpha_df.empty:
        print("\n=== 每个输出文件夹的最佳 alpha ===")
        best_alpha_df = (
            alpha_df
            .dropna(subset=["target_mean_deltaE"])
            .sort_values(by=["output_dir", "target_mean_deltaE", "harm_rate"])
            .groupby("output_dir", as_index=False)
            .first()
        )

        best_cols = [
            "output_dir",
            "white_balance_method",
            "alpha",
            "target_mean_deltaE",
            "harm_rate",
            "classification_acc",
        ]
        best_cols = [c for c in best_cols if c in best_alpha_df.columns]
        print(best_alpha_df[best_cols].to_string(index=False))


if __name__ == "__main__":
    main()