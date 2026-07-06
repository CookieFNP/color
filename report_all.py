import json
import glob
import os
import numpy as np
import pandas as pd


def stat_pack(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {
            "mean": None,
            "median": None,
            "max": None,
            "p95": None,
            "std": None,
        }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
        "std": float(np.std(values)),
    }


rows = []

for report_path in glob.glob("output_*/report.json"):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    out_dir = os.path.dirname(report_path)

    model_info = report.get("model", {})
    model_type = model_info.get("type") or model_info.get("kind") or "unknown"
    ridge_alpha = model_info.get("ridge_alpha", model_info.get("ridge", None))

    chart = report.get("chart_delta_e_2000", {})
    chart_after = chart.get("after_each_patch", [])
    chart_before = chart.get("before_each_patch", [])

    chart_after_stat = stat_pack(chart_after)
    chart_before_stat = stat_pack(chart_before)

    targets = report.get("target_colors", [])
    target_before = []
    target_after = []
    correct_flags = []

    for item in targets:
        de = item.get("delta_e_2000_to_standard") or {}
        if "before" in de:
            target_before.append(de["before"])
        if "after" in de:
            target_after.append(de["after"])
        if item.get("classification_correct_after") is not None:
            correct_flags.append(bool(item.get("classification_correct_after")))

    target_before_stat = stat_pack(target_before)
    target_after_stat = stat_pack(target_after)

    pass_rate = None
    if len(correct_flags) > 0:
        pass_rate = sum(correct_flags) / len(correct_flags)

    rows.append({
        "output_dir": out_dir,
        "model": model_type,
        "ridge_alpha": ridge_alpha,

        "chart_before_mean": chart_before_stat["mean"],
        "chart_after_mean": chart_after_stat["mean"],
        "chart_after_median": chart_after_stat["median"],
        "chart_after_max": chart_after_stat["max"],
        "chart_after_p95": chart_after_stat["p95"],
        "chart_after_std": chart_after_stat["std"],

        "target_before_mean": target_before_stat["mean"],
        "target_after_mean": target_after_stat["mean"],
        "target_after_median": target_after_stat["median"],
        "target_after_max": target_after_stat["max"],
        "target_after_p95": target_after_stat["p95"],
        "target_after_std": target_after_stat["std"],
        "classification_acc": pass_rate,
    })


df = pd.DataFrame(rows)
df = df.sort_values(by=["target_after_mean", "target_after_p95"], na_position="last")

print(df.to_string(index=False))
df.to_csv("model_comparison_summary.csv", index=False, encoding="utf-8-sig")
print("\n已保存：model_comparison_summary.csv")