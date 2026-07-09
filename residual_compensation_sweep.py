from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.color_math import delta_e_2000
from src.standards import nearest_standards


def stat_pack(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": None,
            "median": None,
            "max": None,
            "p95": None,
            "std": None,
        }

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
    }


def parse_targets(report: dict) -> list[dict]:
    rows = []

    for item in report.get("target_colors", []):
        standard = item.get("standard") or {}

        code = standard.get("code")
        std_lab = np.asarray(standard.get("lab"), dtype=np.float64)
        after_lab = np.asarray(item.get("after_lab"), dtype=np.float64)

        delta = item.get("delta_lab_to_standard") or {}
        after_delta = np.asarray(delta.get("after"), dtype=np.float64)

        if std_lab.size != 3 or after_lab.size != 3 or after_delta.size != 3:
            continue

        rows.append(
            {
                "index": item.get("index"),
                "code": code,
                "name": standard.get("name", ""),
                "std_lab": std_lab,
                "after_lab": after_lab,
                "after_delta": after_delta,
                "before_de": float(
                    item.get("delta_e_2000_to_standard", {}).get("before", np.nan)
                ),
                "after_de": float(
                    item.get("delta_e_2000_to_standard", {}).get("after", np.nan)
                ),
            }
        )

    return rows


def make_channel_mask(mode: str) -> np.ndarray:
    """
    控制补偿哪些 Lab 通道。

    L:
        只补亮度

    b:
        只补黄蓝轴

    Lb:
        补亮度 + 黄蓝轴

    Lab:
        L/a/b 都补
    """
    mode = mode.lower()

    if mode == "l":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if mode == "b":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if mode == "lb":
        return np.array([1.0, 0.0, 1.0], dtype=np.float64)

    if mode == "lab":
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)

    raise ValueError(f"未知 channel_mode: {mode}")


def evaluate_global_bias(
    targets: list[dict],
    k: float,
    channel_mode: str,
) -> tuple[dict, list[dict]]:
    """
    乐观版：
    用当前 12 个胶块的平均残差反推补偿量，再评估这 12 个胶块。

    注意：
    这个有“看答案调参”的嫌疑，只用于验证方向有没有用。
    不能直接当最终正式算法。
    """
    residuals = np.asarray([t["after_delta"] for t in targets], dtype=np.float64)
    mean_residual = np.mean(residuals, axis=0)

    channel_mask = make_channel_mask(channel_mode)

    # after_delta = after_lab - standard_lab
    # 所以补偿方向是 -mean_residual
    base_shift = -mean_residual * channel_mask
    shift = float(k) * base_shift

    details = []
    de_values = []
    correct_flags = []

    for t in targets:
        lab2 = t["after_lab"] + shift
        std_lab = t["std_lab"]

        de2 = float(delta_e_2000(lab2[None, :], std_lab[None, :])[0])
        pred = nearest_standards(lab2, top_k=1)[0]
        correct = pred["code"] == t["code"]

        de_values.append(de2)
        correct_flags.append(correct)

        details.append(
            {
                "index": t["index"],
                "code": t["code"],
                "name": t["name"],
                "after_de": t["after_de"],
                "new_de": de2,
                "improvement": t["after_de"] - de2,
                "after_lab_L": float(t["after_lab"][0]),
                "after_lab_a": float(t["after_lab"][1]),
                "after_lab_b": float(t["after_lab"][2]),
                "new_lab_L": float(lab2[0]),
                "new_lab_a": float(lab2[1]),
                "new_lab_b": float(lab2[2]),
                "pred_code": pred["code"],
                "pred_de": pred["delta_e_2000"],
                "correct": correct,
            }
        )

    stats = stat_pack(de_values)

    summary = {
        "bias_mode": "global",
        "channel_mode": channel_mode,
        "k": float(k),
        "base_shift_L": float(base_shift[0]),
        "base_shift_a": float(base_shift[1]),
        "base_shift_b": float(base_shift[2]),
        "actual_shift_L": float(shift[0]),
        "actual_shift_a": float(shift[1]),
        "actual_shift_b": float(shift[2]),
        "mean_deltaE": stats["mean"],
        "median_deltaE": stats["median"],
        "max_deltaE": stats["max"],
        "p95_deltaE": stats["p95"],
        "std_deltaE": stats["std"],
        "classification_acc": float(sum(correct_flags) / len(correct_flags)),
    }

    return summary, details


def evaluate_leave_one_out_bias(
    targets: list[dict],
    k: float,
    channel_mode: str,
) -> tuple[dict, list[dict]]:
    """
    稍微更诚实一点的版本：
    对每个目标 i，用其它 11 个目标的平均残差估计补偿量，再评估 i。

    它仍然不是最终正式算法，但比 global 版少一点“自己看自己答案”的问题。
    """
    channel_mask = make_channel_mask(channel_mode)

    details = []
    de_values = []
    correct_flags = []
    shifts = []

    all_residuals = np.asarray([t["after_delta"] for t in targets], dtype=np.float64)

    for i, t in enumerate(targets):
        others = np.delete(all_residuals, i, axis=0)
        mean_residual = np.mean(others, axis=0)

        base_shift = -mean_residual * channel_mask
        shift = float(k) * base_shift
        shifts.append(shift)

        lab2 = t["after_lab"] + shift
        std_lab = t["std_lab"]

        de2 = float(delta_e_2000(lab2[None, :], std_lab[None, :])[0])
        pred = nearest_standards(lab2, top_k=1)[0]
        correct = pred["code"] == t["code"]

        de_values.append(de2)
        correct_flags.append(correct)

        details.append(
            {
                "index": t["index"],
                "code": t["code"],
                "name": t["name"],
                "after_de": t["after_de"],
                "new_de": de2,
                "improvement": t["after_de"] - de2,
                "actual_shift_L": float(shift[0]),
                "actual_shift_a": float(shift[1]),
                "actual_shift_b": float(shift[2]),
                "new_lab_L": float(lab2[0]),
                "new_lab_a": float(lab2[1]),
                "new_lab_b": float(lab2[2]),
                "pred_code": pred["code"],
                "pred_de": pred["delta_e_2000"],
                "correct": correct,
            }
        )

    stats = stat_pack(de_values)
    shifts_arr = np.asarray(shifts, dtype=np.float64)
    mean_shift = np.mean(shifts_arr, axis=0)

    summary = {
        "bias_mode": "leave_one_out",
        "channel_mode": channel_mode,
        "k": float(k),
        "actual_shift_L": float(mean_shift[0]),
        "actual_shift_a": float(mean_shift[1]),
        "actual_shift_b": float(mean_shift[2]),
        "mean_deltaE": stats["mean"],
        "median_deltaE": stats["median"],
        "max_deltaE": stats["max"],
        "p95_deltaE": stats["p95"],
        "std_deltaE": stats["std"],
        "classification_acc": float(sum(correct_flags) / len(correct_flags)),
    }

    return summary, details


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # 收集所有行里出现过的字段，避免不同类型 row 字段不一致时报错
    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--report",
        required=True,
        help="输入 report.json，例如 output_weighted_gray_light/report.json",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="输出目录，默认 report 所在目录/residual_sweep",
    )

    parser.add_argument(
        "--k-list",
        default="0,0.25,0.5,0.75,1.0,1.25",
        help="补偿强度列表，默认 0,0.25,0.5,0.75,1.0,1.25",
    )

    parser.add_argument(
        "--channel-modes",
        default="L,b,Lb,Lab",
        help="通道模式，默认 L,b,Lb,Lab",
    )

    args = parser.parse_args()

    report_path = Path(args.report).resolve()

    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        out_dir = report_path.parent / "residual_sweep"

    out_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    targets = parse_targets(report)

    if not targets:
        raise RuntimeError("report.json 中没有有效 target_colors")

    k_values = [
        float(x.strip())
        for x in args.k_list.replace("，", ",").split(",")
        if x.strip()
    ]

    channel_modes = [
        x.strip()
        for x in args.channel_modes.replace("，", ",").split(",")
        if x.strip()
    ]

    baseline_values = [t["after_de"] for t in targets]
    baseline_stat = stat_pack(baseline_values)
    baseline_acc = sum(
        nearest_standards(t["after_lab"], top_k=1)[0]["code"] == t["code"]
        for t in targets
    ) / len(targets)

    print("\nBaseline:")
    print(f"  mean ΔE = {baseline_stat['mean']:.4f}")
    print(f"  max  ΔE = {baseline_stat['max']:.4f}")
    print(f"  acc      = {baseline_acc:.4f}")

    summary_rows = []
    all_detail_rows = []

    for mode in channel_modes:
        for k in k_values:
            summary_g, details_g = evaluate_global_bias(
                targets=targets,
                k=k,
                channel_mode=mode,
            )

            summary_l, details_l = evaluate_leave_one_out_bias(
                targets=targets,
                k=k,
                channel_mode=mode,
            )

            summary_rows.append(summary_g)
            summary_rows.append(summary_l)

            for d in details_g:
                all_detail_rows.append(
                    {
                        "bias_mode": "global",
                        "channel_mode": mode,
                        "k": k,
                        **d,
                    }
                )

            for d in details_l:
                all_detail_rows.append(
                    {
                        "bias_mode": "leave_one_out",
                        "channel_mode": mode,
                        "k": k,
                        **d,
                    }
                )

    write_csv(out_dir / "residual_sweep_summary.csv", summary_rows)
    write_csv(out_dir / "residual_sweep_details.csv", all_detail_rows)

    valid_rows = [
        r for r in summary_rows
        if r["mean_deltaE"] is not None
    ]

    best_by_mean = sorted(valid_rows, key=lambda r: r["mean_deltaE"])[:8]
    best_by_max = sorted(valid_rows, key=lambda r: r["max_deltaE"])[:8]

    print("\nBest by mean ΔE:")
    for r in best_by_mean:
        print(
            f"  {r['bias_mode']:13s} | {r['channel_mode']:3s} | "
            f"k={r['k']:.2f} | mean={r['mean_deltaE']:.4f} | "
            f"max={r['max_deltaE']:.4f} | acc={r['classification_acc']:.4f} | "
            f"shift=({r.get('actual_shift_L', 0):.3f}, "
            f"{r.get('actual_shift_a', 0):.3f}, "
            f"{r.get('actual_shift_b', 0):.3f})"
        )

    print("\nBest by max ΔE:")
    for r in best_by_max:
        print(
            f"  {r['bias_mode']:13s} | {r['channel_mode']:3s} | "
            f"k={r['k']:.2f} | mean={r['mean_deltaE']:.4f} | "
            f"max={r['max_deltaE']:.4f} | acc={r['classification_acc']:.4f} | "
            f"shift=({r.get('actual_shift_L', 0):.3f}, "
            f"{r.get('actual_shift_a', 0):.3f}, "
            f"{r.get('actual_shift_b', 0):.3f})"
        )

    print("\n输出文件：")
    print(" ", out_dir / "residual_sweep_summary.csv")
    print(" ", out_dir / "residual_sweep_details.csv")


if __name__ == "__main__":
    main()