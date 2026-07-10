from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def to_float(v: Any, default: float = math.nan) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"true", "1", "yes", "y", "是"}


def stat(values: list[float]) -> dict[str, float]:
    arr = sorted([float(x) for x in values if math.isfinite(float(x))])
    n = len(arr)
    if n == 0:
        return {"count": 0, "mean": math.nan, "median": math.nan, "max": math.nan, "p95": math.nan, "std": math.nan}
    mean = sum(arr) / n
    if n % 2 == 1:
        median = arr[n // 2]
    else:
        median = (arr[n // 2 - 1] + arr[n // 2]) / 2
    p95_idx = min(n - 1, max(0, int(round(0.95 * (n - 1)))))
    var = sum((x - mean) ** 2 for x in arr) / n
    return {
        "count": n,
        "mean": mean,
        "median": median,
        "max": max(arr),
        "p95": arr[p95_idx],
        "std": math.sqrt(var),
    }


def mean_bool(values: list[bool]) -> float:
    if not values:
        return math.nan
    return sum(1 for x in values if x) / len(values)


def find_candidate_row(candidate_rows: list[dict[str, str]], candidate_name: str) -> dict[str, str]:
    for row in candidate_rows:
        if row.get("candidate", "") == candidate_name:
            return row
    return {}


def find_contains_candidate(candidate_rows: list[dict[str, str]], text: str) -> dict[str, str]:
    for row in candidate_rows:
        if text in row.get("candidate", ""):
            return row
    return {}


def summarize_target_rows(rows: list[dict[str, str]], eval_count: int = 0) -> dict[str, Any]:
    if eval_count and eval_count > 0:
        rows = rows[:eval_count]

    before = [to_float(r.get("before_deltaE")) for r in rows]
    root = [to_float(r.get("root_deltaE")) for r in rows]
    final = [to_float(r.get("final_deltaE")) for r in rows]

    before_s = stat(before)
    root_s = stat(root)
    final_s = stat(final)

    correct = [to_bool(r.get("correct", "")) for r in rows]
    harm = [to_bool(r.get("harm", "")) for r in rows]
    low_conf = [str(r.get("confidence", "")).lower() == "low" for r in rows]

    return {
        "target_count": len(rows),
        "before_mean_deltaE_calc": before_s["mean"],
        "before_median_deltaE_calc": before_s["median"],
        "before_max_deltaE_calc": before_s["max"],
        "root_mean_deltaE_calc": root_s["mean"],
        "root_median_deltaE_calc": root_s["median"],
        "root_max_deltaE_calc": root_s["max"],
        "final_mean_deltaE_calc": final_s["mean"],
        "final_median_deltaE_calc": final_s["median"],
        "final_max_deltaE_calc": final_s["max"],
        "final_p95_deltaE_calc": final_s["p95"],
        "classification_acc_calc": mean_bool(correct),
        "harm_rate_calc": mean_bool(harm),
        "low_conf_rate_calc": mean_bool(low_conf),
    }


def candidate_short(row: dict[str, str], prefix: str) -> dict[str, Any]:
    if not row:
        return {
            f"{prefix}_candidate": "",
            f"{prefix}_mean": "",
            f"{prefix}_max": "",
            f"{prefix}_acc": "",
            f"{prefix}_harm": "",
        }
    return {
        f"{prefix}_candidate": row.get("candidate", ""),
        f"{prefix}_mean": to_float(row.get("final_mean_deltaE")),
        f"{prefix}_median": to_float(row.get("final_median_deltaE")),
        f"{prefix}_max": to_float(row.get("final_max_deltaE")),
        f"{prefix}_p95": to_float(row.get("final_p95_deltaE")),
        f"{prefix}_acc": to_float(row.get("classification_acc")),
        f"{prefix}_harm": to_float(row.get("harm_rate")),
        f"{prefix}_low_conf": to_float(row.get("low_conf_count")),
    }


def group_code_from_idx(idx: int, group_size: int = 16) -> str:
    start = ((idx - 1) // group_size) * group_size + 1
    end = start + group_size - 1
    return f"W{start:03d}-W{end:03d}"


def summarize_one_run(run_dir: Path, eval_count: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    report = read_json(run_dir / "report.json")
    candidate_rows = read_csv_dicts(run_dir / "candidate_summary.csv")
    best_rows = read_csv_dicts(run_dir / "best_target_results.csv")
    group_rows = read_csv_dicts(run_dir / "group_summary_best.csv")

    if eval_count and eval_count > 0:
        best_rows_eval = best_rows[:eval_count]
    else:
        best_rows_eval = best_rows

    chart = report.get("chart", {})
    best = report.get("best_candidate_for_diagnostic", {})
    local_bg = report.get("local_background", {})

    root_row = find_candidate_row(candidate_rows, "root_only")
    bg025_row = find_contains_candidate(candidate_rows, "bg0p25")
    best_cand_name = best.get("candidate", "")
    best_row_from_csv = find_candidate_row(candidate_rows, best_cand_name)

    calc_summary = summarize_target_rows(best_rows, eval_count=eval_count)

    run_summary: dict[str, Any] = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "photo": report.get("input", {}).get("photo", ""),
        "target_count_report": report.get("input", {}).get("target_count", ""),
        "eval_count_report": report.get("input", {}).get("eval_count", ""),
        "eval_count_used": eval_count if eval_count else len(best_rows_eval),
        "chart_before_mean": chart.get("before_mean_deltaE", ""),
        "chart_before_max": chart.get("before_max_deltaE", ""),
        "chart_root_mean": chart.get("root_after_mean_deltaE", ""),
        "chart_root_max": chart.get("root_after_max_deltaE", ""),
        "reference_bg_lab": json.dumps(local_bg.get("reference_bg_lab", ""), ensure_ascii=False),
        "global_bg_lab": json.dumps(local_bg.get("global_bg_lab", ""), ensure_ascii=False),
        "best_candidate_report": best_cand_name,
        "best_mean_report": best.get("final_mean_deltaE", ""),
        "best_median_report": best.get("final_median_deltaE", ""),
        "best_max_report": best.get("final_max_deltaE", ""),
        "best_acc_report": best.get("classification_acc", ""),
        "best_harm_report": best.get("harm_rate", ""),
    }
    run_summary.update(calc_summary)
    run_summary.update(candidate_short(root_row, "root_only"))
    run_summary.update(candidate_short(bg025_row, "bg025"))
    run_summary.update(candidate_short(best_row_from_csv, "best_csv"))

    # 每个 target 加 run 信息、分组信息
    target_all = []
    for r in best_rows_eval:
        rr = dict(r)
        rr["run"] = run_dir.name
        rr["run_dir"] = str(run_dir)
        try:
            idx = int(float(rr.get("idx", "")))
            rr["group"] = group_code_from_idx(idx)
        except Exception:
            rr["group"] = ""
        target_all.append(rr)

    # group summary 加 run 信息
    group_all = []
    if group_rows:
        for g in group_rows:
            gg = dict(g)
            gg["run"] = run_dir.name
            gg["run_dir"] = str(run_dir)
            group_all.append(gg)
    else:
        # 如果没有 group_summary_best.csv，就根据 best_rows 自己算每16个
        for start in range(0, len(best_rows_eval), 16):
            group = best_rows_eval[start:start + 16]
            if not group:
                continue
            before = [to_float(x.get("before_deltaE")) for x in group]
            root = [to_float(x.get("root_deltaE")) for x in group]
            final = [to_float(x.get("final_deltaE")) for x in group]
            group_all.append({
                "run": run_dir.name,
                "run_dir": str(run_dir),
                "group": group_code_from_idx(start + 1),
                "count": len(group),
                "before_mean_deltaE": stat(before)["mean"],
                "root_mean_deltaE": stat(root)["mean"],
                "final_mean_deltaE": stat(final)["mean"],
                "final_max_deltaE": stat(final)["max"],
                "classification_acc": mean_bool([to_bool(x.get("correct", "")) for x in group]),
                "harm_rate": mean_bool([to_bool(x.get("harm", "")) for x in group]),
            })

    candidate_all = []
    for c in candidate_rows:
        cc = dict(c)
        cc["run"] = run_dir.name
        cc["run_dir"] = str(run_dir)
        candidate_all.append(cc)

    return run_summary, target_all, group_all, candidate_all


def summarize_by_code(all_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_code: dict[str, list[dict[str, Any]]] = {}
    for r in all_targets:
        code = str(r.get("code", "")).strip().upper()
        if not code:
            continue
        by_code.setdefault(code, []).append(r)

    out = []
    for code, rows in sorted(by_code.items()):
        before = [to_float(r.get("before_deltaE")) for r in rows]
        root = [to_float(r.get("root_deltaE")) for r in rows]
        final = [to_float(r.get("final_deltaE")) for r in rows]
        pred_codes = [str(r.get("pred_code", "")) for r in rows]
        corrects = [to_bool(r.get("correct", "")) for r in rows]
        harms = [to_bool(r.get("harm", "")) for r in rows]

        final_s = stat(final)
        root_s = stat(root)
        before_s = stat(before)

        # 最常被预测成什么
        counts: dict[str, int] = {}
        for p in pred_codes:
            counts[p] = counts.get(p, 0) + 1
        most_pred = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0] if counts else ""

        out.append({
            "code": code,
            "name": rows[0].get("name", ""),
            "n_runs": len(rows),
            "before_mean_deltaE": before_s["mean"],
            "root_mean_deltaE": root_s["mean"],
            "final_mean_deltaE": final_s["mean"],
            "final_median_deltaE": final_s["median"],
            "final_max_deltaE": final_s["max"],
            "final_std_deltaE": final_s["std"],
            "mean_final_improvement_vs_before": before_s["mean"] - final_s["mean"] if math.isfinite(before_s["mean"]) and math.isfinite(final_s["mean"]) else "",
            "classification_acc": mean_bool(corrects),
            "harm_rate": mean_bool(harms),
            "most_pred_code": most_pred,
            "pred_codes": "|".join(pred_codes),
        })
    return out


def summarize_by_group(all_targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in all_targets:
        group = str(r.get("group", ""))
        if not group:
            continue
        by_group.setdefault(group, []).append(r)

    out = []
    for group, rows in sorted(by_group.items()):
        before = [to_float(r.get("before_deltaE")) for r in rows]
        root = [to_float(r.get("root_deltaE")) for r in rows]
        final = [to_float(r.get("final_deltaE")) for r in rows]
        out.append({
            "group": group,
            "n_samples": len(rows),
            "before_mean_deltaE": stat(before)["mean"],
            "root_mean_deltaE": stat(root)["mean"],
            "final_mean_deltaE": stat(final)["mean"],
            "final_max_deltaE": stat(final)["max"],
            "final_p95_deltaE": stat(final)["p95"],
            "classification_acc": mean_bool([to_bool(r.get("correct", "")) for r in rows]),
            "harm_rate": mean_bool([to_bool(r.get("harm", "")) for r in rows]),
        })
    return out


def make_text_report(run_summaries: list[dict[str, Any]], code_summary: list[dict[str, Any]], group_summary: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("=== Runs Summary ===")
    for r in run_summaries:
        lines.append(
            f'{r["run"]}: best={r.get("best_candidate_report","")} '
            f'mean={to_float(r.get("final_mean_deltaE_calc")):.4f}, '
            f'max={to_float(r.get("final_max_deltaE_calc")):.4f}, '
            f'acc={to_float(r.get("classification_acc_calc")):.4f}, '
            f'harm={to_float(r.get("harm_rate_calc")):.4f}, '
            f'chart_root_mean={to_float(r.get("chart_root_mean")):.4f}'
        )

    lines.append("")
    lines.append("=== Worst Colors by final_mean_deltaE ===")
    worst_colors = sorted(code_summary, key=lambda x: to_float(x.get("final_mean_deltaE")), reverse=True)[:20]
    for x in worst_colors:
        lines.append(
            f'{x["code"]} {x.get("name","")}: '
            f'final_mean={to_float(x.get("final_mean_deltaE")):.4f}, '
            f'root_mean={to_float(x.get("root_mean_deltaE")):.4f}, '
            f'acc={to_float(x.get("classification_acc")):.3f}, '
            f'most_pred={x.get("most_pred_code","")}'
        )

    lines.append("")
    lines.append("=== Groups ===")
    for g in group_summary:
        lines.append(
            f'{g["group"]}: final_mean={to_float(g.get("final_mean_deltaE")):.4f}, '
            f'max={to_float(g.get("final_max_deltaE")):.4f}, '
            f'acc={to_float(g.get("classification_acc")):.3f}, '
            f'harm={to_float(g.get("harm_rate")):.3f}'
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="汇总 real_scene_rootpoly_pipeline_v2.py 跑出来的多个 dataset_runs/run_* 目录。")
    p.add_argument("--runs-glob", default="dataset_runs/run_*", help="run 目录通配符")
    p.add_argument("--runs", default="", help="逗号分隔 run 目录；填了就优先用这个")
    p.add_argument("--eval-count", type=int, default=128, help="每个 run 统计前 N 个目标。你现在有效128就填128。")
    p.add_argument("--out", default="summary_out", help="输出目录")
    args = p.parse_args()

    if args.runs.strip():
        run_dirs = [Path(x.strip()) for x in args.runs.split(",") if x.strip()]
    else:
        run_dirs = [Path(x) for x in sorted(glob.glob(args.runs_glob)) if Path(x).is_dir()]

    if not run_dirs:
        raise RuntimeError(f"没有找到 run 目录: {args.runs_glob}")

    run_summaries = []
    all_targets = []
    all_groups_from_files = []
    all_candidates = []

    for rd in run_dirs:
        rs, targets, groups, candidates = summarize_one_run(rd, args.eval_count)
        run_summaries.append(rs)
        all_targets.extend(targets)
        all_groups_from_files.extend(groups)
        all_candidates.extend(candidates)

    code_summary = summarize_by_code(all_targets)
    group_summary = summarize_by_group(all_targets)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / "runs_summary.csv", run_summaries)
    write_csv(out_dir / "all_targets.csv", all_targets)
    write_csv(out_dir / "code_summary.csv", code_summary)
    write_csv(out_dir / "group_summary_all_runs.csv", group_summary)
    write_csv(out_dir / "group_summary_from_each_run_file.csv", all_groups_from_files)
    write_csv(out_dir / "candidate_summary_all_runs.csv", all_candidates)

    text = make_text_report(run_summaries, code_summary, group_summary)
    (out_dir / "summary_report.txt").write_text(text, encoding="utf-8")

    print(text)
    print("")
    print("已输出到：", out_dir.resolve())
    print("重点看：")
    print("  runs_summary.csv")
    print("  code_summary.csv")
    print("  group_summary_all_runs.csv")
    print("  summary_report.txt")


if __name__ == "__main__":
    main()
