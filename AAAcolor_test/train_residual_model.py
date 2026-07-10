from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from skimage import color
except Exception as e:
    raise RuntimeError("缺少 skimage，请先安装：pip install scikit-image") from e


FEATURE_NAMES = [
    "bias",
    "base_L", "base_a", "base_b",
    "root_L", "root_a", "root_b",
    "raw_L", "raw_a", "raw_b",
    "local_bg_L", "local_bg_a", "local_bg_b",
    "bg_minus_ref_L", "bg_minus_ref_a", "bg_minus_ref_b",
    "row_norm", "col_norm",
    "chroma", "sin_hue", "cos_hue",
    "is_gray", "is_red_orange", "is_yellow", "is_light", "is_dark",
]


def parse_lab(text: str) -> np.ndarray:
    text = str(text).strip().strip('"').strip("'")
    parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Lab格式错误: {text}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def read_standards(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            code = row[0].strip().upper()
            if code.lower() in {"code", "编号"}:
                continue
            name = row[1].strip()
            lab_text = ",".join(row[2:])
            try:
                lab = parse_lab(lab_text)
            except Exception:
                continue
            out[code] = {"code": code, "name": name, "lab": lab}
    if not out:
        raise RuntimeError(f"没有从标准CSV读到数据: {path}")
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fval(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def bval(v: Any) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return color.deltaE_ciede2000(np.asarray(lab1, dtype=np.float64), np.asarray(lab2, dtype=np.float64))


def stats(vals: list[float]) -> dict[str, float]:
    arr = np.array([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": np.nan, "median": np.nan, "max": np.nan, "p95": np.nan}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def parse_idx(row: dict[str, Any]) -> int:
    try:
        return int(float(row.get("idx", 1)))
    except Exception:
        return 1


def feature(row: dict[str, Any], ref_bg: np.ndarray) -> np.ndarray:
    # base = v2 的最终基线，一般就是 bg0.25 后的 final_Lab
    base_L = fval(row, "final_L", fval(row, "root_L"))
    base_a = fval(row, "final_a", fval(row, "root_a"))
    base_b = fval(row, "final_b", fval(row, "root_b"))

    root_L = fval(row, "root_L", base_L)
    root_a = fval(row, "root_a", base_a)
    root_b = fval(row, "root_b", base_b)

    raw_L = fval(row, "raw_L", root_L)
    raw_a = fval(row, "raw_a", root_a)
    raw_b = fval(row, "raw_b", root_b)

    bg_L = fval(row, "local_bg_L", ref_bg[0])
    bg_a = fval(row, "local_bg_a", ref_bg[1])
    bg_b = fval(row, "local_bg_b", ref_bg[2])

    idx = parse_idx(row)
    row_norm = ((idx - 1) // 16) / 7.0
    col_norm = ((idx - 1) % 16) / 15.0

    chroma = math.sqrt(base_a * base_a + base_b * base_b)
    hue = math.atan2(base_b, base_a)

    is_gray = 1.0 if chroma < 8 else 0.0
    is_red_orange = 1.0 if base_a > 8 and base_b > -5 else 0.0
    is_yellow = 1.0 if base_b > 18 else 0.0
    is_light = 1.0 if base_L > 70 else 0.0
    is_dark = 1.0 if base_L < 42 else 0.0

    return np.array([
        1.0,
        base_L / 100.0, base_a / 128.0, base_b / 128.0,
        root_L / 100.0, root_a / 128.0, root_b / 128.0,
        raw_L / 100.0, raw_a / 128.0, raw_b / 128.0,
        bg_L / 100.0, bg_a / 128.0, bg_b / 128.0,
        (bg_L - ref_bg[0]) / 100.0,
        (bg_a - ref_bg[1]) / 128.0,
        (bg_b - ref_bg[2]) / 128.0,
        row_norm, col_norm,
        chroma / 128.0, math.sin(hue), math.cos(hue),
        is_gray, is_red_orange, is_yellow, is_light, is_dark,
    ], dtype=np.float64)


def collect_samples(
    run_dirs: list[Path],
    standards: dict[str, dict[str, Any]],
    baseline_file: str,
    eval_count: int,
    ref_bg: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    records: list[dict[str, Any]] = []
    X, Y = [], []

    for rd in run_dirs:
        csv_path = rd / baseline_file
        if not csv_path.exists():
            alt = rd / "target_results_bg0p25_res0p00_family.csv"
            if alt.exists():
                csv_path = alt
            else:
                print(f"[skip] 找不到结果CSV: {csv_path}")
                continue

        rows = read_rows(csv_path)
        if eval_count > 0:
            rows = rows[:eval_count]

        good_count = 0
        for row in rows:
            code = str(row.get("code", "")).strip().upper()
            if code not in standards:
                continue

            std_lab = standards[code]["lab"]
            base_lab = np.array([
                fval(row, "final_L", fval(row, "root_L")),
                fval(row, "final_a", fval(row, "root_a")),
                fval(row, "final_b", fval(row, "root_b")),
            ], dtype=np.float64)

            # 过滤掉明显坏行
            if not np.all(np.isfinite(base_lab)):
                continue

            x = feature(row, ref_bg)
            y = std_lab - base_lab

            rec = dict(row)
            rec["_run"] = rd.name
            rec["_run_dir"] = str(rd)
            rec["_std_L"] = float(std_lab[0])
            rec["_std_a"] = float(std_lab[1])
            rec["_std_b"] = float(std_lab[2])
            rec["_base_L"] = float(base_lab[0])
            rec["_base_a"] = float(base_lab[1])
            rec["_base_b"] = float(base_lab[2])

            records.append(rec)
            X.append(x)
            Y.append(y)
            good_count += 1

        if good_count == 0:
            print(f"[skip] {rd} 没有有效样本")
        else:
            print(f"[ok] {rd}: {good_count} samples")

    if not X:
        raise RuntimeError("没有收集到任何训练样本")

    return records, np.vstack(X), np.vstack(Y)


def standardize_train(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std[std < 1e-8] = 1.0
    return (X - mean) / std, mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    reg = alpha * np.eye(X.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ Y)


def cap_shift(pred: np.ndarray, cap: np.ndarray) -> np.ndarray:
    return np.clip(pred, -cap, cap)


def nearest(lab: np.ndarray, standards: dict[str, dict[str, Any]]) -> tuple[str, str, float]:
    codes = list(standards.keys())
    labs = np.vstack([standards[c]["lab"] for c in codes])
    de = delta_e(lab[None, :], labs)
    i = int(np.argmin(de))
    c = codes[i]
    return c, standards[c]["name"], float(de[i])


def evaluate(records: list[dict[str, Any]], pred_shift: np.ndarray, standards: dict[str, dict[str, Any]], cap: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    before_des, base_des, model_des = [], [], []
    corrects, harm_before, harm_base = [], [], []

    pred_shift = cap_shift(pred_shift, cap)

    for rec, sh in zip(records, pred_shift):
        code = str(rec["code"]).strip().upper()
        std_lab = standards[code]["lab"]

        raw_lab = np.array([
            fval(rec, "raw_L", fval(rec, "root_L")),
            fval(rec, "raw_a", fval(rec, "root_a")),
            fval(rec, "raw_b", fval(rec, "root_b")),
        ], dtype=np.float64)

        base_lab = np.array([rec["_base_L"], rec["_base_a"], rec["_base_b"]], dtype=np.float64)
        model_lab = base_lab + sh
        model_lab[0] = np.clip(model_lab[0], 0, 100)
        model_lab[1:] = np.clip(model_lab[1:], -128, 127)

        before_de = float(delta_e(raw_lab[None, :], std_lab[None, :])[0])
        base_de = float(delta_e(base_lab[None, :], std_lab[None, :])[0])
        model_de = float(delta_e(model_lab[None, :], std_lab[None, :])[0])

        pred_code, pred_name, pred_de = nearest(model_lab, standards)
        ok = pred_code == code

        before_des.append(before_de)
        base_des.append(base_de)
        model_des.append(model_de)
        corrects.append(ok)
        harm_before.append(model_de > before_de)
        harm_base.append(model_de > base_de)

        out = {
            "run": rec["_run"],
            "idx": rec.get("idx", ""),
            "code": code,
            "name": rec.get("name", standards[code]["name"]),
            "before_deltaE": before_de,
            "base_deltaE": base_de,
            "model_deltaE": model_de,
            "base_improve_vs_before": before_de - base_de,
            "model_improve_vs_before": before_de - model_de,
            "model_improve_vs_base": base_de - model_de,
            "base_L": float(base_lab[0]), "base_a": float(base_lab[1]), "base_b": float(base_lab[2]),
            "shift_L": float(sh[0]), "shift_a": float(sh[1]), "shift_b": float(sh[2]),
            "model_L": float(model_lab[0]), "model_a": float(model_lab[1]), "model_b": float(model_lab[2]),
            "std_L": float(std_lab[0]), "std_a": float(std_lab[1]), "std_b": float(std_lab[2]),
            "pred_code": pred_code,
            "pred_name": pred_name,
            "pred_deltaE": pred_de,
            "correct": ok,
            "harm_vs_before": model_de > before_de,
            "harm_vs_base": model_de > base_de,
        }
        rows.append(out)

    s_before = stats(before_des)
    s_base = stats(base_des)
    s_model = stats(model_des)
    summary = {
        "count": len(records),
        "before_mean_deltaE": s_before["mean"],
        "base_mean_deltaE": s_base["mean"],
        "model_mean_deltaE": s_model["mean"],
        "before_median_deltaE": s_before["median"],
        "base_median_deltaE": s_base["median"],
        "model_median_deltaE": s_model["median"],
        "before_max_deltaE": s_before["max"],
        "base_max_deltaE": s_base["max"],
        "model_max_deltaE": s_model["max"],
        "model_p95_deltaE": s_model["p95"],
        "classification_acc": float(np.mean(corrects)),
        "harm_vs_before_rate": float(np.mean(harm_before)),
        "harm_vs_base_rate": float(np.mean(harm_base)),
    }
    return summary, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="训练历史 residual 模型：读取多个 dataset_runs/run_xxx/best_target_results.csv。")
    ap.add_argument("--runs-glob", default="dataset_runs/run_*")
    ap.add_argument("--runs", default="", help="逗号分隔 run 目录。填了则优先使用这个。")
    ap.add_argument("--standards-csv", required=True)
    ap.add_argument("--baseline-file", default="best_target_results.csv")
    ap.add_argument("--eval-count", type=int, default=128)
    ap.add_argument("--background-lab", default="84.71,-1.14,-3.64")
    ap.add_argument("--alpha-list", default="0.1,1,3,10,30,100,300")
    ap.add_argument("--cap-L", type=float, default=12.0)
    ap.add_argument("--cap-a", type=float, default=28.0)
    ap.add_argument("--cap-b", type=float, default=36.0)
    ap.add_argument("--out", default="residual_model_out")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.runs.strip():
        run_dirs = [Path(x.strip()) for x in args.runs.split(",") if x.strip()]
    else:
        run_dirs = [Path(x) for x in sorted(glob.glob(args.runs_glob)) if Path(x).is_dir()]

    # 自动跳过 run_011 这种空目录/无CSV目录，不报死。
    run_dirs = [r for r in run_dirs if r.exists()]

    if len(run_dirs) < 2:
        raise RuntimeError("至少需要 2 个有效 run，建议 5~15 个。")

    standards = read_standards(Path(args.standards_csv))
    ref_bg = parse_lab(args.background_lab)
    alphas = [float(x.strip()) for x in args.alpha_list.replace("，", ",").split(",") if x.strip()]
    cap = np.array([args.cap_L, args.cap_a, args.cap_b], dtype=np.float64)

    print("=== collect all samples ===")
    all_records, X_all, Y_all = collect_samples(run_dirs, standards, args.baseline_file, args.eval_count, ref_bg)

    # 过滤掉没有样本的 run
    valid_runs = sorted(set(Path(r["_run_dir"]) for r in all_records), key=lambda p: str(p))
    if len(valid_runs) < 2:
        raise RuntimeError("有效 run 少于 2 个。")

    print("\n=== leave-one-run-out alpha sweep ===")
    sweep_rows = []
    loo_rows_by_alpha: dict[float, list[dict[str, Any]]] = {}

    for alpha in alphas:
        pred_all = []
        rec_all = []

        for test_run in valid_runs:
            train_runs = [r for r in valid_runs if r != test_run]
            train_rec, X_train, Y_train = collect_samples(train_runs, standards, args.baseline_file, args.eval_count, ref_bg)
            test_rec, X_test, Y_test = collect_samples([test_run], standards, args.baseline_file, args.eval_count, ref_bg)

            Xs_train, mean, std = standardize_train(X_train)
            Xs_test = standardize_apply(X_test, mean, std)

            W = ridge_fit(Xs_train, Y_train, alpha)
            pred = Xs_test @ W

            pred_all.append(pred)
            rec_all.extend(test_rec)

        pred_all_np = np.vstack(pred_all)
        summary, rows = evaluate(rec_all, pred_all_np, standards, cap)
        summary["alpha"] = alpha
        sweep_rows.append(summary)
        loo_rows_by_alpha[alpha] = rows

        print(f"alpha={alpha:g} model_mean={summary['model_mean_deltaE']:.4f} base_mean={summary['base_mean_deltaE']:.4f} acc={summary['classification_acc']:.4f}")

    best = sorted(sweep_rows, key=lambda r: (r["model_mean_deltaE"], r["model_max_deltaE"]))[0]
    best_alpha = float(best["alpha"])

    write_csv(out_dir / "alpha_sweep_summary.csv", sweep_rows)
    write_csv(out_dir / "loo_predictions_best_alpha.csv", loo_rows_by_alpha[best_alpha])

    print("\n=== train final model on all runs ===")
    Xs_all, mean_all, std_all = standardize_train(X_all)
    W_final = ridge_fit(Xs_all, Y_all, best_alpha)
    pred_train = Xs_all @ W_final
    train_summary, train_rows = evaluate(all_records, pred_train, standards, cap)
    write_csv(out_dir / "train_predictions_full_model.csv", train_rows)

    model = {
        "type": "ridge_lab_residual_v1",
        "note": "历史 residual 模型。训练时使用历史标准色，部署时只读取模型预测，不用当前图标准值反推。",
        "feature_names": FEATURE_NAMES,
        "feature_mean": [float(x) for x in mean_all],
        "feature_std": [float(x) for x in std_all],
        "W": W_final.tolist(),
        "alpha": best_alpha,
        "cap": {"L": args.cap_L, "a": args.cap_a, "b": args.cap_b},
        "reference_bg_lab": [float(x) for x in ref_bg],
        "runs": [str(x) for x in valid_runs],
        "eval_count": args.eval_count,
        "loo_best_alpha_summary": best,
        "train_full_model_summary": train_summary,
    }
    save_json(out_dir / "residual_model.json", model)

    print("\n=== best LOO summary ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("\n=== full train summary, optimistic ===")
    print(json.dumps(train_summary, ensure_ascii=False, indent=2))
    print("\nSaved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
