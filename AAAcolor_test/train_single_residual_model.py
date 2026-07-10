from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from skimage import color


FEATURE_NAMES = [
    "bias",
    "base_L", "base_a", "base_b",
    "root_L", "root_a", "root_b",
    "raw_L", "raw_a", "raw_b",
    "local_bg_L", "local_bg_a", "local_bg_b",
    "bg_minus_ref_L", "bg_minus_ref_a", "bg_minus_ref_b",
    "chroma", "sin_hue", "cos_hue",
    "is_gray", "is_red_orange", "is_yellow", "is_light", "is_dark",
    "baseL_x_chroma",
    "basea_x_baseb",
]

# 注意：这个 single 版本故意不使用 idx / row / col / code 作为特征。
# 训练时用 code 查标准 Lab 是为了产生监督标签；
# 部署识别未知单个胶块时，不需要知道 code、位置或行列。


def parse_lab(text: str) -> np.ndarray:
    text = str(text).strip().strip('"').strip("'")
    parts = [p.strip() for p in text.replace("，", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Lab 格式错误: {text}")
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
            try:
                lab = parse_lab(",".join(row[2:]))
            except Exception:
                continue
            out[code] = {"code": code, "name": name, "lab": lab}
    if not out:
        raise RuntimeError(f"没有从标准 CSV 读到数据: {path}")
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
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


def build_feature(row: dict[str, Any], ref_bg: np.ndarray) -> np.ndarray:
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
        chroma / 128.0, math.sin(hue), math.cos(hue),
        is_gray, is_red_orange, is_yellow, is_light, is_dark,
        (base_L / 100.0) * (chroma / 128.0),
        (base_a / 128.0) * (base_b / 128.0),
    ], dtype=np.float64)


def collect(
    run_dirs: list[Path],
    standards: dict[str, dict[str, Any]],
    baseline_file: str,
    eval_count: int,
    ref_bg: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    records, X, Y = [], [], []
    for rd in run_dirs:
        csv_path = rd / baseline_file
        if not csv_path.exists():
            alt = rd / "target_results_bg0p25_res0p00_family.csv"
            if alt.exists():
                csv_path = alt
            else:
                print(f"[skip] 找不到 {csv_path}")
                continue

        rows = read_rows(csv_path)
        if eval_count > 0:
            rows = rows[:eval_count]

        n = 0
        for row in rows:
            code = row.get("code", "").strip().upper()
            if code not in standards:
                continue

            std_lab = standards[code]["lab"]
            base_lab = np.array([
                fval(row, "final_L", fval(row, "root_L")),
                fval(row, "final_a", fval(row, "root_a")),
                fval(row, "final_b", fval(row, "root_b")),
            ], dtype=np.float64)

            if not np.all(np.isfinite(base_lab)):
                continue

            rec = dict(row)
            rec["_run"] = rd.name
            rec["_run_dir"] = str(rd)
            rec["_std_L"] = float(std_lab[0])
            rec["_std_a"] = float(std_lab[1])
            rec["_std_b"] = float(std_lab[2])
            rec["_base_L"] = float(base_lab[0])
            rec["_base_a"] = float(base_lab[1])
            rec["_base_b"] = float(base_lab[2])

            X.append(build_feature(row, ref_bg))
            Y.append(std_lab - base_lab)
            records.append(rec)
            n += 1

        print(f"[ok] {rd}: {n} samples" if n else f"[skip] {rd}: 0 samples")

    if not X:
        raise RuntimeError("没有收集到训练样本")
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


def nearest(lab: np.ndarray, standards: dict[str, dict[str, Any]]) -> tuple[str, str, float]:
    codes = list(standards.keys())
    labs = np.vstack([standards[c]["lab"] for c in codes])
    de = delta_e(lab[None, :], labs)
    i = int(np.argmin(de))
    c = codes[i]
    return c, standards[c]["name"], float(de[i])


def evaluate(records: list[dict[str, Any]], pred_shift: np.ndarray, standards: dict[str, dict[str, Any]], cap: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_shift = np.clip(pred_shift, -cap, cap)
    out_rows = []
    before_list, base_list, model_list = [], [], []
    corrects, harm_before, harm_base = [], [], []

    for rec, sh in zip(records, pred_shift):
        code = rec["code"].strip().upper()
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

        pc, pn, pde = nearest(model_lab, standards)
        ok = pc == code

        before_list.append(before_de)
        base_list.append(base_de)
        model_list.append(model_de)
        corrects.append(ok)
        harm_before.append(model_de > before_de)
        harm_base.append(model_de > base_de)

        out_rows.append({
            "run": rec["_run"],
            "idx": rec.get("idx", ""),
            "code": code,
            "name": rec.get("name", standards[code]["name"]),
            "before_deltaE": before_de,
            "base_deltaE": base_de,
            "model_deltaE": model_de,
            "model_improve_vs_base": base_de - model_de,
            "shift_L": float(sh[0]),
            "shift_a": float(sh[1]),
            "shift_b": float(sh[2]),
            "model_L": float(model_lab[0]),
            "model_a": float(model_lab[1]),
            "model_b": float(model_lab[2]),
            "pred_code": pc,
            "pred_name": pn,
            "pred_deltaE": pde,
            "correct": ok,
            "harm_vs_before": model_de > before_de,
            "harm_vs_base": model_de > base_de,
        })

    sb, ss, sm = stats(before_list), stats(base_list), stats(model_list)
    summary = {
        "count": len(records),
        "before_mean_deltaE": sb["mean"],
        "base_mean_deltaE": ss["mean"],
        "model_mean_deltaE": sm["mean"],
        "before_median_deltaE": sb["median"],
        "base_median_deltaE": ss["median"],
        "model_median_deltaE": sm["median"],
        "before_max_deltaE": sb["max"],
        "base_max_deltaE": ss["max"],
        "model_max_deltaE": sm["max"],
        "model_p95_deltaE": sm["p95"],
        "classification_acc": float(np.mean(corrects)),
        "harm_vs_before_rate": float(np.mean(harm_before)),
        "harm_vs_base_rate": float(np.mean(harm_base)),
    }
    return summary, out_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="单个胶块适用的历史 residual 模型训练。不使用行列/位置/code作为特征。")
    ap.add_argument("--runs-glob", default="dataset_runs/run_*")
    ap.add_argument("--runs", default="")
    ap.add_argument("--standards-csv", required=True)
    ap.add_argument("--baseline-file", default="best_target_results.csv")
    ap.add_argument("--eval-count", type=int, default=128)
    ap.add_argument("--background-lab", default="84.71,-1.14,-3.64")
    ap.add_argument("--alpha-list", default="0.1,1,3,10,30,100,300,1000")
    ap.add_argument("--cap-L", type=float, default=12.0)
    ap.add_argument("--cap-a", type=float, default=28.0)
    ap.add_argument("--cap-b", type=float, default=36.0)
    ap.add_argument("--out", default="single_residual_model_out")
    args = ap.parse_args()

    if args.runs.strip():
        run_dirs = [Path(x.strip()) for x in args.runs.split(",") if x.strip()]
    else:
        run_dirs = [Path(x) for x in sorted(glob.glob(args.runs_glob)) if Path(x).is_dir()]

    standards = read_standards(Path(args.standards_csv))
    ref_bg = parse_lab(args.background_lab)
    alphas = [float(x.strip()) for x in args.alpha_list.replace("，", ",").split(",") if x.strip()]
    cap = np.array([args.cap_L, args.cap_a, args.cap_b], dtype=np.float64)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records, X_all, Y_all = collect(run_dirs, standards, args.baseline_file, args.eval_count, ref_bg)
    valid_runs = sorted(set(Path(r["_run_dir"]) for r in all_records), key=lambda x: str(x))
    if len(valid_runs) < 2:
        raise RuntimeError("有效 run 少于 2 个。")

    sweep_rows = []
    loo_rows_by_alpha = {}

    for alpha in alphas:
        rec_all = []
        pred_all = []

        for test_run in valid_runs:
            train_runs = [r for r in valid_runs if r != test_run]
            train_rec, X_train, Y_train = collect(train_runs, standards, args.baseline_file, args.eval_count, ref_bg)
            test_rec, X_test, Y_test = collect([test_run], standards, args.baseline_file, args.eval_count, ref_bg)

            Xs_train, mean, std = standardize_train(X_train)
            Xs_test = standardize_apply(X_test, mean, std)
            W = ridge_fit(Xs_train, Y_train, alpha)
            pred = Xs_test @ W

            rec_all.extend(test_rec)
            pred_all.append(pred)

        pred_all_np = np.vstack(pred_all)
        summary, rows = evaluate(rec_all, pred_all_np, standards, cap)
        summary["alpha"] = alpha
        sweep_rows.append(summary)
        loo_rows_by_alpha[alpha] = rows

        print(f"alpha={alpha:g} base_mean={summary['base_mean_deltaE']:.4f} model_mean={summary['model_mean_deltaE']:.4f} acc={summary['classification_acc']:.4f}")

    best = sorted(sweep_rows, key=lambda r: (r["model_mean_deltaE"], r["model_max_deltaE"]))[0]
    best_alpha = float(best["alpha"])

    write_csv(out_dir / "alpha_sweep_summary.csv", sweep_rows)
    write_csv(out_dir / "loo_predictions_best_alpha.csv", loo_rows_by_alpha[best_alpha])

    Xs_all, mean_all, std_all = standardize_train(X_all)
    W_final = ridge_fit(Xs_all, Y_all, best_alpha)
    pred_train = Xs_all @ W_final
    train_summary, train_rows = evaluate(all_records, pred_train, standards, cap)
    write_csv(out_dir / "train_predictions_full_model.csv", train_rows)

    model = {
        "type": "single_ridge_lab_residual_v1",
        "note": "单个未知胶块识别适用：不使用 idx/row/col/code 作为特征。训练时用历史标准色产生标签，部署时不需要知道当前胶块编号。",
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
    save_json(out_dir / "single_residual_model.json", model)

    print("\n=== best LOO summary ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("\n=== full train summary, optimistic ===")
    print(json.dumps(train_summary, ensure_ascii=False, indent=2))
    print("\nSaved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
