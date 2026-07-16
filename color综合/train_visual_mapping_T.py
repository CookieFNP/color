# -*- coding: utf-8 -*-
"""
用途：
    训练板材端使用的视觉映射 T。

    胶块端 v0.7 已经生成 glue_visual_library.csv，其中每个已知胶块都有：
        corrected_lab       = ColorChecker 基础校正后测得的 Lab
        visual_display_lab  = v0.7 最终视觉图中实际呈现的 Lab

    本脚本使用 128 胶块视觉库训练一个通用映射：
        T: corrected_lab -> visual_display_lab

    之后板材端不能使用 standard_lab - corrected_lab 这种“已知答案 residual”，
    而应使用本脚本训练出的 T，将未知板材的 board_corrected_lab 映射为：
        board_visual_lab = T(board_corrected_lab)

    最终匹配：
        ΔE(board_visual_lab, glue_visual_library.visual_display_lab)

典型运行：
    python train_visual_mapping_T.py --library output_128/glue_visual_library/glue_visual_library.csv --out output_128/visual_mapping_T

推荐默认模型：
    L_vis = L_corr
    a_vis = p0 + p1*L_corr + p2*a_corr + p3*b_corr
    b_vis = q0 + q1*L_corr + q2*a_corr + q3*b_corr
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.color_math import delta_e_2000


REQUIRED_COLUMNS = [
    "corrected_L",
    "corrected_a",
    "corrected_b",
    "visual_display_L",
    "visual_display_a",
    "visual_display_b",
]


def parse_code_list(text: str | None) -> set[str]:
    if not text:
        return set()

    out = set()
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if part:
            out.add(part.upper())
    return out


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default

    text = str(value).strip()
    if text == "":
        return default

    try:
        return float(text)
    except Exception:
        return default


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到胶块视觉库 CSV：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"CSV 为空：{path}")

    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise RuntimeError(
            "glue_visual_library.csv 缺少必要字段："
            + ", ".join(missing)
            + "\n请确认先运行 build_glue_visual_library.py。"
        )

    return rows


def filter_rows(rows: list[dict], exclude_codes: set[str]) -> list[dict]:
    if not exclude_codes:
        return rows

    kept = []

    for row in rows:
        code = str(row.get("code") or "").strip().upper()
        if code in exclude_codes:
            continue
        kept.append(row)

    if not kept:
        raise RuntimeError("过滤后没有可训练样本，请检查 --exclude-codes。")

    return kept


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    X = corrected_lab
    Y = visual_display_lab
    """
    xs = []
    ys = []
    meta = []

    for row in rows:
        corrected = [
            to_float(row.get("corrected_L")),
            to_float(row.get("corrected_a")),
            to_float(row.get("corrected_b")),
        ]

        visual = [
            to_float(row.get("visual_display_L")),
            to_float(row.get("visual_display_a")),
            to_float(row.get("visual_display_b")),
        ]

        if any(v is None for v in corrected + visual):
            continue

        xs.append(corrected)
        ys.append(visual)
        meta.append(
            {
                "index": row.get("index"),
                "code": row.get("code"),
                "name": row.get("name"),
                "visual_crop_path": row.get("visual_crop_path"),
            }
        )

    if len(xs) < 4:
        raise RuntimeError("有效训练样本少于 4 个，无法训练线性模型。")

    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        meta,
    )


def build_features(x_lab: np.ndarray, feature_mode: str) -> tuple[np.ndarray, list[str]]:
    """
    输入：
        x_lab: shape = (N, 3), columns = [L, a, b]

    输出：
        Phi: shape = (N, D)
    """
    x = np.asarray(x_lab, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, 3)

    L = x[:, 0]
    a = x[:, 1]
    b = x[:, 2]

    if feature_mode == "linear":
        phi = np.stack(
            [
                np.ones_like(L),
                L,
                a,
                b,
            ],
            axis=1,
        )
        names = ["1", "L_corr", "a_corr", "b_corr"]

    elif feature_mode == "poly2":
        phi = np.stack(
            [
                np.ones_like(L),
                L,
                a,
                b,
                L * L,
                a * a,
                b * b,
                L * a,
                L * b,
                a * b,
            ],
            axis=1,
        )
        names = [
            "1",
            "L_corr",
            "a_corr",
            "b_corr",
            "L_corr^2",
            "a_corr^2",
            "b_corr^2",
            "L_corr*a_corr",
            "L_corr*b_corr",
            "a_corr*b_corr",
        ]

    else:
        raise ValueError(f"未知 feature_mode：{feature_mode}")

    return phi, names


def fit_ridge(phi: np.ndarray, y: np.ndarray, ridge_alpha: float) -> np.ndarray:
    """
    岭回归：
        w = (X^T X + λI)^-1 X^T y

    约定：
        第 0 列为截距，不正则化。
    """
    phi = np.asarray(phi, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    d = phi.shape[1]

    reg = np.eye(d, dtype=np.float64) * float(ridge_alpha)
    reg[0, 0] = 0.0

    a = phi.T @ phi + reg
    b = phi.T @ y

    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a) @ b

    return w


def predict_with_model(
    x_lab: np.ndarray,
    model: dict,
) -> np.ndarray:
    """
    使用 visual_mapping_T.json 中的模型预测 visual_lab。
    """
    x = np.asarray(x_lab, dtype=np.float64)
    one_dim = x.ndim == 1

    if one_dim:
        x = x.reshape(1, 3)

    feature_mode = model["feature_mode"]
    l_mode = model["L_mode"]

    phi, _ = build_features(x, feature_mode)

    y = np.zeros_like(x, dtype=np.float64)

    if l_mode == "identity":
        y[:, 0] = x[:, 0]
    elif l_mode == "linear":
        y[:, 0] = phi @ np.asarray(model["coefficients"]["L"], dtype=np.float64)
    else:
        raise ValueError(f"未知 L_mode：{l_mode}")

    y[:, 1] = phi @ np.asarray(model["coefficients"]["a"], dtype=np.float64)
    y[:, 2] = phi @ np.asarray(model["coefficients"]["b"], dtype=np.float64)

    return y[0] if one_dim else y


def fit_visual_mapping_T(
    x: np.ndarray,
    y: np.ndarray,
    *,
    feature_mode: str,
    l_mode: str,
    ridge_alpha: float,
) -> dict:
    phi, feature_names = build_features(x, feature_mode)

    coeffs: dict[str, list[float] | None] = {
        "L": None,
        "a": None,
        "b": None,
    }

    if l_mode == "identity":
        coeffs["L"] = None
    elif l_mode == "linear":
        coeffs["L"] = fit_ridge(phi, y[:, 0], ridge_alpha).tolist()
    else:
        raise ValueError(f"未知 L_mode：{l_mode}")

    coeffs["a"] = fit_ridge(phi, y[:, 1], ridge_alpha).tolist()
    coeffs["b"] = fit_ridge(phi, y[:, 2], ridge_alpha).tolist()

    model = {
        "version": "visual_mapping_T_ab_linear_v0.1" if feature_mode == "linear" and l_mode == "identity" else "visual_mapping_T_custom",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "corrected_lab_to_visual_display_lab",
        "feature_mode": feature_mode,
        "L_mode": l_mode,
        "ridge_alpha": float(ridge_alpha),
        "input_columns": ["corrected_L", "corrected_a", "corrected_b"],
        "output_columns": ["visual_display_L", "visual_display_a", "visual_display_b"],
        "feature_names": feature_names,
        "coefficients": coeffs,
        "formula": {
            "L": "L_vis = L_corr" if l_mode == "identity" else "L_vis = dot(features, coeff_L)",
            "a": "a_vis = dot(features, coeff_a)",
            "b": "b_vis = dot(features, coeff_b)",
        },
        "note": (
            "该 T 用于把未知板材的 ColorChecker corrected_lab 映射到胶块 v0.7 定义的 visual Lab 域。"
            "板材端不能使用 standard_lab - corrected_lab，因为未知板材没有标准答案。"
        ),
    }

    return model


def de2000_array(a_lab: np.ndarray, b_lab: np.ndarray) -> np.ndarray:
    return delta_e_2000(
        np.asarray(a_lab, dtype=np.float64),
        np.asarray(b_lab, dtype=np.float64),
    )


def stat_pack(values: np.ndarray | list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)

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


def channel_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    err = pred - target

    out = {}

    for i, name in enumerate(["L", "a", "b"]):
        e = err[:, i]
        out[name] = {
            "mean_error": float(np.mean(e)),
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e * e))),
            "max_abs": float(np.max(np.abs(e))),
        }

    return out


def evaluate_predictions(pred: np.ndarray, target: np.ndarray) -> dict:
    de = de2000_array(pred, target)

    return {
        "deltaE2000": stat_pack(de),
        "channels": channel_metrics(pred, target),
    }


def make_prediction_rows(
    meta: list[dict],
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
) -> list[dict]:
    de = de2000_array(y_pred, y_true)

    rows = []

    for i, m in enumerate(meta):
        rows.append(
            {
                "index": m.get("index"),
                "code": m.get("code"),
                "name": m.get("name"),
                "visual_crop_path": m.get("visual_crop_path"),

                "corrected_L": float(x[i, 0]),
                "corrected_a": float(x[i, 1]),
                "corrected_b": float(x[i, 2]),

                "target_visual_L": float(y_true[i, 0]),
                "target_visual_a": float(y_true[i, 1]),
                "target_visual_b": float(y_true[i, 2]),

                f"{prefix}_pred_L": float(y_pred[i, 0]),
                f"{prefix}_pred_a": float(y_pred[i, 1]),
                f"{prefix}_pred_b": float(y_pred[i, 2]),

                f"{prefix}_err_L": float(y_pred[i, 0] - y_true[i, 0]),
                f"{prefix}_err_a": float(y_pred[i, 1] - y_true[i, 1]),
                f"{prefix}_err_b": float(y_pred[i, 2] - y_true[i, 2]),
                f"{prefix}_deltaE2000": float(de[i]),
            }
        )

    return rows


def leave_one_out_validation(
    x: np.ndarray,
    y: np.ndarray,
    meta: list[dict],
    *,
    feature_mode: str,
    l_mode: str,
    ridge_alpha: float,
) -> tuple[list[dict], dict]:
    preds = []

    n = len(x)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        model_i = fit_visual_mapping_T(
            x[train_mask],
            y[train_mask],
            feature_mode=feature_mode,
            l_mode=l_mode,
            ridge_alpha=ridge_alpha,
        )

        pred_i = predict_with_model(x[i], model_i)
        preds.append(pred_i)

    y_pred = np.asarray(preds, dtype=np.float64)

    rows = make_prediction_rows(
        meta=meta,
        x=x,
        y_true=y,
        y_pred=y_pred,
        prefix="loo",
    )

    metrics = evaluate_predictions(y_pred, y)

    return rows, metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = []
    seen = set()

    preferred = [
        "index",
        "code",
        "name",
        "visual_crop_path",
        "corrected_L",
        "corrected_a",
        "corrected_b",
        "target_visual_L",
        "target_visual_a",
        "target_visual_b",
    ]

    for k in preferred:
        if any(k in r for r in rows) and k not in seen:
            keys.append(k)
            seen.add(k)

    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_readme(path: Path, summary: dict) -> None:
    text = f"""# Visual Mapping T

## 用途

本目录保存板材端使用的视觉映射模型 T。

T 的作用是：

```text
corrected_lab -> visual_display_lab
```

它不是 ColorChecker 校正模型，也不是标准值 residual。

胶块端 v0.7 已经建立了 `glue_visual_library.csv`，其中包含：

```text
corrected_lab
visual_display_lab
```

本模型用这些胶块样本训练，使未知板材可以被映射到同一个 visual Lab 域。

## 推荐使用方式

板材端流程：

```text
板材原图 + 色卡
↓
ColorChecker 基础校正
↓
提取板材 ROI corrected_lab
↓
使用 visual_mapping_T.json
↓
board_visual_lab = T(board_corrected_lab)
↓
和 glue_visual_library.csv 中的 visual_display_lab 计算 ΔE TopK
```

## 当前模型

```text
feature_mode = {summary["model"]["feature_mode"]}
L_mode = {summary["model"]["L_mode"]}
ridge_alpha = {summary["model"]["ridge_alpha"]}
n_samples = {summary["n_samples"]}
```

## Leave-one-out 验证

```json
{json.dumps(summary["loo_metrics"], ensure_ascii=False, indent=2)}
```

## 重要说明

该 T 只负责将未知板材从 corrected Lab 域映射到胶块 v0.7 视觉域。

最终匹配应使用：

```text
ΔE(board_visual_lab, glue_visual_display_lab)
```

而不是：

```text
ΔE(board_corrected_lab, glue_standard_lab)
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train visual mapping T: corrected_lab -> visual_display_lab."
    )

    parser.add_argument(
        "--library",
        required=True,
        help="build_glue_visual_library.py 生成的 glue_visual_library.csv",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="输出目录。默认与 library 同级的 visual_mapping_T",
    )

    parser.add_argument(
        "--feature-mode",
        choices=["linear", "poly2"],
        default="linear",
        help="特征形式。默认 linear，即 [1,L,a,b]。",
    )

    parser.add_argument(
        "--l-mode",
        choices=["identity", "linear"],
        default="identity",
        help="L 通道处理方式。默认 identity，即 L_vis=L_corr。",
    )

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-6,
        help="岭回归正则化强度，默认 1e-6。",
    )

    parser.add_argument(
        "--exclude-codes",
        default=None,
        help='排除异常胶块，例如 "W053,W098"。默认不排除。',
    )

    args = parser.parse_args()

    library_path = Path(args.library)

    out_dir = Path(args.out) if args.out else (library_path.parent / "visual_mapping_T")
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_codes = parse_code_list(args.exclude_codes)

    raw_rows = read_csv_rows(library_path)
    train_rows = filter_rows(raw_rows, exclude_codes)

    x, y, meta = rows_to_arrays(train_rows)

    model = fit_visual_mapping_T(
        x=x,
        y=y,
        feature_mode=args.feature_mode,
        l_mode=args.l_mode,
        ridge_alpha=args.ridge_alpha,
    )

    model["trained_from"] = str(library_path)
    model["n_samples"] = int(len(x))
    model["excluded_codes"] = sorted(list(exclude_codes))

    train_pred = predict_with_model(x, model)
    train_metrics = evaluate_predictions(train_pred, y)
    train_rows_out = make_prediction_rows(
        meta=meta,
        x=x,
        y_true=y,
        y_pred=train_pred,
        prefix="train",
    )

    loo_rows, loo_metrics = leave_one_out_validation(
        x=x,
        y=y,
        meta=meta,
        feature_mode=args.feature_mode,
        l_mode=args.l_mode,
        ridge_alpha=args.ridge_alpha,
    )

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "library": str(library_path),
        "out_dir": str(out_dir),
        "n_samples": int(len(x)),
        "excluded_codes": sorted(list(exclude_codes)),
        "model": model,
        "train_metrics": train_metrics,
        "loo_metrics": loo_metrics,
        "interpretation": {
            "train_metrics": "全量训练后在训练集上的拟合误差，仅作参考。",
            "loo_metrics": "Leave-one-out 交叉验证，更能反映 T 对未见颜色的泛化能力。",
            "recommended_match": "board_visual_lab vs glue_visual_library.visual_display_lab",
        },
    }

    model_path = out_dir / "visual_mapping_T.json"
    summary_path = out_dir / "training_summary.json"
    train_csv_path = out_dir / "train_fit_predictions.csv"
    loo_csv_path = out_dir / "leave_one_out_predictions.csv"
    readme_path = out_dir / "README.md"

    write_json(model_path, model)
    write_json(summary_path, summary)
    write_csv(train_csv_path, train_rows_out)
    write_csv(loo_csv_path, loo_rows)
    write_readme(readme_path, summary)

    print("\n==== visual_mapping_T 训练完成 ====")
    print("输出目录：", out_dir)
    print("模型：", model_path)
    print("训练摘要：", summary_path)
    print("训练集预测：", train_csv_path)
    print("LOO 验证：", loo_csv_path)

    print("\n训练集 ΔE2000：")
    for k, v in train_metrics["deltaE2000"].items():
        print(f"  {k}: {v}")

    print("\nLeave-one-out ΔE2000：")
    for k, v in loo_metrics["deltaE2000"].items():
        print(f"  {k}: {v}")

    print("\n建议后续板材匹配使用：")
    print("  board_visual_lab = T(board_corrected_lab)")
    print("  ΔE(board_visual_lab, glue_visual_library.visual_display_lab)")


if __name__ == "__main__":
    main()
