from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)


def parse_lab(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in str(text).strip().strip('"').replace("，", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Lab格式错误: {text}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def read_standards(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            code = row[0].strip().upper()
            if not code or code.lower() in {"code", "编号"}:
                continue
            try:
                lab = parse_lab(",".join(row[2:]))
            except Exception:
                continue
            out[code] = {
                "code": code,
                "name": row[1].strip(),
                "L": lab[0],
                "a": lab[1],
                "b": lab[2],
            }
    if not out:
        raise RuntimeError(f"标准CSV没有读到数据: {path}")
    return out


def to_float(v: Any, default: float = math.nan) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def delta_e_ciede2000(lab1, lab2, kL=1, kC=1, kH=1) -> float:
    """
    CIEDE2000 ΔE.
    lab1/lab2: (L, a, b)
    """
    L1, a1, b1 = [float(x) for x in lab1]
    L2, a2, b2 = [float(x) for x in lab2]

    avg_L = (L1 + L2) / 2.0
    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1.0 - math.sqrt((avg_C ** 7) / ((avg_C ** 7) + (25.0 ** 7)))) if avg_C != 0 else 0.0

    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)

    def hp_fun(ap, b):
        if ap == 0 and b == 0:
            return 0.0
        h = math.degrees(math.atan2(b, ap))
        return h + 360.0 if h < 0 else h

    h1p = hp_fun(a1p, b1)
    h2p = hp_fun(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        dhp = dh

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    avg_Lp = (L1 + L2) / 2.0
    avg_Cp = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    else:
        dh_abs = abs(h1p - h2p)
        if dh_abs <= 180:
            avg_hp = (h1p + h2p) / 2.0
        elif (h1p + h2p) < 360:
            avg_hp = (h1p + h2p + 360.0) / 2.0
        else:
            avg_hp = (h1p + h2p - 360.0) / 2.0

    T = (
        1
        - 0.17 * math.cos(math.radians(avg_hp - 30))
        + 0.24 * math.cos(math.radians(2 * avg_hp))
        + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
        - 0.20 * math.cos(math.radians(4 * avg_hp - 63))
    )

    delta_theta = 30.0 * math.exp(-(((avg_hp - 275.0) / 25.0) ** 2))
    Rc = 2.0 * math.sqrt((avg_Cp ** 7) / ((avg_Cp ** 7) + (25.0 ** 7))) if avg_Cp != 0 else 0.0
    Sl = 1.0 + ((0.015 * ((avg_Lp - 50.0) ** 2)) / math.sqrt(20.0 + ((avg_Lp - 50.0) ** 2)))
    Sc = 1.0 + 0.045 * avg_Cp
    Sh = 1.0 + 0.015 * avg_Cp * T
    Rt = -math.sin(math.radians(2.0 * delta_theta)) * Rc

    dE = math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return float(dE)


def get_measured_lab(row: dict[str, str], lab_source: str) -> tuple[float, float, float] | None:
    prefix_map = {
        "final": "final",
        "root": "root",
        "raw": "raw",
    }
    prefix = prefix_map[lab_source]
    L = to_float(row.get(f"{prefix}_L"))
    a = to_float(row.get(f"{prefix}_a"))
    b = to_float(row.get(f"{prefix}_b"))
    if math.isnan(L) or math.isnan(a) or math.isnan(b):
        return None
    return (L, a, b)


def main() -> None:
    ap = argparse.ArgumentParser(description="给 ConvNeXt 分类预测结果补充 ΔE2000。")
    ap.add_argument("--pred-csv", required=True, help="例如 convnext_small_val012_out/val_predictions_best.csv")
    ap.add_argument("--labels-csv", required=True, help="例如 color_cls_dataset_val012/labels.csv")
    ap.add_argument("--standards-csv", required=True, help="例如 data.csv")
    ap.add_argument("--out", default="", help="输出CSV，默认 pred同目录/predictions_with_deltaE.csv")
    ap.add_argument("--lab-source", choices=["final", "root", "raw"], default="final",
                    help="用哪一套实测Lab来算ΔE。推荐 final；没有final时可用 root/raw。")
    ap.add_argument("--tau", type=float, default=5.0, help="融合分数里把ΔE转相似度的温度参数。")
    ap.add_argument("--prob-weight", type=float, default=0.7, help="融合分数中分类概率权重。")
    ap.add_argument("--de-weight", type=float, default=0.3, help="融合分数中ΔE相似度权重。")
    args = ap.parse_args()

    pred_path = Path(args.pred_csv)
    labels_path = Path(args.labels_csv)
    standards_path = Path(args.standards_csv)

    out_path = Path(args.out) if args.out else pred_path.parent / "predictions_with_deltaE.csv"

    preds = read_csv_dicts(pred_path)
    labels = read_csv_dicts(labels_path)
    standards = read_standards(standards_path)

    # labels.csv 按 image_path 匹配最稳；退化时用 run+code+source
    label_by_image = {r.get("image_path", ""): r for r in labels}
    label_by_run_code = {(r.get("run", ""), r.get("code", "")): r for r in labels}

    out_rows = []
    n_has_lab = 0
    n_fused_correct = 0
    n_top1_correct = 0

    for r in preds:
        row = dict(r)
        lab_row = label_by_image.get(row.get("image_path", ""))
        if lab_row is None:
            lab_row = label_by_run_code.get((row.get("run", ""), row.get("code", "")))

        measured_lab = get_measured_lab(lab_row or {}, args.lab_source) if lab_row else None

        # 把当前样本自己的测量Lab也写进去
        if measured_lab is not None:
            n_has_lab += 1
            row[f"{args.lab_source}_L_used"] = measured_lab[0]
            row[f"{args.lab_source}_a_used"] = measured_lab[1]
            row[f"{args.lab_source}_b_used"] = measured_lab[2]

            true_code = row.get("code", "").strip().upper()
            if true_code in standards:
                true_std = standards[true_code]
                row["true_deltaE"] = delta_e_ciede2000(
                    measured_lab, (true_std["L"], true_std["a"], true_std["b"])
                )

            # 给分类模型 top1~top5 加 ΔE
            fused_candidates = []
            for k in range(1, 6):
                code = row.get(f"top{k}_code", "").strip().upper()
                prob = to_float(row.get(f"top{k}_prob"), 0.0)
                if code in standards:
                    std = standards[code]
                    de = delta_e_ciede2000(measured_lab, (std["L"], std["a"], std["b"]))
                    sim = math.exp(-de / max(args.tau, 1e-6))
                    fused_score = args.prob_weight * prob + args.de_weight * sim

                    row[f"top{k}_deltaE_{args.lab_source}"] = de
                    row[f"top{k}_de_sim"] = sim
                    row[f"top{k}_fused_score"] = fused_score

                    fused_candidates.append({
                        "rank": k,
                        "label": row.get(f"top{k}_label", ""),
                        "code": code,
                        "name": row.get(f"top{k}_name", ""),
                        "prob": prob,
                        "deltaE": de,
                        "score": fused_score,
                    })

            fused_candidates.sort(key=lambda x: x["score"], reverse=True)
            if fused_candidates:
                best = fused_candidates[0]
                row["fused_top1_code"] = best["code"]
                row["fused_top1_name"] = best["name"]
                row["fused_top1_prob"] = best["prob"]
                row["fused_top1_deltaE"] = best["deltaE"]
                row["fused_top1_score"] = best["score"]
                row["fused_top1_correct"] = int(best["code"] == row.get("code", "").strip().upper())
                n_fused_correct += int(row["fused_top1_correct"])

        row["top1_correct"] = int(str(row.get("top1_correct", "0")) in {"1", "True", "true"})
        n_top1_correct += int(row["top1_correct"])
        out_rows.append(row)

    write_csv(out_path, out_rows)

    summary = {
        "pred_csv": str(pred_path),
        "labels_csv": str(labels_path),
        "standards_csv": str(standards_path),
        "lab_source": args.lab_source,
        "n_rows": len(out_rows),
        "n_has_lab": n_has_lab,
        "original_top1_acc": n_top1_correct / max(len(out_rows), 1),
        "fused_top1_acc": n_fused_correct / max(n_has_lab, 1) if n_has_lab else None,
        "tau": args.tau,
        "prob_weight": args.prob_weight,
        "de_weight": args.de_weight,
        "out_csv": str(out_path),
    }
    (out_path.parent / "deltaE_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Done ===")
    print("rows:", len(out_rows))
    print("has measured Lab:", n_has_lab)
    print("original top1 acc:", summary["original_top1_acc"])
    print("fused top1 acc:", summary["fused_top1_acc"])
    print("out:", out_path)
    print("summary:", out_path.parent / "deltaE_summary.json")


if __name__ == "__main__":
    main()
