from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """
    Windows + 中文路径下，cv2.imread 可能因为路径编码失败。
    用 np.fromfile + cv2.imdecode 可以绕过这个问题。
    """
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path: str | Path, img: np.ndarray, params: list[int] | None = None) -> bool:
    """
    Windows + 中文路径下，cv2.imwrite 也可能失败。
    用 cv2.imencode + tofile 保存。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix
    if not ext:
        ext = ".jpg"
    params = params or []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(str(path))
    return True



def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def parse_lab(text: str) -> list[float]:
    parts = [p.strip() for p in str(text).strip().strip('"').replace("，", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Lab格式错误: {text}")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


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
            out[code] = {"name": row[1].strip(), "std_L": lab[0], "std_a": lab[1], "std_b": lab[2]}
    if not out:
        raise RuntimeError(f"没有从标准CSV读到数据: {path}")
    return out


def find_file(path_str: str, project_root: Path, run_dir: Path) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    candidates = [p] if p.is_absolute() else [project_root / p, run_dir / p, run_dir.parent / p, Path.cwd() / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def load_rois(path: Path) -> list[dict[str, Any]]:
    obj = read_json(path)
    if isinstance(obj, dict):
        obj = obj.get("rois") or obj.get("items")
    if not isinstance(obj, list):
        raise RuntimeError(f"无法识别ROI格式: {path}")

    out = []
    for i, item in enumerate(obj, start=1):
        if isinstance(item, dict):
            idx = int(item.get("idx", i))
            code = str(item.get("code", f"W{i:03d}")).strip().upper()
            roi = item.get("roi")
        else:
            idx = i
            code = f"W{i:03d}"
            roi = item
        if roi is None or len(roi) != 4:
            continue
        out.append({"idx": idx, "code": code, "roi": [int(round(float(x))) for x in roi]})
    return out


def clip_roi(roi: list[int], w: int, h: int) -> list[int] | None:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def crop_square_resize(img: np.ndarray, roi: list[int], image_size: int, pad_ratio: float) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    rw, rh = x2 - x1, y2 - y1
    pad = int(round(max(rw, rh) * pad_ratio))
    roi2 = clip_roi([x1 - pad, y1 - pad, x2 + pad, y2 + pad], w, h)
    if roi2 is None:
        raise RuntimeError(f"ROI无效: {roi}")
    x1, y1, x2, y2 = roi2
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    top = (side - ch) // 2
    bottom = side - ch - top
    left = (side - cw) // 2
    right = side - cw - left
    crop = cv2.copyMakeBorder(crop, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)


def crop_stats(bgr: np.ndarray) -> dict[str, float]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    row = {}
    for i, name in enumerate(["R", "G", "B"]):
        ch = rgb[:, :, i].reshape(-1)
        row[f"crop_{name}_mean"] = float(np.mean(ch))
        row[f"crop_{name}_std"] = float(np.std(ch))
    for i, name in enumerate(["H", "S", "V"]):
        ch = hsv[:, :, i].reshape(-1)
        row[f"crop_{name}_mean"] = float(np.mean(ch))
        row[f"crop_{name}_std"] = float(np.std(ch))
    return row


def split_by_run(runs: list[str], val_run: str = "", test_run: str = "") -> dict[str, str]:
    uniq = sorted(set(runs))
    m = {r: "train" for r in uniq}
    if test_run and test_run in m:
        m[test_run] = "test"
    if val_run and val_run in m:
        m[val_run] = "val"
    else:
        available = [r for r in uniq if m[r] == "train"]
        if available:
            m[available[-1]] = "val"
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="从 dataset_runs/run_xxx 裁剪 ROI，构建 ConvNeXt 训练数据集。")
    ap.add_argument("--runs-glob", default="dataset_runs/run_*")
    ap.add_argument("--runs", default="")
    ap.add_argument("--standards-csv", required=True)
    ap.add_argument("--out", default="color_cls_dataset")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--pad-ratio", type=float, default=0.08)
    ap.add_argument("--eval-count", type=int, default=128)
    ap.add_argument("--crop-source", choices=["original", "root", "both"], default="original")
    ap.add_argument("--val-run", default="")
    ap.add_argument("--test-run", default="")
    ap.add_argument("--skip-runs", default="")
    args = ap.parse_args()

    project_root = Path.cwd()
    out_dir = Path(args.out)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    standards = read_standards(Path(args.standards_csv))
    code_to_label = {f"W{i:03d}": i - 1 for i in range(1, 129)}

    if args.runs.strip():
        run_dirs = [Path(x.strip()) for x in args.runs.split(",") if x.strip()]
    else:
        run_dirs = [Path(x) for x in sorted(glob.glob(args.runs_glob)) if Path(x).is_dir()]

    skip = {x.strip() for x in args.skip_runs.split(",") if x.strip()}
    run_dirs = [r for r in run_dirs if r.name not in skip]

    labels = []
    errors = []

    for run_dir in run_dirs:
        report_path = run_dir / "report.json"
        roi_path = run_dir / "selected_rois.json"
        result_path = run_dir / "best_target_results.csv"

        if not (report_path.exists() and roi_path.exists() and result_path.exists()):
            print(f"[skip] {run_dir.name}: 缺 report/selected_rois/best_target_results")
            errors.append({"run": run_dir.name, "error": "missing required files"})
            continue

        report = read_json(report_path)
        photo_str = report.get("input", {}).get("photo", "")
        photo_path = find_file(photo_str, project_root, run_dir)
        if photo_path is None:
            print(f"[skip] {run_dir.name}: 找不到原图 {photo_str}")
            errors.append({"run": run_dir.name, "error": f"photo not found: {photo_str}"})
            continue

        original_img = imread_unicode(photo_path, cv2.IMREAD_COLOR)
        if original_img is None:
            errors.append({"run": run_dir.name, "error": f"image read failed, possibly bad path or non-image: {photo_path}"})
            continue

        root_img = None
        if args.crop_source in {"root", "both"}:
            root_str = report.get("outputs", {}).get("root_corrected_photo", "")
            root_path = find_file(root_str, project_root, run_dir)
            if root_path is None:
                fallback = run_dir / "02_rootpoly2_corrected_photo.png"
                if fallback.exists():
                    root_path = fallback
            if root_path is not None:
                root_img = imread_unicode(root_path, cv2.IMREAD_COLOR)
            if root_img is None and args.crop_source == "root":
                print(f"[skip] {run_dir.name}: 找不到root校正图")
                errors.append({"run": run_dir.name, "error": "root corrected image missing"})
                continue

        rois = load_rois(roi_path)
        if args.eval_count > 0:
            rois = rois[:args.eval_count]

        result_rows = read_csv_dicts(result_path)
        res_map = {r.get("code", "").strip().upper(): r for r in result_rows}

        n = 0
        for item in rois:
            idx, code, roi = item["idx"], item["code"], item["roi"]
            if code not in code_to_label or code not in standards:
                continue

            sources = []
            if args.crop_source in {"original", "both"}:
                sources.append(("original", original_img))
            if args.crop_source in {"root", "both"} and root_img is not None:
                sources.append(("root", root_img))

            for source_name, img in sources:
                try:
                    crop = crop_square_resize(img, roi, args.image_size, args.pad_ratio)
                except Exception as e:
                    errors.append({"run": run_dir.name, "code": code, "error": str(e)})
                    continue

                fname = f"{run_dir.name}_{code}_{source_name}.jpg"
                rel_path = f"images/{fname}"
                imwrite_unicode(img_dir / fname, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                res = res_map.get(code, {})
                row = {
                    "image_path": rel_path,
                    "run": run_dir.name,
                    "source": source_name,
                    "idx": idx,
                    "code": code,
                    "label_id": code_to_label[code],
                    "name": standards[code]["name"],
                    "std_L": standards[code]["std_L"],
                    "std_a": standards[code]["std_a"],
                    "std_b": standards[code]["std_b"],
                    "roi": json.dumps(roi, ensure_ascii=False),
                    "photo": str(photo_path),
                }

                for key in [
                    "raw_L", "raw_a", "raw_b",
                    "root_L", "root_a", "root_b",
                    "final_L", "final_a", "final_b",
                    "local_bg_L", "local_bg_a", "local_bg_b",
                    "before_deltaE", "root_deltaE", "final_deltaE",
                    "pred_code", "pred_name", "pred_deltaE",
                    "top2_code", "top2_deltaE", "confidence",
                ]:
                    if key in res:
                        row[key] = res[key]

                row.update(crop_stats(crop))
                labels.append(row)
                n += 1

        print(f"[ok] {run_dir.name}: {n} samples")

    if not labels:
        raise RuntimeError("没有生成任何样本。")

    split_map = split_by_run([r["run"] for r in labels], args.val_run, args.test_run)
    for r in labels:
        r["split"] = split_map.get(r["run"], "train")

    write_csv(out_dir / "labels.csv", labels)
    write_csv(out_dir / "build_errors.csv", errors)

    run_rows = []
    for run in sorted(set(r["run"] for r in labels)):
        rr = [x for x in labels if x["run"] == run]
        run_rows.append({
            "run": run,
            "split": split_map.get(run, "train"),
            "n_samples": len(rr),
            "n_original": sum(1 for x in rr if x["source"] == "original"),
            "n_root": sum(1 for x in rr if x["source"] == "root"),
        })
    write_csv(out_dir / "runs_split.csv", run_rows)

    meta = {
        "n_samples": len(labels),
        "n_runs": len(set(r["run"] for r in labels)),
        "n_classes": len(set(r["code"] for r in labels)),
        "image_size": args.image_size,
        "crop_source": args.crop_source,
        "split_map": split_map,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    print("out:", out_dir.resolve())
    print("samples:", len(labels))
    print("runs:", meta["n_runs"])
    print("classes:", meta["n_classes"])
    print("labels:", out_dir / "labels.csv")
    print("split:", out_dir / "runs_split.csv")
    if errors:
        print("errors:", len(errors), "see", out_dir / "build_errors.csv")


if __name__ == "__main__":
    main()
