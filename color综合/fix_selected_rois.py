# 用途：
#   单独重选 main.py 已保存的部分胶块 ROI，而不用重新框选全部 126/128 个胶块。
#   适合某几个 ROI 从一开始框错、顺序错、框到边缘/背景的情况。
#
# 用法示例：
#   python fix_selected_rois.py --photo pic_all2.jpg --out output_qiangying_126 --codes "强鹰109,强鹰111,强鹰114"
#
# 修完后重新运行 main.py，但不要再加 --force-select-rois：
#   python main.py --photo pic_all2.jpg --standard standard_chart.png --data data_QY.csv --out output_qiangying_126 --top-k 10

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path) -> np.ndarray:
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return img


def find_report(out_dir: Path) -> Path:
    p = out_dir / "report.json"
    if not p.exists():
        raise FileNotFoundError(f"没有找到 report.json：{p}")
    return p


def find_rois_file(out_dir: Path) -> Path:
    candidates = [
        out_dir / "rois_128.json",
        out_dir / "rois.json",
        out_dir / "target_rois.json",
        out_dir / "targets_rois.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    roi_jsons = sorted(out_dir.glob("*roi*.json"))
    if roi_jsons:
        return roi_jsons[0]

    return out_dir / "rois_128.json"


def select_roi_scaled(
    image_bgr: np.ndarray,
    title: str,
    max_w: int = 1100,
    max_h: int = 700,
) -> list[int]:
    h, w = image_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    shown = cv2.resize(
        image_bgr,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, shown.shape[1], shown.shape[0])
    cv2.moveWindow(title, 30, 30)

    roi = cv2.selectROI(title, shown, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)

    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        raise RuntimeError("没有选择有效 ROI，已中止。")

    x1 = int(round(x / scale))
    y1 = int(round(y / scale))
    x2 = int(round((x + rw) / scale))
    y2 = int(round((y + rh) / scale))

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    return [x1, y1, x2, y2]


def load_report_targets(report_path: Path) -> list[dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    targets = report.get("target_colors", [])
    if not targets:
        raise RuntimeError("report.json 里没有 target_colors。")
    return targets


def parse_list_arg(s: str) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.replace("，", ",").split(",") if x.strip()]


def find_target_indices(targets: list[dict], codes: list[str], indices: list[str]) -> list[int]:
    result: list[int] = []

    for code in codes:
        matched = []
        for i, t in enumerate(targets):
            std = t.get("standard", {})
            keys = [
                str(t.get("input_label", "")),
                str(std.get("code", "")),
                str(std.get("label", "")),
                str(std.get("name", "")),
            ]
            if code in keys:
                matched.append(i)

        if not matched:
            for i, t in enumerate(targets):
                std = t.get("standard", {})
                keys = [
                    str(t.get("input_label", "")),
                    str(std.get("code", "")),
                    str(std.get("label", "")),
                    str(std.get("name", "")),
                ]
                if any(code in k for k in keys):
                    matched.append(i)

        if not matched:
            raise ValueError(f"找不到指定 code：{code}")

        if len(matched) > 1:
            print(f"[警告] {code} 匹配到多个目标，将使用第一个：")
            for j in matched:
                t = targets[j]
                print(f"  report_index={t.get('index')} input_label={t.get('input_label')} name={t.get('standard', {}).get('name')}")

        result.append(matched[0])

    for idx_text in indices:
        idx = int(idx_text)
        matched = []
        for i, t in enumerate(targets):
            if int(t.get("index", -999999)) == idx:
                matched.append(i)
        if not matched:
            if 1 <= idx <= len(targets):
                matched.append(idx - 1)
        if not matched:
            raise ValueError(f"找不到指定 index：{idx}")
        result.append(matched[0])

    seen = set()
    uniq = []
    for i in result:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def build_rois_from_report(targets: list[dict]) -> list[list[int]]:
    rois = []
    for t in targets:
        xyxy = t.get("roi_xyxy")
        if not xyxy or len(xyxy) != 4:
            raise RuntimeError("没有 ROI 文件，且 report.json 里也无法完整重建 roi_xyxy。")
        rois.append([int(round(float(v))) for v in xyxy])
    return rois


def update_rois_object(rois_obj, targets: list[dict], target_pos: int, new_xyxy: list[int]):
    t = targets[target_pos]
    code = str(t.get("standard", {}).get("code", t.get("input_label", "")))
    input_label = str(t.get("input_label", code))

    if isinstance(rois_obj, list):
        if target_pos >= len(rois_obj):
            raise IndexError(f"ROI 列表长度不够：len={len(rois_obj)}, target_pos={target_pos}")

        item = rois_obj[target_pos]
        if isinstance(item, list):
            rois_obj[target_pos] = new_xyxy
        elif isinstance(item, dict):
            if "roi_xyxy" in item:
                item["roi_xyxy"] = new_xyxy
            elif "xyxy" in item:
                item["xyxy"] = new_xyxy
            elif "roi" in item:
                item["roi"] = new_xyxy
            else:
                item["roi_xyxy"] = new_xyxy
            item.setdefault("code", code)
            item.setdefault("input_label", input_label)
        else:
            rois_obj[target_pos] = new_xyxy
        return rois_obj

    if isinstance(rois_obj, dict):
        for key in ["rois", "target_rois", "targets"]:
            if key in rois_obj and isinstance(rois_obj[key], list):
                rois_obj[key] = update_rois_object(rois_obj[key], targets, target_pos, new_xyxy)
                return rois_obj

        possible_keys = [code, input_label, str(t.get("index", ""))]
        for k in possible_keys:
            if k in rois_obj:
                rois_obj[k] = new_xyxy
                return rois_obj

        rois_obj[code] = new_xyxy
        return rois_obj

    raise TypeError("无法识别 ROI JSON 格式。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", required=True, help="原始胶块照片，例如 pic_all2.jpg")
    parser.add_argument("--out", required=True, help="main.py 输出目录，例如 output_qiangying_126")
    parser.add_argument("--codes", default="", help='要重选的 code，逗号分隔，例如 "强鹰109,强鹰111,强鹰114"')
    parser.add_argument("--indices", default="", help='要重选的 report index，逗号分隔，例如 "108,110,113"')
    parser.add_argument("--max-w", type=int, default=1100, help="交互窗口最大宽度")
    parser.add_argument("--max-h", type=int, default=700, help="交互窗口最大高度")
    args = parser.parse_args()

    out_dir = Path(args.out)
    report_path = find_report(out_dir)
    rois_path = find_rois_file(out_dir)

    targets = load_report_targets(report_path)

    codes = parse_list_arg(args.codes)
    indices = parse_list_arg(args.indices)
    if not codes and not indices:
        raise RuntimeError('请指定 --codes 或 --indices，例如 --codes "强鹰109,强鹰111,强鹰114"')

    target_positions = find_target_indices(targets, codes, indices)

    image = imread_unicode(args.photo)

    if rois_path.exists():
        rois_obj = json.loads(rois_path.read_text(encoding="utf-8"))
    else:
        print(f"[提示] 未找到 ROI 文件，将从 report.json 重建：{rois_path}")
        rois_obj = build_rois_from_report(targets)

    backup_path = rois_path.with_suffix(rois_path.suffix + ".bak")
    if rois_path.exists() and not backup_path.exists():
        backup_path.write_text(rois_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[备份] 已保存原 ROI 文件：{backup_path}")

    for pos in target_positions:
        t = targets[pos]
        std = t.get("standard", {})
        code = std.get("code", t.get("input_label"))
        name = std.get("name", "")
        old = t.get("roi_xyxy")

        print()
        print(f"准备重选：report_index={t.get('index')} / pos={pos + 1} / {code} {name}")
        print(f"原 ROI: {old}")
        print("在弹窗里框选新的 ROI，然后按 Enter 或 Space 确认；按 Esc 会中止。")

        title = f"重新选择 ROI - {code} {name}"
        new_xyxy = select_roi_scaled(image, title=title, max_w=args.max_w, max_h=args.max_h)

        print(f"新 ROI: {new_xyxy}")
        rois_obj = update_rois_object(rois_obj, targets, pos, new_xyxy)

    rois_path.write_text(
        json.dumps(rois_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"[完成] 已更新 ROI 文件：{rois_path}")
    print("下一步请重新运行 main.py，但不要加 --force-select-rois。")


if __name__ == "__main__":
    main()
