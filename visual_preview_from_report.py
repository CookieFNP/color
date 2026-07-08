from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import cv2
import numpy as np
from skimage import color


# ============================================================
# 1. 基础工具：安全读 JSON、中文路径读写图片
# ============================================================

def read_json_safe(report_path: Path) -> Optional[dict]:
    """
    安全读取 report.json。

    如果文件不存在、为空、损坏、不是合法 JSON，返回 None。
    这样批量处理时不会因为一个坏 report 中断全部。
    """
    if not report_path.exists():
        print(f"跳过：report 不存在：{report_path}")
        return None

    if report_path.stat().st_size == 0:
        print(f"跳过空 report：{report_path}")
        return None

    try:
        text = report_path.read_text(encoding="utf-8-sig")
        if not text.strip():
            print(f"跳过空 report：{report_path}")
            return None
        return json.loads(text)
    except Exception as e:
        print(f"跳过损坏 report：{report_path}")
        print(f"原因：{e}")
        return None


def imread_unicode(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    支持中文路径的图片读取。

    Windows 下 cv2.imread 经常读不了中文路径，
    所以这里用 np.fromfile + cv2.imdecode。
    """
    path = str(path)

    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: Union[str, Path], img: np.ndarray) -> bool:
    """
    支持中文路径的图片保存。

    Windows 下 cv2.imwrite 也可能受中文路径影响，
    所以这里用 cv2.imencode + tofile。
    """
    path = Path(path)
    ext = path.suffix

    if not ext:
        ext = ".png"

    try:
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


# ============================================================
# 2. 路径解析
# ============================================================

def find_file_by_name(root: Path, filename: str) -> Optional[Path]:
    """
    在项目目录下递归查找某个文件名。
    用于解决 report.json 里只记录了 qiangguang.jpg，
    但图片实际在子目录里的情况。
    """
    if not filename:
        return None

    try:
        matches = list(root.rglob(filename))
        if matches:
            return matches[0].resolve()
    except Exception:
        pass

    return None


def resolve_path(path_str: str, report_path: Path, root: Path) -> Path:
    """
    尝试解析 report.json 里记录的图片路径。

    会按以下顺序查找：
    1. 原始路径
    2. 项目根目录 / 原始路径
    3. report 所在目录 / 原始路径
    4. report 上一级目录 / 原始路径
    5. 在项目目录下按文件名递归搜索
    """
    p = Path(path_str)

    candidates = [
        p,
        root / p,
        report_path.parent / p,
        report_path.parent.parent / p,
    ]

    for c in candidates:
        try:
            if c.exists():
                return c.resolve()
        except Exception:
            pass

    found = find_file_by_name(root, p.name)
    if found is not None:
        return found

    raise FileNotFoundError(f"找不到图片文件：{path_str}")


def get_photo_path(
    report: dict,
    report_path: Path,
    root: Path,
    photo_override: Optional[str] = None,
) -> Path:
    """
    获取原图路径。

    优先级：
    1. 命令行 --photo 指定路径
    2. report['input']['photo']
    3. report['photo']
    4. report['photo_path']
    """
    if photo_override:
        p = Path(photo_override)
        if p.exists():
            return p.resolve()

        found = find_file_by_name(root, p.name)
        if found is not None:
            return found

        raise FileNotFoundError(f"--photo 指定的图片不存在：{photo_override}")

    photo_str = None

    if isinstance(report.get("input"), dict):
        photo_str = report.get("input", {}).get("photo")

    if not photo_str:
        photo_str = report.get("photo")

    if not photo_str:
        photo_str = report.get("photo_path")

    if not photo_str:
        raise ValueError(f"{report_path} 里没有 input.photo / photo / photo_path")

    return resolve_path(str(photo_str), report_path, root)


# ============================================================
# 3. Lab 图像转换
# ============================================================

def bgr_to_lab_image(bgr: np.ndarray) -> np.ndarray:
    """
    OpenCV BGR uint8 -> skimage Lab float64。

    L: 0~100
    a,b: 大约 -128~127
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    lab = color.rgb2lab(rgb)
    return lab.astype(np.float64)


def lab_to_bgr_image(lab: np.ndarray) -> np.ndarray:
    """
    skimage Lab float64 -> OpenCV BGR uint8。
    """
    lab = lab.copy()

    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 100)
    lab[:, :, 1] = np.clip(lab[:, :, 1], -128, 127)
    lab[:, :, 2] = np.clip(lab[:, :, 2], -128, 127)

    rgb = color.lab2rgb(lab)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


# ============================================================
# 4. ROI / Mask
# ============================================================

def make_soft_roi_mask(
    shape_hw: tuple[int, int],
    xyxy: Union[List[int], tuple],
    feather: int = 25,
    mode: str = "rectangle",
) -> np.ndarray:
    """
    生成一个 0~1 的软 mask。

    作用：
    - 胶块区域内部修正强；
    - 边缘逐渐过渡；
    - 避免像贴图一样生硬。
    """
    h, w = shape_hw
    x1, y1, x2, y2 = [int(v) for v in xyxy]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=np.float64)

    mask = np.zeros((h, w), dtype=np.float64)

    if mode == "ellipse":
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ax = max(1, (x2 - x1) // 2)
        ay = max(1, (y2 - y1) // 2)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
    else:
        mask[y1:y2, x1:x2] = 1.0

    if feather > 1:
        if feather % 2 == 0:
            feather += 1

        mask = cv2.GaussianBlur(mask, (feather, feather), 0)

        max_val = float(mask.max())
        if max_val > 1e-8:
            mask = mask / max_val

    return np.clip(mask, 0.0, 1.0)


def build_exclude_target_mask(
    shape_hw: tuple[int, int],
    target_colors: List[Dict[str, Any]],
    expand: int = 10,
) -> np.ndarray:
    """
    生成胶块区域 mask。

    背景中性化时要排除胶块，
    否则可能把浅色胶块误认为白色背景。
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    for item in target_colors:
        roi = item.get("roi_xyxy")
        if not roi:
            continue

        x1, y1, x2, y2 = [int(v) for v in roi]

        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(w - 1, x2 + expand)
        y2 = min(h - 1, y2 + expand)

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    return mask.astype(bool)


def build_background_mask(
    bgr: np.ndarray,
    lab: np.ndarray,
    target_colors: List[Dict[str, Any]],
    bg_min_L: float = 50.0,
    bg_max_saturation: float = 80.0,
    exclude_targets: bool = True,
) -> np.ndarray:
    """
    自动找白色/灰白背景区域。

    判断依据：
    - L 较高：说明是亮区或灰白背景；
    - HSV 饱和度较低：说明不是明显彩色物体；
    - 排除胶块 ROI：避免把浅色胶块当背景。
    """
    h, w = bgr.shape[:2]

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float64)
    L = lab[:, :, 0]

    bg_mask = (L >= bg_min_L) & (s <= bg_max_saturation)

    if exclude_targets:
        target_mask = build_exclude_target_mask((h, w), target_colors, expand=12)
        bg_mask[target_mask] = False

    bg_mask_u8 = bg_mask.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    bg_mask_u8 = cv2.morphologyEx(bg_mask_u8, cv2.MORPH_OPEN, kernel)
    bg_mask_u8 = cv2.morphologyEx(bg_mask_u8, cv2.MORPH_CLOSE, kernel)

    bg_soft = bg_mask_u8.astype(np.float64) / 255.0
    bg_soft = cv2.GaussianBlur(bg_soft, (31, 31), 0)

    max_val = float(bg_soft.max())
    if max_val > 1e-8:
        bg_soft = bg_soft / max_val

    return np.clip(bg_soft, 0.0, 1.0)


# ============================================================
# 5. 背景中性化 + 胶块视觉修正
# ============================================================

def apply_background_neutralization(
    lab: np.ndarray,
    bg_mask: np.ndarray,
    strength: float = 0.55,
) -> np.ndarray:
    """
    背景视觉修正。

    保留 L：
        不把阴影区域强行拉成纯白。

    修正 a/b：
        把白色/灰白背景的色偏往 0 拉，
        让偏黄、偏蓝、偏绿的灰白背景变中性。
    """
    out = lab.copy()
    m = bg_mask.astype(np.float64)

    out[:, :, 1] = out[:, :, 1] * (1.0 - strength * m)
    out[:, :, 2] = out[:, :, 2] * (1.0 - strength * m)

    return out


def get_standard_lab(item: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    从 target_colors 的一个 item 里读取标准 Lab。

    兼容几种可能结构：
    - item['standard']['lab']
    - item['standard_lab']
    - item['matched_standard_lab']
    """
    standard = item.get("standard")

    if isinstance(standard, dict) and standard.get("lab") is not None:
        return np.asarray(standard.get("lab"), dtype=np.float64)

    if item.get("standard_lab") is not None:
        return np.asarray(item.get("standard_lab"), dtype=np.float64)

    if item.get("matched_standard_lab") is not None:
        return np.asarray(item.get("matched_standard_lab"), dtype=np.float64)

    return None


def get_before_lab_from_roi(
    lab: np.ndarray,
    roi: Union[List[int], tuple],
) -> Optional[np.ndarray]:
    """
    如果 report 里没有 before_lab，
    就从原图 ROI 里用中位数估计一个 before_lab。
    """
    h, w = lab.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in roi]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    roi_lab = lab[y1:y2, x1:x2]

    if roi_lab.size == 0:
        return None

    return np.median(roi_lab.reshape(-1, 3), axis=0)


def apply_target_visual_correction(
    bgr: np.ndarray,
    lab: np.ndarray,
    target_colors: List[Dict[str, Any]],
    ab_strength: float = 0.50,
    l_strength: float = 0.10,
    feather: int = 25,
    mask_mode: str = "rectangle",
    protect_extreme_light: bool = True,
) -> np.ndarray:
    """
    胶块视觉修正。

    思路：
    - L 通道只轻微修正，保留阴影、明暗、反光；
    - a/b 通道向标准色靠近，让胶块视觉颜色更准；
    - 极亮高光和极暗阴影区域降低修正强度；
    - ROI 边缘羽化，避免贴图感。
    """
    out = lab.copy()
    h, w = lab.shape[:2]

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float64)

    if protect_extreme_light:
        protect = np.ones((h, w), dtype=np.float64)

        protect[v < 25] = 0.25
        protect[(v >= 25) & (v < 45)] = 0.60

        protect[v > 245] = 0.25
        protect[(v > 230) & (v <= 245)] = 0.60
    else:
        protect = np.ones((h, w), dtype=np.float64)

    used_count = 0

    for item in target_colors:
        roi = item.get("roi_xyxy")
        if not roi:
            continue

        standard_lab = get_standard_lab(item)
        if standard_lab is None:
            continue

        before_lab = item.get("before_lab")

        if before_lab is None:
            before_lab = get_before_lab_from_roi(lab, roi)
        else:
            before_lab = np.asarray(before_lab, dtype=np.float64)

        if before_lab is None:
            continue

        dL = float(standard_lab[0] - before_lab[0])
        da = float(standard_lab[1] - before_lab[1])
        db = float(standard_lab[2] - before_lab[2])

        mask = make_soft_roi_mask(
            (h, w),
            roi,
            feather=feather,
            mode=mask_mode,
        )

        mask = mask * protect

        out[:, :, 0] = out[:, :, 0] + l_strength * dL * mask
        out[:, :, 1] = out[:, :, 1] + ab_strength * da * mask
        out[:, :, 2] = out[:, :, 2] + ab_strength * db * mask

        used_count += 1

    print(f"胶块视觉修正数量：{used_count}")

    return out


# ============================================================
# 6. 对比图和统计信息
# ============================================================

def make_compare_image(original_bgr: np.ndarray, preview_bgr: np.ndarray) -> np.ndarray:
    """
    生成左右对比图。
    """
    h1, w1 = original_bgr.shape[:2]
    h2, w2 = preview_bgr.shape[:2]

    h = max(h1, h2)
    gap = 20
    top = 55

    canvas = np.full((h + top, w1 + w2 + gap, 3), 255, dtype=np.uint8)

    canvas[top:top + h1, 0:w1] = original_bgr
    canvas[top:top + h2, w1 + gap:w1 + gap + w2] = preview_bgr

    cv2.putText(
        canvas,
        "Original",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Visual Preview",
        (w1 + gap + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return canvas


def mean_lab_in_mask(lab: np.ndarray, mask: np.ndarray) -> Optional[List[float]]:
    """
    计算 mask 区域平均 Lab。
    """
    m = mask > 0.5

    if np.count_nonzero(m) == 0:
        return None

    values = lab[m]
    mean = values.mean(axis=0)
    return [float(mean[0]), float(mean[1]), float(mean[2])]


# ============================================================
# 7. 处理单个 report
# ============================================================

def process_one_report(
    report_path: Path,
    root: Path,
    ab_strength: float,
    l_strength: float,
    bg_strength: float,
    bg_min_L: float,
    bg_max_saturation: float,
    feather: int,
    mask_mode: str,
    photo_override: Optional[str] = None,
) -> Optional[dict]:
    """
    读取一个 report.json，生成 visual_preview。
    """
    report = read_json_safe(report_path)

    if report is None:
        return None

    photo_path = get_photo_path(
        report=report,
        report_path=report_path,
        root=root,
        photo_override=photo_override,
    )

    bgr = imread_unicode(photo_path)

    if bgr is None:
        raise RuntimeError(f"图片读取失败：{photo_path}")

    target_colors = report.get("target_colors") or []

    if not target_colors:
        raise ValueError(f"{report_path} 里没有 target_colors，无法做胶块视觉修正")

    lab_before = bgr_to_lab_image(bgr)

    bg_mask = build_background_mask(
        bgr=bgr,
        lab=lab_before,
        target_colors=target_colors,
        bg_min_L=bg_min_L,
        bg_max_saturation=bg_max_saturation,
        exclude_targets=True,
    )

    bg_before_mean_lab = mean_lab_in_mask(lab_before, bg_mask)

    lab_bg = apply_background_neutralization(
        lab=lab_before,
        bg_mask=bg_mask,
        strength=bg_strength,
    )

    lab_preview = apply_target_visual_correction(
        bgr=bgr,
        lab=lab_bg,
        target_colors=target_colors,
        ab_strength=ab_strength,
        l_strength=l_strength,
        feather=feather,
        mask_mode=mask_mode,
        protect_extreme_light=True,
    )

    bg_after_mean_lab = mean_lab_in_mask(lab_preview, bg_mask)

    preview_bgr = lab_to_bgr_image(lab_preview)

    out_dir = report_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    preview_path = out_dir / "14_visual_preview.png"
    compare_path = out_dir / "15_visual_preview_compare.png"
    bg_mask_path = out_dir / "16_background_mask.png"
    info_path = out_dir / "14_visual_preview_info.json"

    ok1 = imwrite_unicode(preview_path, preview_bgr)
    ok2 = imwrite_unicode(compare_path, make_compare_image(bgr, preview_bgr))
    ok3 = imwrite_unicode(bg_mask_path, (bg_mask * 255).astype(np.uint8))

    if not ok1:
        raise RuntimeError(f"保存失败：{preview_path}")
    if not ok2:
        raise RuntimeError(f"保存失败：{compare_path}")
    if not ok3:
        raise RuntimeError(f"保存失败：{bg_mask_path}")

    info = {
        "report": str(report_path),
        "photo": str(photo_path),
        "output_visual_preview": str(preview_path),
        "output_compare": str(compare_path),
        "output_background_mask": str(bg_mask_path),
        "strategy": {
            "name": "preserve_L_correct_ab",
            "description": "保留原图 L 明暗层次；胶块 a/b 向标准色靠近；白色/灰白背景 a/b 向 0 靠近。",
            "target_ab_strength": ab_strength,
            "target_l_strength": l_strength,
            "background_ab_neutral_strength": bg_strength,
            "background_min_L": bg_min_L,
            "background_max_saturation": bg_max_saturation,
            "target_mask_mode": mask_mode,
            "feather": feather,
        },
        "background_lab_mean": {
            "before": bg_before_mean_lab,
            "after": bg_after_mean_lab,
            "note": "理想中性白/灰背景的 a,b 应接近 0；L 不强制变成 100，以保留阴影和明暗层次。",
        },
    }

    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return info


# ============================================================
# 8. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=".",
        help="项目根目录，默认当前目录",
    )

    parser.add_argument(
        "--report",
        default=None,
        help="单个 report.json 路径，例如 output_root_poly2/report.json",
    )

    parser.add_argument(
        "--reports",
        default="output*/report.json",
        help="批量处理 report.json 的 glob 规则，默认 output*/report.json",
    )

    parser.add_argument(
        "--photo",
        default=None,
        help="手动指定原图路径。当 report.json 里的 input.photo 找不到时使用。",
    )

    parser.add_argument(
        "--ab-strength",
        type=float,
        default=0.50,
        help="胶块 a/b 向标准色靠近的强度，默认 0.50",
    )

    parser.add_argument(
        "--l-strength",
        type=float,
        default=0.10,
        help="胶块 L 向标准亮度靠近的强度，默认 0.10。想完全保留明暗可设 0",
    )

    parser.add_argument(
        "--bg-strength",
        type=float,
        default=0.55,
        help="背景 a/b 向中性 0 靠近的强度，默认 0.55",
    )

    parser.add_argument(
        "--bg-min-L",
        type=float,
        default=50.0,
        help="背景候选区域最低 L，默认 50",
    )

    parser.add_argument(
        "--bg-max-saturation",
        type=float,
        default=80.0,
        help="背景候选区域最高 HSV 饱和度，默认 80",
    )

    parser.add_argument(
        "--feather",
        type=int,
        default=25,
        help="胶块 ROI 边缘羽化大小，默认 25",
    )

    parser.add_argument(
        "--mask-mode",
        choices=["rectangle", "ellipse"],
        default="rectangle",
        help="胶块视觉修正 mask 形状，默认 rectangle",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if args.report:
        report_paths = [Path(args.report).resolve()]
    else:
        report_paths = sorted(Path(p).resolve() for p in glob.glob(str(root / args.reports)))

    if not report_paths:
        print("没有找到 report.json")
        return

    print(f"共找到 {len(report_paths)} 个 report.json")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for report_path in report_paths:
        print("\n----------------------------------------")
        print(f"开始处理：{report_path}")

        try:
            info = process_one_report(
                report_path=report_path,
                root=root,
                ab_strength=args.ab_strength,
                l_strength=args.l_strength,
                bg_strength=args.bg_strength,
                bg_min_L=args.bg_min_L,
                bg_max_saturation=args.bg_max_saturation,
                feather=args.feather,
                mask_mode=args.mask_mode,
                photo_override=args.photo,
            )

            if info is None:
                skip_count += 1
                continue

            success_count += 1

            print("处理完成")
            print("visual_preview:", info["output_visual_preview"])
            print("compare:", info["output_compare"])
            print("background_mask:", info["output_background_mask"])

            bg_before = info.get("background_lab_mean", {}).get("before")
            bg_after = info.get("background_lab_mean", {}).get("after")

            if bg_before is not None and bg_after is not None:
                print("背景平均 Lab before:", bg_before)
                print("背景平均 Lab after :", bg_after)

        except Exception as e:
            fail_count += 1
            print("处理失败")
            print("原因：", e)

    print("\n========================================")
    print("全部处理结束")
    print(f"成功：{success_count}")
    print(f"跳过：{skip_count}")
    print(f"失败：{fail_count}")
    print("========================================")


if __name__ == "__main__":
    main()