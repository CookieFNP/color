# OpenCV交互

from __future__ import annotations

import cv2
import numpy as np


def resize_for_display(img_bgr: np.ndarray, max_w: int = 1200, max_h: int = 800) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    return resized, scale


def select_four_points(img_bgr: np.ndarray) -> np.ndarray:
    # 按左上、右上、右下、左下顺序依次点击色卡四个角点
    display_img, scale = resize_for_display(img_bgr)
    temp = display_img.copy()
    points: list[list[float]] = []
    instructions = ["click top-left", "click top-right", "click bottom-right", "click bottom-left"]

    def mouse_callback(event, x, y, flags, param):
        nonlocal temp, points
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
            return

        points.append([x / scale, y / scale])
        temp = display_img.copy()

        for i, p in enumerate(points):
            px = int(p[0] * scale)
            py = int(p[1] * scale)
            cv2.circle(temp, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(temp, str(i + 1), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        text = instructions[len(points)] if len(points) < 4 else "Enter confirm | R reset | Esc cancel"
        cv2.putText(temp, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.namedWindow("select chart corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("select chart corners", mouse_callback)
    cv2.putText(temp, instructions[0], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    print("\n请依次点击实拍色卡四角：左上、右上、右下、左下。")
    print("点完后按 Enter 确认；按 R 重选；按 Esc 退出。")

    while True:
        cv2.imshow("select chart corners", temp)
        key = cv2.waitKey(20) & 0xFF
        if key in [13, 10] and len(points) == 4:
            break
        if key in [ord("r"), ord("R")]:
            points = []
            temp = display_img.copy()
            cv2.putText(temp, instructions[0], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if key == 27:
            cv2.destroyWindow("select chart corners")
            raise RuntimeError("用户取消了四角选择。")

    cv2.destroyWindow("select chart corners")
    return np.asarray(points, dtype=np.float32)


def select_roi(
    img_bgr: np.ndarray,
    window_name: str = "select target ROI",
    prompt: str | None = None,
    allow_cancel: bool = False,
) -> tuple[int, int, int, int] | None:

    # 框选胶块目标ROI

    display_img, scale = resize_for_display(img_bgr)
    base = display_img.copy()
    temp = base.copy()

    drawing = False
    start: tuple[int, int] | None = None
    end: tuple[int, int] | None = None
    roi_disp: tuple[int, int, int, int] | None = None

    help_text = prompt or "Drag ROI around glue block | Enter confirm | R reset | Esc cancel"

    def redraw(current_x: int | None = None, current_y: int | None = None):
        nonlocal temp
        temp = base.copy()
        cv2.rectangle(temp, (0, 0), (temp.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(temp, help_text[:80], (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)

        if start is not None and (end is not None or (current_x is not None and current_y is not None)):
            ex, ey = end if end is not None else (current_x, current_y)
            x1, y1 = min(start[0], ex), min(start[1], ey)
            x2, y2 = max(start[0], ex), max(start[1], ey)
            cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(temp, f"{x1},{y1} -> {x2},{y2}", (x1, max(62, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start, end, roi_disp
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (x, y)
            end = None
            roi_disp = None
            redraw(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            redraw(x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            end = (x, y)
            if start is not None:
                x1, y1 = min(start[0], end[0]), min(start[1], end[1])
                x2, y2 = max(start[0], end[0]), max(start[1], end[1])
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    roi_disp = (x1, y1, x2, y2)
            redraw()

    print("\n" + (prompt or "请框选胶块目标区域。"))
    print("拖动鼠标框选；Enter/Space 确认；R 重选；Esc 取消。")

    redraw()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, temp)
        key = cv2.waitKey(20) & 0xFF

        if key in [13, 10, 32]:
            if roi_disp is None:
                print("还没有有效 ROI，请先拖框。")
                continue
            x1, y1, x2, y2 = roi_disp
            cv2.destroyWindow(window_name)
            return int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)

        if key in [ord("r"), ord("R")]:
            drawing = False
            start = None
            end = None
            roi_disp = None
            redraw()

        if key == 27:
            cv2.destroyWindow(window_name)
            if allow_cancel:
                return None
            raise RuntimeError("用户取消了目标 ROI 选择。")


def parse_corners(corner_str: str) -> np.ndarray:
    """Parse 'x1,y1;x2,y2;x3,y3;x4,y4'."""
    points = []
    for item in corner_str.split(";"):
        x, y = item.split(",")
        points.append([float(x), float(y)])
    if len(points) != 4:
        raise ValueError("corners 必须包含 4 个点。")
    return np.asarray(points, dtype=np.float32)


def parse_roi(roi_str: str) -> tuple[int, int, int, int]:
    # x1, x2, y1, y2
    nums = [int(float(v.strip())) for v in roi_str.split(",")]
    if len(nums) != 4:
        raise ValueError("ROI 必须是 4 个数字。")
    x1, y1, x2, y2 = nums
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI 应为 x1,y1,x2,y2，且 x2>x1, y2>y1。")
    return x1, y1, x2, y2


def parse_roi_list(roi_list_str: str) -> list[tuple[int, int, int, int]]:

    sep = "|" if "|" in roi_list_str else ";"
    return [parse_roi(item.strip()) for item in roi_list_str.split(sep) if item.strip()]


def parse_lab(lab_str: str) -> np.ndarray:
    nums = [float(v.strip()) for v in lab_str.split(",")]
    if len(nums) != 3:
        raise ValueError("target_lab 必须是 L,a,b 三个数字。")
    return np.asarray(nums, dtype=np.float32)


def prompt_target_class(default: str | None = None) -> str:
    suffix = f"，直接回车使用 {default}" if default else ""
    text = input(f"请输入当前框选胶块的标准类别/编号，例如 W032 或 柠檬黄{suffix}：").strip()
    if not text and default:
        return default
    return text


def prompt_continue(message: str = "是否继续选择下一个胶块？输入 y 继续，其他键结束：") -> bool:
    return input(message).strip().lower() in {"y", "yes", "1", "继续", "是"}
