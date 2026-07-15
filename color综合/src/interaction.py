from __future__ import annotations

import cv2
import numpy as np


def resize_for_display(img_bgr: np.ndarray, max_w: int = 1300, max_h: int = 850) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def select_four_points(img_bgr: np.ndarray) -> np.ndarray:
    display_img, scale = resize_for_display(img_bgr)
    temp = display_img.copy()
    points: list[list[float]] = []
    instructions = ["click top-left", "click top-right", "click bottom-right", "click bottom-left"]

    def redraw():
        nonlocal temp
        temp = display_img.copy()
        for i, p in enumerate(points):
            px = int(p[0] * scale)
            py = int(p[1] * scale)
            cv2.circle(temp, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(temp, str(i + 1), (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        text = instructions[len(points)] if len(points) < 4 else "Enter confirm | R reset | Esc cancel"
        cv2.rectangle(temp, (0, 0), (temp.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(temp, text, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x / scale, y / scale])
            redraw()

    redraw()
    cv2.namedWindow("select ColorChecker corners", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("select ColorChecker corners", mouse_callback)

    print("\n请依次点击 ColorChecker 四角：左上、右上、右下、左下。")
    print("点完按 Enter；R 重选；Esc 取消。")

    while True:
        cv2.imshow("select ColorChecker corners", temp)
        key = cv2.waitKey(20) & 0xFF
        if key in [13, 10] and len(points) == 4:
            break
        if key in [ord("r"), ord("R")]:
            points.clear()
            redraw()
        if key == 27:
            cv2.destroyWindow("select ColorChecker corners")
            raise RuntimeError("用户取消了色卡四角选择")

    cv2.destroyWindow("select ColorChecker corners")
    return np.asarray(points, dtype=np.float32)


def select_roi(
    img_bgr: np.ndarray,
    window_name: str,
    prompt: str,
) -> tuple[int, int, int, int]:
    display_img, scale = resize_for_display(img_bgr)
    base = display_img.copy()
    temp = base.copy()

    drawing = False
    start: tuple[int, int] | None = None
    end: tuple[int, int] | None = None
    roi_disp: tuple[int, int, int, int] | None = None

    def redraw(current=None):
        nonlocal temp
        temp = base.copy()
        cv2.rectangle(temp, (0, 0), (temp.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(temp, prompt[:95], (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, cv2.LINE_AA)
        if start is not None:
            p2 = current if current is not None else end
            if p2 is not None:
                cv2.rectangle(temp, start, p2, (0, 0, 255), 2)

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start, end, roi_disp
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start = (x, y)
            end = (x, y)
            redraw((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            end = (x, y)
            redraw((x, y))
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            end = (x, y)
            if start is not None:
                x1, y1 = start
                x2, y2 = end
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                if abs(x2 - x1) >= 5 and abs(y2 - y1) >= 5:
                    roi_disp = (x1, y1, x2, y2)
            redraw()

    redraw()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print(f"\n{prompt}")
    print("拖框后按 Enter；R 重选；Esc 取消。")

    while True:
        cv2.imshow(window_name, temp)
        key = cv2.waitKey(20) & 0xFF
        if key in [13, 10] and roi_disp is not None:
            x1, y1, x2, y2 = roi_disp
            cv2.destroyWindow(window_name)
            return (
                int(round(x1 / scale)),
                int(round(y1 / scale)),
                int(round(x2 / scale)),
                int(round(y2 / scale)),
            )
        if key in [ord("r"), ord("R")]:
            drawing = False
            start = None
            end = None
            roi_disp = None
            redraw()
        if key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("用户取消了 ROI 选择")
