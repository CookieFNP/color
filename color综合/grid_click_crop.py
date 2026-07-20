
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# =========================
# 中文路径兼容读写
# =========================
def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray, jpg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        ext = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
    elif suffix == ".png":
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 2]
    else:
        ext = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 2]

    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise OSError(f"图片编码失败: {path}")
    encoded.tofile(str(path))


# =========================
# 数据结构
# =========================
@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2


@dataclass
class Selection:
    order: int
    name: str
    cell: Cell


# =========================
# 名单读取
# =========================
def load_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"名单文件不存在: {path}")

    names: list[str] = []

    if path.suffix.lower() == ".csv":
        # CSV 默认取每一行拼接后的非空内容。
        # 例如：
        # 101,玉石灰  ->  101玉石灰
        # 101玉石灰   ->  101玉石灰
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                parts = [part.strip() for part in row if part.strip()]
                if parts:
                    names.append("".join(parts))
    else:
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)

    if not names:
        raise ValueError("名单为空，请保证每行至少有一个名称。")

    return names


def sanitize_filename(name: str) -> str:
    # Windows 文件名非法字符：<>:"/\|?*
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.strip().rstrip(". ")
    return name or "未命名"


# =========================
# 主交互程序
# =========================
class GridCropper:
    def __init__(
        self,
        image: np.ndarray,
        names: list[str],
        out_dir: Path,
        pad: int,
        add_index: bool,
        max_width: int,
        max_height: int,
        output_ext: str,
    ) -> None:
        self.image = image
        self.names = names
        self.out_dir = out_dir
        self.pad = max(0, pad)
        self.add_index = add_index
        self.output_ext = output_ext.lower()

        self.height, self.width = image.shape[:2]

        scale_w = max_width / self.width
        scale_h = max_height / self.height
        self.scale = min(1.0, scale_w, scale_h)

        self.display_width = max(1, int(round(self.width * self.scale)))
        self.display_height = max(1, int(round(self.height * self.scale)))

        self.window_name = "Grid Click Crop"

        # draw 阶段 / select 阶段
        self.phase = "draw"
        self.line_mode = "vertical"

        # 原图坐标
        self.x_lines: list[int] = []
        self.y_lines: list[int] = []
        self.action_history: list[tuple[str, int]] = []

        self.cells: list[Cell] = []
        self.selections: list[Selection] = []
        self.selected_cell_keys: set[tuple[int, int]] = set()

        self.status_message = ""

    def original_xy(self, display_x: int, display_y: int) -> tuple[int, int]:
        x = int(round(display_x / self.scale))
        y = int(round(display_y / self.scale))
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))
        return x, y

    def display_xy(self, x: int, y: int) -> tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))

    @staticmethod
    def deduplicate_sorted(values: list[int], min_gap: int = 3) -> list[int]:
        result: list[int] = []
        for value in sorted(values):
            if not result or value - result[-1] >= min_gap:
                result.append(value)
        return result

    def build_cells(self) -> bool:
        xs = self.deduplicate_sorted(self.x_lines)
        ys = self.deduplicate_sorted(self.y_lines)

        if len(xs) < 2 or len(ys) < 2:
            self.status_message = "Need >= 2 vertical and >= 2 horizontal lines."
            print("至少需要 2 条竖线和 2 条横线，且要包含区域外边界。")
            return False

        self.x_lines = xs
        self.y_lines = ys

        cells: list[Cell] = []
        for row in range(len(ys) - 1):
            for col in range(len(xs) - 1):
                x1, x2 = xs[col], xs[col + 1]
                y1, y2 = ys[row], ys[row + 1]
                if x2 > x1 and y2 > y1:
                    cells.append(Cell(row, col, x1, y1, x2, y2))

        if not cells:
            self.status_message = "No valid cells."
            return False

        self.cells = cells
        self.phase = "select"
        self.status_message = f"{len(cells)} cells. Click in name order."
        print(f"\n已生成 {len(cells)} 个格子。")
        print(f"名单共有 {len(self.names)} 个名称。")
        if len(cells) != len(self.names):
            print(
                f"注意：格子数 {len(cells)} 与名称数 {len(self.names)} 不一致。"
                "允许继续，但最多只能选择较少的一方。"
            )
        print(f"当前待命名：1/{len(self.names)}  {self.names[0]}")
        return True

    def find_cell(self, x: int, y: int) -> Cell | None:
        for cell in self.cells:
            if cell.contains(x, y):
                return cell
        return None

    def add_line(self, x: int, y: int) -> None:
        if self.line_mode == "vertical":
            self.x_lines.append(x)
            self.action_history.append(("vertical", x))
            self.status_message = f"Added vertical x={x}"
            print(f"添加竖线 x={x}")
        else:
            self.y_lines.append(y)
            self.action_history.append(("horizontal", y))
            self.status_message = f"Added horizontal y={y}"
            print(f"添加横线 y={y}")

    def undo_line(self) -> None:
        if not self.action_history:
            self.status_message = "Nothing to undo."
            return

        kind, value = self.action_history.pop()
        target = self.x_lines if kind == "vertical" else self.y_lines

        for i in range(len(target) - 1, -1, -1):
            if target[i] == value:
                target.pop(i)
                break

        self.status_message = f"Undo {kind} {value}"
        print(f"撤销：{kind} {value}")

    def clear_lines(self) -> None:
        self.x_lines.clear()
        self.y_lines.clear()
        self.action_history.clear()
        self.status_message = "All lines cleared."
        print("已清空全部横竖线。")

    def select_cell(self, x: int, y: int) -> None:
        if len(self.selections) >= len(self.names):
            self.status_message = "All names have been assigned."
            print("名单中的名称已经全部分配完成。")
            return

        cell = self.find_cell(x, y)
        if cell is None:
            self.status_message = "Click is outside cells."
            print("点击位置不在任何格子内。")
            return

        key = (cell.row, cell.col)
        if key in self.selected_cell_keys:
            self.status_message = f"Cell R{cell.row + 1}C{cell.col + 1} already selected."
            print(f"这个格子已经选择过：第 {cell.row + 1} 行，第 {cell.col + 1} 列。")
            return

        order = len(self.selections) + 1
        name = self.names[order - 1]
        self.selections.append(Selection(order, name, cell))
        self.selected_cell_keys.add(key)

        print(
            f"[{order}/{len(self.names)}] {name} "
            f"<- 第 {cell.row + 1} 行，第 {cell.col + 1} 列"
        )

        if len(self.selections) < len(self.names):
            next_name = self.names[len(self.selections)]
            self.status_message = f"Next {len(self.selections) + 1}: {next_name}"
            print(f"下一个：{len(self.selections) + 1}/{len(self.names)}  {next_name}")
        else:
            self.status_message = "All names assigned. Press Enter to save."
            print("全部名称已点击完成，按 Enter 保存。")

    def undo_selection(self) -> None:
        if not self.selections:
            self.status_message = "Nothing to undo."
            return

        item = self.selections.pop()
        self.selected_cell_keys.discard((item.cell.row, item.cell.col))
        self.status_message = f"Undo {item.order}: {item.name}"
        print(f"撤销：{item.order}  {item.name}")

        next_index = len(self.selections)
        if next_index < len(self.names):
            print(f"当前待命名：{next_index + 1}/{len(self.names)}  {self.names[next_index]}")

    def draw_display(self) -> np.ndarray:
        display = cv2.resize(
            self.image,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )

        # 网格线
        for x in sorted(self.x_lines):
            dx, _ = self.display_xy(x, 0)
            cv2.line(display, (dx, 0), (dx, self.display_height - 1), (0, 0, 255), 2)

        for y in sorted(self.y_lines):
            _, dy = self.display_xy(0, y)
            cv2.line(display, (0, dy), (self.display_width - 1, dy), (0, 0, 255), 2)

        # 点击选择结果
        if self.phase == "select":
            for item in self.selections:
                c = item.cell
                x1, y1 = self.display_xy(c.x1, c.y1)
                x2, y2 = self.display_xy(c.x2, c.y2)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

                label = str(item.order)
                tx = x1 + 5
                ty = min(y2 - 5, y1 + 25)
                cv2.putText(
                    display,
                    label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # 顶部状态栏
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_width, 42), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.55, display, 0.45, 0)

        if self.phase == "draw":
            text = (
                f"DRAW | mode={self.line_mode.upper()} | "
                f"V:{len(self.x_lines)} H:{len(self.y_lines)} | "
                "V/H switch, Z undo, C clear, Enter confirm"
            )
        else:
            text = (
                f"SELECT | {len(self.selections)}/{len(self.names)} | "
                "click cell, Z undo, Enter save"
            )

        cv2.putText(
            display,
            text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return display

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        ox, oy = self.original_xy(x, y)

        if self.phase == "draw":
            self.add_line(ox, oy)
        else:
            self.select_cell(ox, oy)

    def save_results(self) -> None:
        if not self.selections:
            raise RuntimeError("还没有选择任何格子，无法保存。")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = self.out_dir / "mapping.csv"

        used_names: dict[str, int] = {}
        rows: list[list[object]] = []

        for item in self.selections:
            cell = item.cell

            x1 = min(max(cell.x1 + self.pad, 0), self.width)
            y1 = min(max(cell.y1 + self.pad, 0), self.height)
            x2 = min(max(cell.x2 - self.pad, 0), self.width)
            y2 = min(max(cell.y2 - self.pad, 0), self.height)

            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"第 {item.order} 个格子裁剪后为空，请减小 --pad。"
                )

            crop = self.image[y1:y2, x1:x2]

            safe_name = sanitize_filename(item.name)
            used_names[safe_name] = used_names.get(safe_name, 0) + 1
            duplicate_no = used_names[safe_name]

            if self.add_index:
                stem = f"{item.order:03d}_{safe_name}"
            else:
                stem = safe_name

            if duplicate_no > 1:
                stem = f"{stem}_{duplicate_no}"

            filename = f"{stem}{self.output_ext}"
            output_path = self.out_dir / filename
            imwrite_unicode(output_path, crop)

            rows.append(
                [
                    item.order,
                    item.name,
                    filename,
                    cell.row + 1,
                    cell.col + 1,
                    x1,
                    y1,
                    x2,
                    y2,
                    x2 - x1,
                    y2 - y1,
                ]
            )

        with mapping_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "顺序",
                    "名称",
                    "文件名",
                    "网格行",
                    "网格列",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "宽度",
                    "高度",
                ]
            )
            writer.writerows(rows)

        print(f"\n保存完成：{len(self.selections)} 张")
        print(f"输出目录：{self.out_dir.resolve()}")
        print(f"映射表：{mapping_path.resolve()}")

        if len(self.selections) < len(self.names):
            print(
                f"注意：名单共 {len(self.names)} 个，仅保存了前 "
                f"{len(self.selections)} 个名称对应的图片。"
            )

    def run(self) -> None:
        print("\n========== 阶段 1：画网格线 ==========")
        print("左键：在鼠标位置添加当前方向的整条分割线")
        print("V：切换到竖线模式")
        print("H：切换到横线模式")
        print("Z / Backspace：撤销上一条线")
        print("C：清空全部线")
        print("Enter：确认网格并进入点击命名")
        print("Esc：退出")
        print("\n注意：需要把最左、最右、最上、最下的外边界也画进去。")
        print("当前模式：竖线")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while True:
            display = self.draw_display()
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            if key == 27:  # Esc
                print("已取消。")
                break

            if self.phase == "draw":
                if key in (ord("v"), ord("V")):
                    self.line_mode = "vertical"
                    self.status_message = "Vertical mode."
                    print("已切换：竖线模式")
                elif key in (ord("h"), ord("H")):
                    self.line_mode = "horizontal"
                    self.status_message = "Horizontal mode."
                    print("已切换：横线模式")
                elif key in (ord("z"), ord("Z"), 8):
                    self.undo_line()
                elif key in (ord("c"), ord("C")):
                    self.clear_lines()
                elif key in (10, 13):
                    self.build_cells()

            else:
                if key in (ord("z"), ord("Z"), 8):
                    self.undo_selection()
                elif key in (10, 13):
                    if not self.selections:
                        print("请至少点击一个格子后再保存。")
                        continue

                    self.save_results()
                    break

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="手动画网格线，并按名单顺序点击格子后批量裁剪命名。"
    )
    parser.add_argument("--image", required=True, type=Path, help="待分割的大图")
    parser.add_argument(
        "--names",
        required=True,
        type=Path,
        help="名称名单：txt 每行一个；csv 每行各列会自动拼接",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cropped_output"),
        help="输出目录，默认 cropped_output",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=3,
        help="每个格子四周向内缩的像素，避免保留分割线，默认 3",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="文件名前不添加 001_、002_ 顺序前缀",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="交互窗口最大宽度，默认 1600",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=900,
        help="交互窗口最大高度，默认 900",
    )
    parser.add_argument(
        "--ext",
        choices=[".jpg", ".png"],
        default=".jpg",
        help="输出图片格式，默认 .jpg",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        image = imread_unicode(args.image)
        names = load_names(args.names)

        cropper = GridCropper(
            image=image,
            names=names,
            out_dir=args.out,
            pad=args.pad,
            add_index=not args.no_index,
            max_width=args.max_width,
            max_height=args.max_height,
            output_ext=args.ext,
        )
        cropper.run()
        return 0

    except Exception as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
