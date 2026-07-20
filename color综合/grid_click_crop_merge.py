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

    @property
    def key(self) -> tuple[int, int]:
        return self.row, self.col

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2


@dataclass
class Selection:
    order: int
    name: str
    cells: list[Cell]


# =========================
# 名单读取
# =========================
def load_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"名单文件不存在: {path}")

    names: list[str] = []

    if path.suffix.lower() == ".csv":
        # CSV 每一行的非空列自动拼接。
        # 例如：101,玉石灰 -> 101玉石灰
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

        self.window_name = "Grid Click Crop - Merge Cells"

        # draw 阶段 / select 阶段
        self.phase = "draw"
        self.line_mode = "vertical"

        # 原图坐标
        self.x_lines: list[int] = []
        self.y_lines: list[int] = []
        self.action_history: list[tuple[str, int]] = []

        self.cells: list[Cell] = []

        # 已确认的“名称 -> 多个格子”
        self.selections: list[Selection] = []
        self.used_cell_keys: set[tuple[int, int]] = set()

        # 当前名称正在选择、尚未确认的格子
        self.current_cells: list[Cell] = []
        self.current_cell_keys: set[tuple[int, int]] = set()

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

    @staticmethod
    def group_bounds(cells: list[Cell]) -> tuple[int, int, int, int]:
        if not cells:
            raise ValueError("格子组为空。")
        return (
            min(cell.x1 for cell in cells),
            min(cell.y1 for cell in cells),
            max(cell.x2 for cell in cells),
            max(cell.y2 for cell in cells),
        )

    @staticmethod
    def is_full_rectangle(cells: list[Cell]) -> bool:
        """判断选中的网格是否完整覆盖一个行列矩形。"""
        if not cells:
            return False
        rows = {cell.row for cell in cells}
        cols = {cell.col for cell in cells}
        expected = {(row, col) for row in rows for col in cols}
        actual = {cell.key for cell in cells}
        return actual == expected

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
        self.status_message = f"{len(cells)} base cells. Select one or more per name."

        print(f"\n已生成 {len(cells)} 个基础小格。")
        print(f"名单共有 {len(self.names)} 个最终结果。")
        print("基础小格数和最终名称数可以不同，例如 80 个小格合并为 50 张结果。")
        print("同一名称可点击 1 个或多个小格，再按空格/Enter 确认。")
        print(f"当前待命名：1/{len(self.names)}  {self.names[0]}")
        return True

    def find_cell(self, x: int, y: int) -> Cell | None:
        for cell in self.cells:
            if cell.contains(x, y):
                return cell
        return None

    # -------------------------
    # 画线阶段
    # -------------------------
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

    # -------------------------
    # 合并选择阶段
    # -------------------------
    def current_name(self) -> str | None:
        index = len(self.selections)
        if index >= len(self.names):
            return None
        return self.names[index]

    def toggle_current_cell(self, x: int, y: int) -> None:
        if len(self.selections) >= len(self.names):
            self.status_message = "All names assigned. Press Enter to save."
            print("名单中的名称已经全部分配完成，按 Enter 保存。")
            return

        cell = self.find_cell(x, y)
        if cell is None:
            self.status_message = "Click is outside cells."
            print("点击位置不在任何格子内。")
            return

        if cell.key in self.used_cell_keys:
            self.status_message = (
                f"Cell R{cell.row + 1}C{cell.col + 1} belongs to a committed result."
            )
            print(
                f"这个格子已经属于之前确认的结果："
                f"第 {cell.row + 1} 行，第 {cell.col + 1} 列。"
            )
            return

        # 再点一次当前格子即可取消
        if cell.key in self.current_cell_keys:
            self.current_cells = [item for item in self.current_cells if item.key != cell.key]
            self.current_cell_keys.discard(cell.key)
            self.status_message = f"Removed R{cell.row + 1}C{cell.col + 1} from current group."
            print(f"取消当前格子：第 {cell.row + 1} 行，第 {cell.col + 1} 列")
            return

        self.current_cells.append(cell)
        self.current_cell_keys.add(cell.key)

        name = self.current_name() or ""
        self.status_message = (
            f"Current {len(self.selections) + 1}: {name} | "
            f"{len(self.current_cells)} cell(s)"
        )
        print(
            f"当前名称 [{len(self.selections) + 1}/{len(self.names)}] {name}："
            f"加入第 {cell.row + 1} 行，第 {cell.col + 1} 列"
        )

    def commit_current_group(self) -> bool:
        if len(self.selections) >= len(self.names):
            print("全部名称已经确认，按 Enter 保存。")
            return False

        if not self.current_cells:
            self.status_message = "Select at least one cell before committing."
            print("当前名称还没有选择任何格子。")
            return False

        order = len(self.selections) + 1
        name = self.names[order - 1]
        cells = list(self.current_cells)

        if not self.is_full_rectangle(cells):
            print(
                "提示：当前选择不是完整矩形，保存时会按照所有已选格子的"
                "最小外接矩形裁剪，因此中间未选区域也会进入图片。"
            )

        selection = Selection(order=order, name=name, cells=cells)
        self.selections.append(selection)

        for cell in cells:
            self.used_cell_keys.add(cell.key)

        self.current_cells.clear()
        self.current_cell_keys.clear()

        cell_text = "、".join(
            f"R{cell.row + 1}C{cell.col + 1}"
            for cell in sorted(cells, key=lambda c: (c.row, c.col))
        )
        print(
            f"[{order}/{len(self.names)}] 已确认 {name}，"
            f"合并 {len(cells)} 个小格：{cell_text}"
        )

        if len(self.selections) < len(self.names):
            next_name = self.names[len(self.selections)]
            self.status_message = f"Next {len(self.selections) + 1}: {next_name}"
            print(
                f"下一个：{len(self.selections) + 1}/{len(self.names)}  {next_name}"
            )
        else:
            self.status_message = "All names assigned. Press Enter to save."
            print("全部名称已确认，再按一次 Enter 保存。")

        return True

    def undo_selection(self) -> None:
        # 当前组有格子：仅撤销当前组最后一次点击
        if self.current_cells:
            cell = self.current_cells.pop()
            self.current_cell_keys.discard(cell.key)
            self.status_message = (
                f"Removed current R{cell.row + 1}C{cell.col + 1}"
            )
            print(
                f"撤销当前组最后一个格子："
                f"第 {cell.row + 1} 行，第 {cell.col + 1} 列"
            )
            return

        # 当前组为空：撤回上一个已经确认的结果，重新进入编辑
        if self.selections:
            item = self.selections.pop()
            for cell in item.cells:
                self.used_cell_keys.discard(cell.key)

            self.current_cells = list(item.cells)
            self.current_cell_keys = {cell.key for cell in item.cells}
            self.status_message = f"Reopened {item.order}: {item.name}"
            print(
                f"已撤回并重新编辑：[{item.order}/{len(self.names)}] "
                f"{item.name}，当前含 {len(item.cells)} 个小格。"
            )
            return

        self.status_message = "Nothing to undo."
        print("没有可撤销的选择。")

    def clear_current_group(self) -> None:
        if not self.current_cells:
            print("当前名称没有已选格子。")
            return
        self.current_cells.clear()
        self.current_cell_keys.clear()
        self.status_message = "Current group cleared."
        print("已清空当前名称的临时选区。")

    # -------------------------
    # 界面绘制
    # -------------------------
    def draw_group(
        self,
        display: np.ndarray,
        cells: list[Cell],
        color: tuple[int, int, int],
        label: str,
        cell_thickness: int = 2,
        bounds_thickness: int = 4,
    ) -> None:
        if not cells:
            return

        for cell in cells:
            x1, y1 = self.display_xy(cell.x1, cell.y1)
            x2, y2 = self.display_xy(cell.x2, cell.y2)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, cell_thickness)

        bx1, by1, bx2, by2 = self.group_bounds(cells)
        x1, y1 = self.display_xy(bx1, by1)
        x2, y2 = self.display_xy(bx2, by2)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, bounds_thickness)

        tx = x1 + 6
        ty = min(max(y1 + 27, 27), max(27, y2 - 6))
        cv2.putText(
            display,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            color,
            2,
            cv2.LINE_AA,
        )

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

        if self.phase == "select":
            # 已确认：绿色
            for item in self.selections:
                self.draw_group(
                    display,
                    item.cells,
                    color=(0, 255, 0),
                    label=str(item.order),
                )

            # 当前待确认：黄色
            if self.current_cells:
                self.draw_group(
                    display,
                    self.current_cells,
                    color=(0, 255, 255),
                    label=f"{len(self.selections) + 1}?",
                )

        # 顶部状态栏
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_width, 46), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.58, display, 0.42, 0)

        if self.phase == "draw":
            text = (
                f"DRAW | mode={self.line_mode.upper()} | "
                f"V:{len(self.x_lines)} H:{len(self.y_lines)} | "
                "V/H switch, Z undo, C clear, Enter confirm"
            )
        else:
            text = (
                f"GROUP {len(self.selections) + 1}/{len(self.names)} | "
                f"current cells:{len(self.current_cells)} | "
                "click toggle, Space/Enter commit, Z undo"
            )

        cv2.putText(
            display,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
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
            self.toggle_current_cell(ox, oy)

    # -------------------------
    # 保存
    # -------------------------
    def save_results(self) -> None:
        if not self.selections:
            raise RuntimeError("还没有确认任何结果，无法保存。")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = self.out_dir / "mapping.csv"

        used_names: dict[str, int] = {}
        rows: list[list[object]] = []

        for item in self.selections:
            raw_x1, raw_y1, raw_x2, raw_y2 = self.group_bounds(item.cells)

            # pad 只作用于合并区域的最外侧，不会在内部小格之间制造缝隙
            x1 = min(max(raw_x1 + self.pad, 0), self.width)
            y1 = min(max(raw_y1 + self.pad, 0), self.height)
            x2 = min(max(raw_x2 - self.pad, 0), self.width)
            y2 = min(max(raw_y2 - self.pad, 0), self.height)

            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"第 {item.order} 个合并区域裁剪后为空，请减小 --pad。"
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

            sorted_cells = sorted(item.cells, key=lambda c: (c.row, c.col))
            grid_cells = "|".join(
                f"R{cell.row + 1}C{cell.col + 1}" for cell in sorted_cells
            )

            min_row = min(cell.row for cell in item.cells) + 1
            max_row = max(cell.row for cell in item.cells) + 1
            min_col = min(cell.col for cell in item.cells) + 1
            max_col = max(cell.col for cell in item.cells) + 1

            rows.append(
                [
                    item.order,
                    item.name,
                    filename,
                    len(item.cells),
                    grid_cells,
                    min_row,
                    max_row,
                    min_col,
                    max_col,
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
                    "合并小格数",
                    "包含网格",
                    "起始行",
                    "结束行",
                    "起始列",
                    "结束列",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "宽度",
                    "高度",
                ]
            )
            writer.writerows(rows)

        used_base_cells = sum(len(item.cells) for item in self.selections)
        print(f"\n保存完成：{len(self.selections)} 张最终图片")
        print(f"共使用：{used_base_cells}/{len(self.cells)} 个基础小格")
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
        print("Enter：确认网格并进入合并选择")
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
                elif key in (ord("c"), ord("C")):
                    self.clear_current_group()
                elif key in (32, ord("n"), ord("N")):  # Space / N
                    self.commit_current_group()
                elif key in (10, 13):  # Enter
                    if self.current_cells:
                        self.commit_current_group()
                    elif len(self.selections) >= len(self.names):
                        self.save_results()
                        break
                    elif self.selections:
                        print(
                            "当前组为空。请先点击格子；"
                            "若要提前保存已确认结果，请按 S。"
                        )
                    else:
                        print("请先选择至少一个格子。")
                elif key in (ord("s"), ord("S")):
                    if self.current_cells:
                        print("当前还有未确认格子，请先按空格/Enter 确认或按 C 清空。")
                    elif self.selections:
                        self.save_results()
                        break
                    else:
                        print("还没有确认任何结果，无法保存。")

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "手动画网格线；每个名称可选择一个或多个基础格子，"
            "按其整体外接矩形批量裁剪命名。"
        )
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
        help="每个最终合并区域四周向内缩的像素，默认 3",
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
