# 存储了12个标准Lab值。类别查询

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color_math import delta_e_2000


@dataclass(frozen=True)
class ColorStandard:
    code: str
    name: str
    lab: tuple[float, float, float]
    # 标准颜色数据结构
    @property
    def label(self) -> str:
        return f"{self.code}{self.name}"

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "label": self.label,
            "lab": [float(v) for v in self.lab],
        }


STANDARD_COLORS: list[ColorStandard] = [
    ColorStandard("W015", "桔红", (55.74, 43.62, 41.71)),
    ColorStandard("W016", "橙红色", (52.79, 33.02, 27.21)),
    ColorStandard("W031", "淡雅黄", (77.97, -2.16, 28.66)),
    ColorStandard("W032", "柠檬黄", (82.78, -5.38, 30.47)),
    ColorStandard("W047", "浅棕桐", (65.94, 0.97, 14.80)),
    ColorStandard("W048", "灰米色", (73.34, 3.62, 10.90)),
    ColorStandard("W063", "木纹灰", (75.47, -1.39, 4.54)),
    ColorStandard("W064", "乌托银", (58.72, -2.12, 0.87)),
    ColorStandard("W079", "钛晶灰", (41.96, 0.97, 2.06)),
    ColorStandard("W080", "中行灰", (50.81, 0.50, 0.62)),
    ColorStandard("W095", "暗灰色", (42.77, 1.65, -1.27)),
    ColorStandard("W096", "慕云灰", (76.90, -4.65, 11.02)),
]


STANDARD_BY_CODE = {item.code.upper(): item for item in STANDARD_COLORS}
STANDARD_BY_NAME = {item.name: item for item in STANDARD_COLORS}


def normalize_label(text: str) -> str:
    return text.strip().replace(" ", "").replace("\t", "")


def resolve_standard(label: str | None) -> ColorStandard | None:
    # 用户输入
    if not label:
        return None

    raw = normalize_label(label)
    upper = raw.upper()

    if upper in STANDARD_BY_CODE:
        return STANDARD_BY_CODE[upper]
    if raw in STANDARD_BY_NAME:
        return STANDARD_BY_NAME[raw]

    for item in STANDARD_COLORS:
        if item.code.upper() in upper or item.name in raw or item.label in raw:
            return item

    return None


def get_standard_lab(label: str) -> np.ndarray:
    item = resolve_standard(label)
    if item is None:
        raise ValueError(f"未找到标准颜色类别：{label}")
    return np.asarray(item.lab, dtype=np.float32)


def nearest_standards(lab: np.ndarray, top_k: int = 3) -> list[dict]:
    # 计算与输入颜色最接近的标准颜色
    lab = np.asarray(lab, dtype=np.float32).reshape(1, 3)
    rows = []
    for item in STANDARD_COLORS:
        ref_lab = np.asarray(item.lab, dtype=np.float32).reshape(1, 3)
        de = float(delta_e_2000(lab, ref_lab)[0])
        rows.append({
            "code": item.code,
            "name": item.name,
            "label": item.label,
            "lab": [float(v) for v in item.lab],
            "delta_e_2000": de,
        })
    rows.sort(key=lambda x: x["delta_e_2000"])
    return rows[:top_k]


def standards_as_rows() -> list[dict]:
    return [item.as_dict() for item in STANDARD_COLORS]
    # 返回标准颜色数据库


def standard_codes() -> list[str]:
    # 返回编号
    return [item.code for item in STANDARD_COLORS]


def parse_standard_sequence(sequence: str | None) -> list[str]:
    # 解析待验证颜色顺序
    if sequence is None or normalize_label(sequence).lower() in {"", "all", "builtin", "default", "全部"}:
        return standard_codes()

    if normalize_label(sequence).lower() in {"manual", "手动"}:
        return []

    items = []
    for part in sequence.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        standard = resolve_standard(part)
        if standard is None:
            raise ValueError(f"target_sequence 中存在未识别类别：{part}")
        items.append(standard.code)
    return items


def standards_help_text() -> str:
    # 说明信息
    lines = ["可输入标准颜色编号或名称"]
    for item in STANDARD_COLORS:
        L, a, b = item.lab
        lines.append(f"  {item.code} {item.name}: Lab=({L:.2f}, {a:.2f}, {b:.2f})")
    return "\n".join(lines)
