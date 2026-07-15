from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .color_math import delta_e_2000


@dataclass(frozen=True)
class ColorStandard:
    code: str
    name: str
    lab: tuple[float, float, float]

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


STANDARD_COLORS: list[ColorStandard] = []
STANDARD_BY_CODE: dict[str, ColorStandard] = {}
STANDARD_BY_NAME: dict[str, ColorStandard] = {}


def _rebuild_index() -> None:
    global STANDARD_BY_CODE, STANDARD_BY_NAME
    STANDARD_BY_CODE = {item.code.upper(): item for item in STANDARD_COLORS}
    STANDARD_BY_NAME = {item.name: item for item in STANDARD_COLORS}


def normalize_label(text: str) -> str:
    return str(text).strip().replace(" ", "").replace("\t", "")


def _parse_lab_text(text: str) -> tuple[float, float, float]:
    parts = str(text).strip().strip('"').replace("，", ",").split(",")
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"LAB 字段格式错误：{text}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _sort_key(item: ColorStandard):
    code = item.code.upper()
    if code.startswith("W") and code[1:].isdigit():
        return int(code[1:])
    return 999999


def load_standard_database(csv_path: str | Path) -> list[ColorStandard]:
    """
    128 色专用标准库加载。

    用户当前格式：
        编号,名称,LAB
        W001,大红,"44.5, 46.99, 19.2"

    也兼容：
        code,name,L,a,b
    """
    global STANDARD_COLORS

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到标准色数据库：{csv_path}")

    rows: list[ColorStandard] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} 没有表头")

        fieldnames = {x.strip() for x in reader.fieldnames}

        for line_no, row in enumerate(reader, start=2):
            try:
                code = str(row.get("编号") or row.get("code") or row.get("Code") or "").strip().upper()
                name = str(row.get("名称") or row.get("name") or row.get("Name") or "").strip()

                if not code:
                    raise ValueError("编号/code 为空")
                if not name:
                    name = code

                if "LAB" in fieldnames:
                    lab = _parse_lab_text(row["LAB"])
                elif "lab" in fieldnames:
                    lab = _parse_lab_text(row["lab"])
                elif {"L", "a", "b"}.issubset(fieldnames):
                    lab = (float(row["L"]), float(row["a"]), float(row["b"]))
                else:
                    raise ValueError("CSV 需要：编号,名称,LAB 或 code,name,L,a,b")

                rows.append(ColorStandard(code=code, name=name, lab=lab))
            except Exception as e:
                raise ValueError(f"{csv_path} 第 {line_no} 行解析失败：{row}，原因：{e}") from e

    if not rows:
        raise ValueError(f"{csv_path} 没有读到标准色")

    rows.sort(key=_sort_key)
    STANDARD_COLORS = rows
    _rebuild_index()

    print(f"已加载标准色库：{csv_path}，共 {len(STANDARD_COLORS)} 色")
    if len(STANDARD_COLORS) != 128:
        print(f"注意：当前不是 128 色，而是 {len(STANDARD_COLORS)} 色。调试时可以少量，正式请放完整 128 色。")
    return STANDARD_COLORS


def resolve_standard(label: str | None) -> ColorStandard | None:
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


def standard_codes() -> list[str]:
    return [item.code for item in STANDARD_COLORS]


def standards_as_rows() -> list[dict]:
    return [item.as_dict() for item in STANDARD_COLORS]


def parse_standard_sequence(sequence: str | None = "all") -> list[str]:
    if not sequence:
        return standard_codes()

    norm = normalize_label(sequence).lower()
    if norm in {"all", "全部", "128", "all128"}:
        return standard_codes()

    labels: list[str] = []
    for part in sequence.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        standard = resolve_standard(part)
        if standard is None:
            raise ValueError(f"未找到标准色：{part}")
        labels.append(standard.code)
    return labels


def nearest_standards(lab: np.ndarray, top_k: int = 10) -> list[dict]:
    if not STANDARD_COLORS:
        raise RuntimeError("标准色库为空，请先 load_standard_database(data.csv)")

    lab = np.asarray(lab, dtype=np.float64).reshape(1, 3)
    out = []
    for item in STANDARD_COLORS:
        ref = np.asarray(item.lab, dtype=np.float64).reshape(1, 3)
        de = float(delta_e_2000(lab, ref)[0])
        out.append({
            "code": item.code,
            "name": item.name,
            "label": item.label,
            "lab": [float(x) for x in item.lab],
            "delta_e_2000": de,
        })
    out.sort(key=lambda x: x["delta_e_2000"])
    return out[:top_k]


def confidence_from_nearest(nearest: list[dict]) -> dict:
    if not nearest:
        return {"level": "none", "margin": None, "reason": "no nearest"}
    top1 = float(nearest[0]["delta_e_2000"])
    top2 = float(nearest[1]["delta_e_2000"]) if len(nearest) >= 2 else 999.0
    margin = top2 - top1

    if top1 <= 3.0 and margin >= 1.5:
        level = "high"
    elif top1 <= 5.0 and margin >= 0.8:
        level = "medium"
    elif top1 <= 5.0:
        level = "medium_low_margin"
    else:
        level = "low"

    return {
        "level": level,
        "margin": margin,
        "top1_delta_e": top1,
        "top2_delta_e": top2,
    }
