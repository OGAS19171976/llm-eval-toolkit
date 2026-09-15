"""标注表:生成与校验。

为什么把"标注者"这一维**预先展开**成"一行 = 一个(题目, 标注者)":

    一致性分析(Kappa)要的是"同一题被多人**独立**标注"的矩阵。
    如果表里只留一列"标注",事后还得再对齐一次 —— 而那次对齐往往正是
    错误的来源:谁标的是哪一题、有没有人漏标、有没有人复制了别人的答案。

    预展开之后,"漏标"就是一整行缺失,一眼看得见;而"谁和谁不一致"也直接是
    同一 item_id 下几行的对比。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

__all__ = [
    "build_annotation_sheet", "validate_annotation_sheet",
    "to_csv", "to_jsonl", "from_jsonl",
]

# 每行固定带上这几个:题目 + 标注者,再加上标注内容
BASE_COLUMNS = ("item_id", "annotator")


def build_annotation_sheet(
    items,
    annotators,
    item_key: str = "id",
    fields=("label", "confidence", "note"),
    keep=(),
) -> list[dict]:
    """把题目 × 标注者展开成待标注的长表。

    `annotators` 是标注者名字的序列(例如 `["我", "另一个人"]`)。
    需要多个人独立标注同一批题,否则算不出一致性 —— 而单人标注的
    "一致性"这个概念根本不存在,只能靠事后自抽查。
    """
    annotators = list(annotators)
    if len(annotators) < 2:
        raise ValueError(
            f"至少需要 2 名标注者才能算一致性,收到 {len(annotators)} 个。"
            f"如果确实只有一个人标,那就没法用它证明标注可信。")
    rows = []
    for item in items:
        if item_key not in item:
            raise KeyError(f"题目里没有 {item_key!r} 字段: {item!r}")
        for who in annotators:
            row = {"item_id": item[item_key], "annotator": who}
            for k in keep:
                if k in item:
                    row[k] = item[k]
            for f in fields:
                row[f] = None
            rows.append(row)
    return rows


def validate_annotation_sheet(rows, required=BASE_COLUMNS + ("label",)) -> list[str]:
    """返回问题清单(空列表 = 没问题)。

    只报问题,不抛异常 —— 一张标了一半的表是**正常的工作状态**,
    应该能随时校验看看还差多少,而不是一校验就崩。
    """
    problems: list[str] = []
    if not rows:
        return ["标注表是空的"]
    columns = set(rows[0])
    for col in required:
        if col not in columns:
            problems.append(f"缺少必需列:{col}")
    if problems:
        return problems

    seen: set[tuple] = set()
    duplicates = 0
    for row in rows:
        key = (row.get("item_id"), row.get("annotator"))
        if key in seen:
            duplicates += 1
        seen.add(key)
    if duplicates:
        problems.append(f"有 {duplicates} 行 (item_id, annotator) 重复 —— "
                        f"同一题同一人只能标一次")

    missing = [r for r in rows if r.get("label") in (None, "")]
    if missing:
        items = sorted({r["item_id"] for r in missing}, key=str)
        problems.append(f"还有 {len(missing)} 行没有 label,涉及 {len(items)} 道题"
                        f"(前几道:{items[:5]})")

    # 每题被标注的人数是否一致 —— 不一致会让 Fleiss' Kappa 无法使用
    per_item: dict = {}
    for row in rows:
        per_item.setdefault(row["item_id"], set()).add(row["annotator"])
    counts = {len(v) for v in per_item.values()}
    if len(counts) > 1:
        problems.append(
            f"每道题的标注人数不一致(出现 {sorted(counts)} 种)—— "
            f"这会让 Fleiss' Kappa 无法计算,只能看逐题分歧")
    return problems


def to_csv(rows, path) -> Path:
    """写成 CSV。用 utf-8-sig —— 否则 Excel 打开中文会是乱码,
    而标注表十有八九是要用 Excel 给人填的。"""
    if not rows:
        raise ValueError("空表")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return out


def to_jsonl(rows, path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def from_jsonl(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
