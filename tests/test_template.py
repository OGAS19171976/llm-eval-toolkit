"""标注表生成与校验的测试。"""

from __future__ import annotations

import json

import pytest

from llm_eval_toolkit.build_evalset import template as T

ITEMS = [{"id": "q1", "latex": "x^2"}, {"id": "q2", "latex": "x^3"}]


def test_sheet_expands_items_by_annotators():
    rows = T.build_annotation_sheet(ITEMS, ["我", "同事"])
    assert len(rows) == 4
    assert [r["item_id"] for r in rows] == ["q1", "q1", "q2", "q2"]
    assert [r["annotator"] for r in rows] == ["我", "同事", "我", "同事"]
    assert all(r["label"] is None for r in rows)


def test_sheet_requires_at_least_two_annotators():
    """单人标注算不出一致性 —— "一致性"这个概念在单标注者下根本不存在。"""
    with pytest.raises(ValueError) as excinfo:
        T.build_annotation_sheet(ITEMS, ["只有我"])
    assert "至少需要 2 名" in str(excinfo.value)


def test_sheet_can_carry_context_columns():
    """把题干带进表里 —— 否则标注的人得对着 id 去另一个文件里查。"""
    rows = T.build_annotation_sheet(ITEMS, ["A", "B"], keep=("latex",))
    assert rows[0]["latex"] == "x^2"
    assert "id" not in rows[0]          # item_key 变成了 item_id


def test_sheet_requires_the_item_key():
    with pytest.raises(KeyError):
        T.build_annotation_sheet([{"nope": 1}], ["A", "B"])


# ======================================================================
# 校验
# ======================================================================
def test_validate_accepts_a_complete_sheet():
    rows = T.build_annotation_sheet(ITEMS, ["A", "B"])
    for r in rows:
        r["label"] = "通过"
    assert T.validate_annotation_sheet(rows) == []


def test_validate_reports_missing_labels_without_raising():
    """标了一半的表是**正常的工作状态**。

    校验应该能随时跑、只报还差多少,而不是一跑就抛异常 ——
    那样就没法用它跟踪标注进度了。
    """
    rows = T.build_annotation_sheet(ITEMS, ["A", "B"])
    rows[0]["label"] = "通过"
    problems = T.validate_annotation_sheet(rows)
    assert any("3 行没有 label" in p for p in problems)
    assert any("q1" in p for p in problems)


def test_validate_detects_duplicate_annotation():
    rows = T.build_annotation_sheet(ITEMS, ["A", "B"])
    for r in rows:
        r["label"] = "通过"
    rows.append(dict(rows[0]))          # 同一题同一人标了两次
    problems = T.validate_annotation_sheet(rows)
    assert any("重复" in p for p in problems)


def test_validate_detects_unequal_rater_counts():
    """每题标注人数不等 → Fleiss' Kappa 用不了,必须提前提醒。

    这个问题的可怕之处在于它不会报错:矩阵算出来是个数,只是那个数没有意义。
    """
    rows = T.build_annotation_sheet(ITEMS, ["A", "B"])
    for r in rows:
        r["label"] = "通过"
    rows = [r for r in rows if not (r["item_id"] == "q2" and r["annotator"] == "B")]
    problems = T.validate_annotation_sheet(rows)
    assert any("标注人数不一致" in p for p in problems)


def test_validate_handles_empty_and_missing_columns():
    assert T.validate_annotation_sheet([]) == ["标注表是空的"]
    problems = T.validate_annotation_sheet([{"foo": 1}])
    assert any("缺少必需列" in p for p in problems)


# ======================================================================
# 落盘
# ======================================================================
def test_to_csv_uses_utf8_sig_so_excel_does_not_mangle_chinese(workdir):
    rows = T.build_annotation_sheet(ITEMS, ["我", "同事"])
    path = T.to_csv(rows, workdir / "sheet.csv")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "缺 BOM,Excel 打开中文会是乱码"
    text = raw.decode("utf-8-sig")
    assert "item_id,annotator,label,confidence,note" in text
    assert "我" in text


def test_jsonl_round_trip(workdir):
    rows = T.build_annotation_sheet(ITEMS, ["A", "B"])
    rows[0]["label"] = "通过"
    path = T.to_jsonl(rows, workdir / "sub" / "sheet.jsonl")
    assert T.from_jsonl(path) == rows


def test_to_csv_creates_parent_directories(workdir):
    rows = T.build_annotation_sheet(ITEMS, ["A", "B"])
    path = T.to_csv(rows, workdir / "deep" / "nest" / "sheet.csv")
    assert path.exists()
