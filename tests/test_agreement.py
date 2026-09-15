"""标注一致性的测试。

核心要证明的是那句反复出现的话:**一致率高不等于标注可信**。
所以最有价值的用例是 `test_high_agreement_low_kappa_trap`。
"""

from __future__ import annotations

import math

import pytest

from llm_eval_toolkit.build_evalset import agreement as A


def rows_two_raters():
    """手算基准:3 题、2 人。
       i1 A=P B=P(一致)  i2 A=P B=F(分歧)  i3 A=F B=F(一致)
       一致率 = 2/3
       Cohen's Kappa:po = 2/3,pe = (2/3)(1/3)+(1/3)(2/3) = 4/9
                      κ = (2/3 - 4/9) / (1 - 4/9) = 0.4
    """
    return [
        {"item_id": "i1", "annotator": "A", "label": "P"},
        {"item_id": "i1", "annotator": "B", "label": "P"},
        {"item_id": "i2", "annotator": "A", "label": "P"},
        {"item_id": "i2", "annotator": "B", "label": "F"},
        {"item_id": "i3", "annotator": "A", "label": "F"},
        {"item_id": "i3", "annotator": "B", "label": "F"},
    ]


# ======================================================================
# 基础
# ======================================================================
def test_pivot_drops_unlabelled_entries():
    """没标的行不该变成"标注为 None" —— 那是漏标,不是一种标签。"""
    rows = rows_two_raters() + [{"item_id": "i9", "annotator": "A", "label": None}]
    table = A.pivot(rows)
    assert "i9" not in table
    assert table["i1"] == {"A": "P", "B": "P"}


def test_raw_agreement_hand_computed():
    assert A.raw_agreement(A.pivot(rows_two_raters())) == pytest.approx(2 / 3)


def test_raw_agreement_is_nan_without_any_pair():
    rows = [{"item_id": "i1", "annotator": "A", "label": "P"}]
    assert math.isnan(A.raw_agreement(A.pivot(rows)))


def test_raw_agreement_generalises_to_three_raters():
    """3 人、标签 P/P/F:两两配对共 3 对,同意的只有 (A,B) 那一对 → 1/3。

    注意这里**不**用"出现过半数的标签"当基准 —— 那会引入"谁是多数"这个
    额外假设,人一多还会出现没有多数的情况。配对数对任意人数都成立。
    """
    rows = [
        {"item_id": "i1", "annotator": "A", "label": "P"},
        {"item_id": "i1", "annotator": "B", "label": "P"},
        {"item_id": "i1", "annotator": "C", "label": "F"},
    ]
    assert A.raw_agreement(A.pivot(rows)) == pytest.approx(1 / 3)


# ======================================================================
# 报告
# ======================================================================
def test_report_uses_cohen_kappa_for_two_raters():
    rep = A.agreement_report(rows_two_raters())
    assert rep["kappa_method"] == "Cohen's Kappa"
    assert rep["raw_agreement"] == pytest.approx(2 / 3)
    assert rep["kappa"] == pytest.approx(0.4)
    assert rep["n_items"] == 3 and rep["n_raters"] == 2
    assert rep["n_disagreements"] == 1
    assert rep["notes"], "报告必须给出结论,不能只给数字"


def test_high_agreement_low_kappa_trap():
    """**这个测试是整个模块存在的理由。**

    20 道题,A 全判"通过",B 只在 2 道题上判"不通过"。
    原始一致率 90%,看着很漂亮;Kappa 却精确等于 **0** ——
    也就是两个人的判断之间**一点关系都没有**。

    只看一致率就会得出"标注质量良好"的错误结论。
    """
    rows = []
    for i in range(20):
        rows.append({"item_id": f"i{i}", "annotator": "A", "label": "通过"})
    for i in range(18):
        rows.append({"item_id": f"i{i}", "annotator": "B", "label": "通过"})
    for i in range(18, 20):
        rows.append({"item_id": f"i{i}", "annotator": "B", "label": "不通过"})

    rep = A.agreement_report(rows)
    assert rep["raw_agreement"] == pytest.approx(0.9)
    assert rep["kappa"] == pytest.approx(0.0, abs=1e-12)
    assert rep["dominant_share"] > 0.9
    # 报告必须**自己**把这件事说出来,而不是等着读者去算
    assert any("虚高" in n for n in rep["notes"])
    assert any("规范" in n for n in rep["notes"])


def test_report_cannot_compute_kappa_when_raters_are_uneven():
    rows = [
        {"item_id": "i1", "annotator": "A", "label": "P"},
        {"item_id": "i1", "annotator": "B", "label": "P"},
        {"item_id": "i2", "annotator": "A", "label": "P"},
    ]
    rep = A.agreement_report(rows)
    assert math.isnan(rep["kappa"])
    assert rep["kappa_method"] == "无法计算"
    assert any("补齐" in n for n in rep["notes"])


def test_report_uses_fleiss_for_three_raters():
    """3 人、2 类,手算 κ = 0.55。

    i1 [3,0]  i2 [0,3]  i3 [2,1]
    P_i = 1, 1, 1/3 → P̄ = 7/9
    p_P = 5/9, p_F = 4/9 → Pe = 41/81
    κ = (7/9 - 41/81) / (1 - 41/81) = 22/40 = 0.55
    """
    rows = []
    data = {
        "i1": {"A": "P", "B": "P", "C": "P"},
        "i2": {"A": "F", "B": "F", "C": "F"},
        "i3": {"A": "P", "B": "P", "C": "F"},
    }
    for item, labels in data.items():
        for who, label in labels.items():
            rows.append({"item_id": item, "annotator": who, "label": label})
    rep = A.agreement_report(rows)
    assert "Fleiss" in rep["kappa_method"]
    assert rep["kappa"] == pytest.approx(0.55)


def test_perfect_agreement_is_one():
    rows = [
        {"item_id": "i1", "annotator": "A", "label": "P"},
        {"item_id": "i1", "annotator": "B", "label": "P"},
        {"item_id": "i2", "annotator": "A", "label": "F"},
        {"item_id": "i2", "annotator": "B", "label": "F"},
    ]
    rep = A.agreement_report(rows)
    assert rep["raw_agreement"] == pytest.approx(1.0)
    assert rep["kappa"] == pytest.approx(1.0)
    assert rep["n_disagreements"] == 0
    assert any("可以接受" in n for n in rep["notes"])


# ======================================================================
# 分歧清单
# ======================================================================
def test_disagreements_lists_exactly_the_conflicting_items():
    found = A.disagreements(rows_two_raters())
    assert len(found) == 1
    assert found[0]["item_id"] == "i2"
    assert found[0]["labels"] == {"A": "P", "B": "F"}


def test_disagreements_is_sorted_and_empty_when_clean():
    assert A.disagreements([
        {"item_id": "i1", "annotator": "A", "label": "P"},
        {"item_id": "i1", "annotator": "B", "label": "P"},
    ]) == []


def test_render_report_contains_the_key_numbers():
    text = A.render_report(A.agreement_report(rows_two_raters()))
    assert "原始一致率" in text
    assert "Cohen" in text
    assert "分歧题 1 道" in text
