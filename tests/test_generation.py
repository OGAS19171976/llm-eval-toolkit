"""生成层指标(可确定性判定的那部分)的测试。"""

from __future__ import annotations

import math

import pytest

from llm_eval_toolkit.metrics import generation as G


# ======================================================================
# 引用有效性
# ======================================================================
def test_citation_validity_hand_computed():
    assert G.citation_validity(["s1", "s2"], ["s1", "s2", "s3"]) == pytest.approx(1.0)
    assert G.citation_validity(["s1", "s9"], ["s1", "s2"]) == pytest.approx(0.5)
    assert G.citation_validity(["s9"], ["s1"]) == 0.0


def test_citation_validity_without_citations_is_nan():
    """"没引用"既不该被奖励也不该被惩罚 —— 它该由另一个指标(覆盖率)去说。

    返回 1.0 会让"什么都不引用"变成满分;返回 0.0 又会把它变成零分。
    两个都是错的。
    """
    assert math.isnan(G.citation_validity([], ["s1"]))


def test_citation_report_counts_and_lists_problems():
    cases = [
        {"id": "q1", "cited": ["s1"], "retrieved": ["s1"]},          # 好
        {"id": "q2", "cited": ["s9"], "retrieved": ["s1"]},          # 引用了不存在的
        {"id": "q3", "cited": [], "retrieved": ["s1"]},              # 没引用
    ]
    rep = G.citation_report(cases)
    assert rep.n_cases == 3
    assert rep.n_no_citation == 1
    assert rep.n_defined == 2
    assert rep.mean_validity == pytest.approx(0.5)
    assert [p["id"] for p in rep.problem_cases] == ["q2"]
    assert rep.problem_cases[0]["invalid"] == ["s9"]


def test_citation_report_is_honest_about_what_it_measures():
    """报告必须**自己**说清楚它测的不是"出处支持结论"。

    一个叫"引用准确率"的数字最容易被过度解读成"答案有据可查",
    所以这句话必须出现在输出里,而不是只写在文档里。
    """
    rep = G.citation_report([{"id": "q", "cited": ["s1"], "retrieved": ["s1"]}])
    text = "\n".join(rep.summary_lines())
    assert "出处存在" in text
    assert "支持" in text and "评委" in text


# ======================================================================
# 拒答
# ======================================================================
def test_refusal_rate_denominator_is_all_items():
    """分母是全部题,不是"作答题"。

    只算作答题会把"大部分都拒答"这种严重情况掩盖成一个漂亮的准确率。
    """
    assert G.refusal_rate([True, False, False, False]) == pytest.approx(0.25)
    assert G.refusal_rate([]) != G.refusal_rate([])          # nan
    assert G.refusal_rate([True, True]) == pytest.approx(1.0)


def test_looks_like_refusal_basic():
    assert G.looks_like_refusal("抱歉,资料中没有相关内容。")
    assert G.looks_like_refusal("I don't know the answer.")
    assert not G.looks_like_refusal("根据规范第 3.2 条,应当……")
    assert not G.looks_like_refusal("")


def test_refusal_detector_must_be_validated_before_use():
    """验证器本身要能给出"别用这个启发式"的结论。

    结构:预测和标注完全错开 → 一致率 0.5、Kappa 0 → 判定不可用。
    """
    predicted = [True, True, False, False]
    labels = [True, False, True, False]
    out = G.validate_refusal_detector(predicted, labels)
    assert out["tp"] == 1 and out["fp"] == 1 and out["fn"] == 1 and out["tn"] == 1
    assert out["accuracy"] == pytest.approx(0.5)
    assert out["kappa"] == pytest.approx(0.0)
    assert "不可用" in out["verdict"]


def test_refusal_detector_good_case():
    predicted = [True, True, False, False, False, False]
    labels = [True, True, False, False, False, False]
    out = G.validate_refusal_detector(predicted, labels)
    assert out["kappa"] == pytest.approx(1.0)
    assert "可用" in out["verdict"]


def test_refusal_detector_requires_equal_length():
    with pytest.raises(ValueError):
        G.validate_refusal_detector([True], [True, False])
