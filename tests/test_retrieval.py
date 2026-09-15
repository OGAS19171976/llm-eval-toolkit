"""检索层指标的测试。

全部用手算得出的小例子。检索指标的好处是**答案唯一** ——
只要定义说清楚,谁算都一样,所以这里可以逐个数字对齐。
"""

from __future__ import annotations

import math

import pytest

from llm_eval_toolkit.metrics import retrieval as R

# ranked = a, b, c, d;relevant = {b, d}
RANKED = ["a", "b", "c", "d"]
RELEVANT = {"b", "d"}
LOG2_3 = math.log2(3)
LOG2_5 = math.log2(5)


def test_hit_at_k():
    assert R.hit_at_k(RANKED, RELEVANT, 1) is False
    assert R.hit_at_k(RANKED, RELEVANT, 2) is True
    assert R.hit_at_k(RANKED, RELEVANT, 4) is True
    assert R.hit_at_k(RANKED, set(), 4) is False


def test_hit_at_k_rejects_bad_k():
    with pytest.raises(ValueError):
        R.hit_at_k(RANKED, RELEVANT, 0)


def test_recall_at_k_hand_computed():
    assert R.recall_at_k(RANKED, RELEVANT, 1) == 0.0
    assert R.recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)
    assert R.recall_at_k(RANKED, RELEVANT, 4) == pytest.approx(1.0)


def test_recall_is_nan_when_nothing_is_relevant():
    """没有相关项时 Recall 没有定义,必须返回 nan 而不是 0。

    返回 0 会把"这道题标注缺失"变成"系统漏掉了它",凭空制造指标下降。
    """
    assert math.isnan(R.recall_at_k(RANKED, set(), 3))
    assert math.isnan(R.precision_at_k(RANKED, set(), 3))
    assert math.isnan(R.ndcg_at_k(RANKED, set(), 3))


def test_precision_denominator_is_k_not_list_length():
    """分母是 k,不是列表长度。

    否则"只返回 1 条且命中"会拿到 100%,而这显然不是我们想奖励的行为。
    """
    assert R.precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)
    assert R.precision_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)


def test_reciprocal_rank():
    assert R.reciprocal_rank(RANKED, RELEVANT) == pytest.approx(0.5)   # b 在第 2 位
    assert R.reciprocal_rank(["b", "a"], RELEVANT) == pytest.approx(1.0)
    assert R.reciprocal_rank(["c"], RELEVANT) == 0.0
    # 没有相关项时是 0,不是 nan —— 第 1 名的位置永远存在
    assert R.reciprocal_rank(RANKED, set()) == 0.0


def test_average_precision_hand_computed():
    # 相关项出现在第 2、4 位 → (1/2 + 2/4) / 2 = 0.5
    assert R.average_precision(RANKED, RELEVANT) == pytest.approx(0.5)
    assert math.isnan(R.average_precision(RANKED, set()))


def test_ndcg_hand_computed():
    """DCG = Σ (2^rel - 1) / log2(rank + 1)。

    actual gains [0,1,0,1]:DCG  = 1/log2(3) + 1/log2(5)
    ideal  gains [1,1]:    IDCG = 1/log2(2) + 1/log2(3)
    """
    dcg = 1 / LOG2_3 + 1 / LOG2_5
    idcg = 1.0 + 1 / LOG2_3
    assert R.ndcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(dcg / idcg)
    assert R.ndcg_at_k(RANKED, RELEVANT, 2) == pytest.approx((1 / LOG2_3) / idcg)


def test_ndcg_perfect_ranking_is_one():
    assert R.ndcg_at_k(["b", "d", "a", "c"], RELEVANT, 4) == pytest.approx(1.0)


def test_ndcg_supports_graded_relevance():
    """分级相关度:gain 用 2^r - 1,所以"高度相关"比"部分相关"值钱得多。"""
    gains = {"a": 2.0, "b": 1.0, "c": 0.0}
    # [a, b] 优于 [b, a]
    assert (R.ndcg_at_k(["a", "b"], {"a", "b"}, 2, gains)
            > R.ndcg_at_k(["b", "a"], {"a", "b"}, 2, gains))
    assert R.ndcg_at_k(["a", "b"], {"a", "b"}, 2, gains) == pytest.approx(1.0)


def test_dcg_discounts_by_rank():
    # 第 1 位的增益不打折
    assert R.dcg_at_k(["a"], {"a": 1.0}) == pytest.approx(1.0)
    assert R.dcg_at_k(["x", "a"], {"a": 1.0}) == pytest.approx(1 / LOG2_3)


# ======================================================================
# 批量
# ======================================================================
def _case(ranked, relevant, cid="x", gains=None):
    c = {"id": cid, "ranked": ranked, "relevant": relevant}
    if gains:
        c["gains"] = gains
    return c


def test_evaluate_rankings_aggregates_and_keeps_per_case_flags():
    cases = [
        _case(["a", "b"], {"b"}, "q1"),      # hit@1 假,hit@2 真
        _case(["b", "a"], {"b"}, "q2"),      # hit@1 真
        _case(["c"], {"b"}, "q3"),           # 都没命中
    ]
    m = R.evaluate_rankings(cases, ks=(1, 2))
    assert m.n == 3 and m.n_no_relevant == 0
    assert m.hit[1] == pytest.approx(1 / 3)
    assert m.hit[2] == pytest.approx(2 / 3)
    # 逐题布尔被保留下来,上层才能算区间、才能做配对检验
    assert m.per_case_hit[1] == [False, True, False]
    assert m.per_case_hit[2] == [True, True, False]


def test_evaluate_rankings_excludes_cases_without_relevant_items():
    """标注缺失的题必须被**排除**,而不是当成 0 分。

    否则 3 道题里 1 道没标注,整体指标就被拉低 1/3 ——
    而这和系统改动毫无关系。

    **Hit@k 也一起排除**:一道题没有相关项标注,"有没有命中"同样没有定义。
    算成"没命中"等于凭空扣分。
    """
    cases = [
        _case(["a"], {"a"}, "q1"),
        _case(["a"], {"a"}, "q2"),
        _case(["a"], set(), "q3"),          # 没有相关项标注
    ]
    m = R.evaluate_rankings(cases, ks=(1,))
    assert m.n == 3
    assert m.n_no_relevant == 1
    assert m.n_defined == 2
    assert m.recall[1] == pytest.approx(1.0)
    assert m.hit[1] == pytest.approx(1.0)
    # 逐题结果与 defined_ids 一一对应 —— 配对检验时必须靠它对齐
    assert m.defined_ids == ["q1", "q2"]
    assert m.per_case_hit[1] == [True, True]
    assert len(m.per_case_hit[1]) == len(m.defined_ids)


def test_case_without_id_falls_back_to_index():
    m = R.evaluate_rankings([{"ranked": ["a"], "relevant": {"a"}}], ks=(1,))
    assert m.defined_ids == [0]


def test_evaluate_rankings_empty():
    m = R.evaluate_rankings([])
    assert m.n == 0 and m.hit == {}


def test_summary_lines_marks_nan_as_nan():
    m = R.evaluate_rankings([_case(["a"], set(), "q1")], ks=(1,))
    text = "\n".join(m.summary_lines())
    assert "nan" in text
    assert "无相关项" in text
