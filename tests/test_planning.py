"""评估集规模规划的测试。

这一层的价值全在"给出的数字对不对、以及不可行时会不会老实说不可行"上,
所以两类都测:数值单调性,以及**拒绝编造数字**。
"""

from __future__ import annotations

import math

import pytest

from llm_eval_toolkit.build_evalset import planning as P


def test_required_pairs_delegates_to_mcnemar():
    # π01=0.05, π10=0.15 → 净差异 0.10
    n = P.required_pairs(0.05, 0.15)
    assert n > 0
    # 反向退化越多,需要越多样本(净差异不变)
    assert P.required_pairs(0.20, 0.30) > n


def test_required_pairs_zero_difference_returns_minus_one():
    """净差异为 0 时再多样本也检不出,必须返回 -1 而不是一个假的有限值。"""
    assert P.required_pairs(0.10, 0.10) == -1


# ======================================================================
# 可行性
# ======================================================================
def test_feasibility_rejects_improvement_beyond_available_headroom():
    """基线 90% 的准确率,不可能有 20% 的题从错变对 —— 错的题一共才 10%。"""
    ok, reason = P.feasibility(delta=0.20, baseline=0.90, churn_ratio=0.0)
    assert not ok
    assert "不可能" in reason and "10.0%" in reason


def test_feasibility_rejects_regression_beyond_available_correct():
    ok, reason = P.feasibility(delta=0.05, baseline=0.05, churn_ratio=2.0)
    assert not ok
    assert "从对变错" in reason


def test_feasibility_accepts_reasonable_case():
    ok, reason = P.feasibility(delta=0.05, baseline=0.5, churn_ratio=0.5)
    assert ok and reason == ""


def test_feasibility_rejects_nonpositive_delta():
    ok, reason = P.feasibility(delta=0.0, baseline=0.5, churn_ratio=0.0)
    assert not ok and "为正" in reason


# ======================================================================
# 样本量表
# ======================================================================
def test_sample_size_grows_with_churn():
    """**这条是这一层最重要的结论。**

    r 越大(改好的同时改坏的越多),需要越多样本 —— 因为净差异虽然不变,
    分歧对却变多了,差异更难从噪声里分辨。

    直觉上"改好和改坏一样多"听起来是扯平了,实际上它对样本量的要求最高。
    """
    rows = P.sample_size_table(delta=0.05, baseline=0.5,
                               churn_ratios=(0.0, 0.5, 1.0))
    ns = [r["n_pairs"] for r in rows]
    assert all(n is not None for n in ns)
    assert ns == sorted(ns), f"样本量应随退化比单调不减,实得 {ns}"
    assert ns[0] < ns[-1]


def test_sample_size_exposes_discordant_rate():
    rows = P.sample_size_table(delta=0.05, baseline=0.5, churn_ratios=(0.0, 1.0))
    assert rows[0]["discordant_rate"] == pytest.approx(0.05)    # π01+π10 = d
    assert rows[1]["discordant_rate"] == pytest.approx(0.15)    # d(1+2r)
    assert rows[1]["pi10_improve"] == pytest.approx(0.10)
    assert rows[1]["pi01_regress"] == pytest.approx(0.05)


def test_infeasible_rows_get_none_not_a_made_up_number():
    """不可行的组合必须老实返回 None + 原因。

    编一个数字出来是最坏的选择:它会让人以为"多标 300 题就能检出 20% 提升",
    而这件事在数学上根本不可能发生。
    """
    rows = P.sample_size_table(delta=0.20, baseline=0.90, churn_ratios=(0.0, 1.0))
    assert all(not r["feasible"] for r in rows)
    assert all(r["n_pairs"] is None for r in rows)
    assert all(r["reason"] for r in rows)


def test_render_table_marks_infeasible_rows():
    rows = P.sample_size_table(delta=0.20, baseline=0.90, churn_ratios=(0.0,))
    text = P.render_table(0.20, 0.90, rows)
    assert "不可行" in text
    assert "不可能" in text
    assert "目标" in text


def test_render_table_contains_numbers_for_feasible_rows():
    rows = P.sample_size_table(delta=0.05, baseline=0.5, churn_ratios=(0.0,))
    text = P.render_table(0.05, 0.5, rows)
    assert str(rows[0]["n_pairs"]) in text
    assert "退化比" in text


def test_sample_size_is_larger_for_smaller_delta():
    small = P.sample_size_table(delta=0.02, baseline=0.5, churn_ratios=(0.0,))[0]
    large = P.sample_size_table(delta=0.10, baseline=0.5, churn_ratios=(0.0,))[0]
    assert small["n_pairs"] > large["n_pairs"]
