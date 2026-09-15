"""llm-eval-toolkit:把「点估计」升级成「带不确定性的结论」。

它只回答一个问题,但这个问题决定其它所有事:

    **我们怎么知道它真的变好了?**

大多数团队的答案是"看几个样例"或者"指标从 0.72 涨到 0.75"。这个库要做的是
把这种说法换成能被追问得住的说法:带置信区间的估计、配对显著性检验、
多重比较校正,以及一句"在现有样本量下你能检出的最小差异是多少"。

四个模块
--------
stats
    统计基础。Wilson 区间、Bootstrap(BCa)、McNemar、Cochran's Q、
    Holm / BH 多重比较校正、Cohen's & Fleiss' Kappa、ECE、分层估计。
metrics
    指标计算。检索层 Hit@k / Recall@k / MRR / NDCG@k;生成层里**可以确定性
    判定**的那部分(引用是否指向真实存在的来源、拒答率)。
build_evalset
    评估集**设计**。要多少题才检得出目标差异、按什么分层抽、标注表长什么样、
    标注者之间一致性够不够。
report
    把两份评估结果做配对比较,输出带置信区间与显著性标注的报告。

设计约束
--------
**除 numpy 外零运行时依赖。** 这不是洁癖:

* 评估基础设施最不该出现的情况就是"装不上包所以跑不了";
* 没有 scipy 这一条还有个附带好处 —— 所有公式都在明处,评审者能读;
* scipy 只在**测试**里出现,当独立裁判做交叉验证(见 tests/test_stats.py)。

模块间不许有循环依赖:`stats` 是叶子,`build_evalset` 和 `report` 依赖它,
`metrics` 只依赖标准库。
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import build_evalset, metrics, report, stats   # noqa: F401

__all__ = ["stats", "metrics", "build_evalset", "report", "__version__"]
