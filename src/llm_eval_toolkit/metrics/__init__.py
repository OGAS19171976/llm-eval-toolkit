"""指标计算。

分两层,对应 LLM 应用里两类完全不同的东西:

retrieval
    检索层。完全由"排序列表 + 相关集"决定,**可确定性计算**,不需要模型评判。
    Hit@k / Recall@k / Precision@k / MRR / NDCG@k。

generation
    生成层。**这里只放能确定性判定的部分** —— 引用是否真的指向被检索到的来源、
    系统有没有拒答。忠实度(faithfulness)、答案相关性这类需要 LLM 或人工评判的
    指标不在这里,因为它们的结果取决于评委,而评委本身要先被评估
    (见 build_evalset.agreement 与 stats.cohen_kappa)。

把这两类混在一起是评测代码最常见的问题:一个指标到底"谁说了算"看不清,
于是也没法判断它可不可信。
"""

from __future__ import annotations

from . import retrieval   # noqa: F401

__all__ = ["retrieval"]
