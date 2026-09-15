"""评估集**设计**。

这个模块回答的是评估开始**之前**就该回答的问题:

  * 要检出 5% 的提升,需要多少道题?(planning)
  * 这些题该怎么抽,才能让指标的含义稳定? (抽样在 stats.stratified_sample)
  * 标注表长什么样、怎么校验? (template)
  * 标注者之间到底一致到什么程度,Kappa 够不够用? (agreement)

为什么值得单独一层:评估集本身的质量决定了后面所有结论的上限。
用 56 道题去下"新版本更好"的结论,和用 400 道题下同一个结论,
是两件可信度完全不同的事 —— 而这个差别在报告里往往一个字都不写。
"""

from __future__ import annotations

from . import agreement, planning, template   # noqa: F401

__all__ = ["planning", "template", "agreement"]
