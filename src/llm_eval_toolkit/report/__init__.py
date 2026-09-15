"""报告:把两份评估结果做成带显著性标注的对比。

compare
    核心。按题目 id 配对,逐指标给:点估计、Wilson 区间、配对 Bootstrap 区间、
    McNemar 检验、Holm / BH 校正后的 p,以及一句结论。
view
    把报告里任意一个逐题指标"顶"成主指标,这样才能用同一套配对检验去比
    不同策略(例如"只看 top-1" vs "放宽到 top-3")。

`compare.analyze_pair` 是纯函数(输入是内存里的 records),不碰文件系统,
所以可以被直接单元测试 —— 命令行入口只是它外面薄薄一层。
"""

from __future__ import annotations

from . import compare, view   # noqa: F401

__all__ = ["compare", "view"]
