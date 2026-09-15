"""检索层指标:Hit@k / Recall@k / Precision@k / MRR / NDCG@k。

这些指标**完全由"排序列表 + 相关集"决定**,不需要任何模型评判。这一点很关键:
它们是整个评估里唯一能保证"换个人来算,结果完全一样"的部分。生成层那些
需要 LLM 或人工打分的指标做不到这一点 —— 所以两类必须分开存放、分开汇报。

本模块**只依赖标准库**。

关于"没有相关项"的处理(这是个容易出错的地方)
------------------------------------------------
一道题如果标注里根本没有相关项,那么 Recall@k、NDCG@k 在数学上**没有定义**,
不是 0。这里一律返回 `nan`,并且聚合时把 nan **排除**而不是当成 0:

  * 当成 0 → 一道标注缺失的题会把整体 NDCG 拉低,凭空制造"指标下降",
    而这和系统改动毫无关系;
  * 排除 → 同时报告被排除了多少题,读者知道这个数字覆盖了多少数据。

`RankingMetrics` 里的 `n_no_relevant` 就是为这件事存在的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "hit_at_k", "recall_at_k", "precision_at_k", "reciprocal_rank",
    "dcg_at_k", "ndcg_at_k", "average_precision",
    "RankingMetrics", "evaluate_rankings",
]


def _as_lists(ranked, relevant) -> tuple[list, set]:
    return list(ranked), set(relevant)


def hit_at_k(ranked, relevant, k: int) -> bool:
    """top-k 里有没有**至少一个**相关项。

    这是最常用也最宽容的指标:它只问"能不能找到",不问"排得好不好"。
    报 Hit@k 时必须同时报 k 和它的业务依据 —— "上下文预算只够放 5 条"
    这种理由,而不是"我试了几个觉得 5 不错"。
    """
    if k <= 0:
        raise ValueError("k 必须为正")
    items, rel = _as_lists(ranked, relevant)
    return bool(set(items[:k]) & rel)


def recall_at_k(ranked, relevant, k: int) -> float:
    """top-k 覆盖了多大比例的相关项。没有相关项时返回 nan(不是 0)。"""
    if k <= 0:
        raise ValueError("k 必须为正")
    items, rel = _as_lists(ranked, relevant)
    if not rel:
        return float("nan")
    return len(set(items[:k]) & rel) / len(rel)


def precision_at_k(ranked, relevant, k: int) -> float:
    """top-k 里有多大比例真的相关。

    注意分母是 **k**,不是 min(k, 列表长度) —— 检索只返回 3 条却要 top-5 时,
    缺的那两条按"不相关"计。否则一个只返回 1 条且命中的系统会得到 100%,
    这显然不是我们想奖励的行为。
    """
    if k <= 0:
        raise ValueError("k 必须为正")
    items, rel = _as_lists(ranked, relevant)
    if not rel:
        return float("nan")
    return len(set(items[:k]) & rel) / k


def reciprocal_rank(ranked, relevant) -> float:
    """第一个相关项的排名倒数(1/rank);没有相关项时为 0。

    MRR 就是它在一批题上的均值。它和 Hit@k 的区别在于**在乎排第几**:
    相关项排在第 1 位和第 5 位,Hit@5 一样都是 1,RR 分别是 1.0 和 0.2。

    为什么没有相关项时返回 0 而不是 nan:第 1 名的位置永远存在,
    "没有相关项"在这道题上就是一种确定的不理想结果,不是"无法定义"。
    """
    items, rel = _as_lists(ranked, relevant)
    for i, item in enumerate(items, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranked, relevant) -> float:
    """AP:每个相关项位置上的 Precision 的平均。

    比 Recall 更严格 —— 它同时惩罚"漏掉"和"排在后面"。
    """
    items, rel = _as_lists(ranked, relevant)
    if not rel:
        return float("nan")
    hits = 0
    total = 0.0
    for i, item in enumerate(items, start=1):
        if item in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def _dcg(gains: list[float]) -> float:
    """折损累积增益:gain / log2(rank + 1),rank 从 1 开始。

    增益用 2^r - 1(相关度等级 r,二值标注时 r ∈ {0,1}) —— 这是 NDCG 的
    标准形式,好处是相关度等级越高增益增长越快,而不是线性。
    """
    return sum((2.0 ** g - 1.0) / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def dcg_at_k(ranked, gains: dict, k: int | None = None) -> float:
    items = list(ranked)
    if k is not None:
        items = items[:k]
    return _dcg([float(gains.get(x, 0.0)) for x in items])


def ndcg_at_k(ranked, relevant, k: int | None = None, gains: dict | None = None) -> float:
    """NDCG@k:实际折损增益 / 理想折损增益。

    `gains` 给分级相关度(例如 2=完全相关、1=部分相关、0=不相关);
    不给就按二值处理,`relevant` 里的算 1。

    没有相关项时返回 nan —— 因为理想折损增益为 0,比值没有定义。
    """
    items, rel = _as_lists(ranked, relevant)
    if k is not None:
        items = items[:k]
    if gains is None:
        actual = [1.0 if x in rel else 0.0 for x in items]
        ideal = [1.0] * min(len(rel), len(items))
    else:
        actual = [float(gains.get(x, 0.0)) for x in items]
        ideal = sorted((float(gains.get(x, 0.0)) for x in rel), reverse=True)[:len(items)]
    idcg = _dcg(ideal)
    if idcg == 0:
        return float("nan")
    return _dcg(actual) / idcg


# ======================================================================
# 批量
# ======================================================================
@dataclass
class RankingMetrics:
    """一批题上的检索指标。

    `per_case_hit[k]` 特意保留成**逐题的布尔列表**,而不是只给均值:
    上层要用它算 Wilson 区间,更要用它做配对检验(McNemar 需要同一批题上
    两个系统的逐题结果)。只留下均值,等于把最贵的那部分信息丢了。

    它与 `defined_ids` **一一对应** —— 没有相关项标注的题已经被排除,
    不在这个列表里。这一点必须记住,否则配对时会把两边的题错位对齐。
    """

    n: int
    n_no_relevant: int
    defined_ids: list = field(default_factory=list)
    hit: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float = float("nan")
    map_score: float = float("nan")
    per_case_hit: dict[int, list[bool]] = field(default_factory=dict)

    @property
    def n_defined(self) -> int:
        """参与了比例类指标的题数(排除了没有相关项的题)。"""
        return self.n - self.n_no_relevant

    def summary_lines(self) -> list[str]:
        out = [f"题数 {self.n}(其中 {self.n_no_relevant} 道无相关项标注,已从比例类指标中排除)"]
        for k in sorted(self.hit):
            out.append(f"  Hit@{k:<3} {self.hit[k]:.1%}    "
                       f"Recall@{k:<3} {_fmt(self.recall[k])}    "
                       f"Precision@{k:<3} {_fmt(self.precision[k])}    "
                       f"NDCG@{k:<3} {_fmt(self.ndcg[k])}")
        out.append(f"  MRR {self.mrr:.4f}    MAP {_fmt(self.map_score)}")
        return out


def _fmt(x: float) -> str:
    return "  nan " if x != x else f"{x:.1%}"


def _nanmean(values: list[float]) -> float:
    ok = [v for v in values if v == v]          # 排除 nan
    return sum(ok) / len(ok) if ok else float("nan")


def evaluate_rankings(cases, ks=(1, 3, 5, 10)) -> RankingMetrics:
    """`cases` 是若干 `{"ranked": [...], "relevant": [...], "gains": {...}}`。

    两遍扫描:第一遍收集逐题结果,第二遍聚合。这样"排除没有相关项的题"这件事
    只在一处发生,不会散落到每个指标的实现里。

    **Hit@k 也一起排除**,这一点值得单独说:一道题如果没有相关项标注,
    "有没有命中"同样是没有定义的。把它算成"没命中"等于凭空扣分 ——
    标注缺失从来不是系统的问题。
    """
    cases = list(cases)
    if not cases:
        return RankingMetrics(n=0, n_no_relevant=0)

    ks = sorted(set(ks))
    defined_ids: list = []
    hit_vals: dict[int, list[float]] = {k: [] for k in ks}
    hit_bools: dict[int, list[bool]] = {k: [] for k in ks}
    recall_vals: dict[int, list[float]] = {k: [] for k in ks}
    prec_vals: dict[int, list[float]] = {k: [] for k in ks}
    ndcg_vals: dict[int, list[float]] = {k: [] for k in ks}
    rr_vals: list[float] = []
    ap_vals: list[float] = []
    n_no_rel = 0

    for index, case in enumerate(cases):
        ranked = case["ranked"]
        relevant = case.get("relevant") or []
        gains = case.get("gains")
        if not relevant:
            n_no_rel += 1
            continue
        defined_ids.append(case["id"] if "id" in case else index)
        rr_vals.append(reciprocal_rank(ranked, relevant))
        ap_vals.append(average_precision(ranked, relevant))
        for k in ks:
            h = hit_at_k(ranked, relevant, k)
            hit_bools[k].append(h)
            hit_vals[k].append(1.0 if h else 0.0)
            recall_vals[k].append(recall_at_k(ranked, relevant, k))
            prec_vals[k].append(precision_at_k(ranked, relevant, k))
            ndcg_vals[k].append(ndcg_at_k(ranked, relevant, k, gains))

    return RankingMetrics(
        n=len(cases),
        n_no_relevant=n_no_rel,
        defined_ids=defined_ids,
        hit={k: _nanmean(v) for k, v in hit_vals.items()},
        recall={k: _nanmean(v) for k, v in recall_vals.items()},
        precision={k: _nanmean(v) for k, v in prec_vals.items()},
        ndcg={k: _nanmean(v) for k, v in ndcg_vals.items()},
        mrr=_nanmean(rr_vals),
        map_score=_nanmean(ap_vals),
        per_case_hit=hit_bools,
    )
