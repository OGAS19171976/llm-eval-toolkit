"""评估集设计:要多少题,才检得出你想检出的差异?

这个模块存在的直接原因,是评估里最常见、也最容易答砸的一个问题:

    **「要多少样本才能检出 5% 的提升?」**

不算这个,你只能回答"越多越好"。算了,你才能说:"在 5% 的净提升下,
如果两版之间几乎没有反向退化,需要 400 道题;如果每改好 1 道就改坏 1 道,
需要 800 道 —— 而在现有 56 道的规模下,这个差异根本测不出来。"

关键洞察:决定样本量的**不是总题量,而是分歧率**。两版表现一模一样的题,
无论加多少,都不会让检验更有判别力(见 `stats.minimum_detectable_delta`)。
"""

from __future__ import annotations

from .. import stats as S

__all__ = ["sample_size_table", "render_table", "required_pairs", "feasibility"]


def required_pairs(pi01: float, pi10: float, alpha: float = 0.05,
                   power: float = 0.8) -> int:
    """`stats.mcnemar_sample_size` 的直白封装,用评估里的术语命名。

    pi01 —— 改前对、改后错的比例(**反向退化**)
    pi10 —— 改前错、改后对的比例(**净改善**那一侧)
    """
    return S.mcnemar_sample_size(pi01, pi10, alpha=alpha, power=power)


def feasibility(delta: float, baseline: float, churn_ratio: float) -> tuple[bool, str]:
    """这个 (提升幅度, 基线, 退化比) 组合在数学上是否可能。

    pi01 = r·d,pi10 = d(1+r)。两个都是比例,所以必须:
        pi10 <= 1 - baseline   (能改好的题,最多只有当前错的那部分)
        pi01 <= baseline       (能改坏的题,最多只有当前对的那部分)
    """
    if delta <= 0:
        return False, "目标提升必须为正"
    pi01 = churn_ratio * delta
    pi10 = delta * (1 + churn_ratio)
    if pi10 > 1 - baseline + 1e-12:
        return False, (f"改前错的比例只有 {1 - baseline:.1%},"
                       f"不可能有 {pi10:.1%} 的题从错变对")
    if pi01 > baseline + 1e-12:
        return False, (f"改前对的比例只有 {baseline:.1%},"
                       f"不可能有 {pi01:.1%} 的题从对变错")
    return True, ""


def sample_size_table(
    delta: float,
    baseline: float = 0.5,
    churn_ratios=(0.0, 0.25, 0.5, 1.0),
    alpha: float = 0.05,
    power: float = 0.8,
) -> list[dict]:
    """在不同"反向退化"程度下,检出 delta 需要多少配对样本。

    `churn_ratios` 里的 r 表示:每拿到 1 道题的净改善,同时有多少道题改坏了。
    r = 0 是最乐观的情形(只改好不改坏),r = 1 表示改好的和改坏的一样多 ——
    此时净差异仍然是 delta,但分歧对数量翻倍,需要**更多**样本才能把它
    从噪声里分辨出来。

    这恰恰是很多人会算错的地方:直觉上"改好和改坏一样多"听起来是扯平了,
    实际上它对样本量的要求是最高的,因为它制造了最多的分歧。

    返回按 r 排序的列表,每项含 pi01 / pi10 / n_pairs / 可行性。
    不可行的组合 `n_pairs` 为 None 并带 `reason`,而不是编一个数字出来。
    """
    rows = []
    for r in churn_ratios:
        ok, reason = feasibility(delta, baseline, r)
        pi01 = r * delta
        pi10 = delta * (1 + r)
        n = None if not ok else required_pairs(pi01, pi10, alpha=alpha, power=power)
        rows.append({
            "churn_ratio": r,
            "pi01_regress": pi01,
            "pi10_improve": pi10,
            "discordant_rate": pi01 + pi10,
            "n_pairs": n,
            "feasible": ok,
            "reason": reason,
        })
    return rows


def render_table(delta: float, baseline: float, rows: list[dict],
                 alpha: float = 0.05, power: float = 0.8) -> str:
    lines = [
        f"目标:在基线 {baseline:.1%} 上检出 {delta:+.1%} 的提升"
        f"(α={alpha}, 功效={power:.0%})",
        "",
        f"{'退化比 r':>9}{'改坏 π01':>11}{'改好 π10':>11}{'分歧率':>9}{'需要题数':>11}  说明",
        "-" * 78,
    ]
    for row in rows:
        if row["feasible"]:
            lines.append(
                f"{row['churn_ratio']:>9.2f}{row['pi01_regress']:>11.1%}"
                f"{row['pi10_improve']:>11.1%}{row['discordant_rate']:>9.1%}"
                f"{row['n_pairs']:>11}  ")
        else:
            lines.append(
                f"{row['churn_ratio']:>9.2f}{'—':>11}{'—':>11}{'—':>9}{'不可行':>11}"
                f"  {row['reason']}")
    lines += [
        "-" * 78,
        "r = 0 是最乐观的情形(只改好不改坏),真实项目几乎不会出现。",
        "注意 r 越大需要越多样本 —— 因为分歧对更多,差异更难从噪声里分辨。",
    ]
    return "\n".join(lines)
