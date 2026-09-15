"""标注一致性:评委之间到底一致到什么程度。

**为什么不能只报一致率** —— 这是本模块存在的全部理由。

假设标签里 95% 都是"通过"。让两个评委各自标注,即使其中一个人的判断
和另一个人基本上没关系,原始一致率也会有 90% 上下。这时候报"一致率 90%,
标注质量良好"是**错的**:90% 里绝大部分来自"两边都倾向说通过"这个巧合。

Kappa 把这个巧合扣掉:上面那种情形下它会落在 0 附近,甚至为负。

所以本模块的输出里,一致率和 Kappa **永远同时出现**,并且会额外给出
标签分布 —— 因为"标签是否高度偏斜"正是判断一致率可不可信的关键信息。

真正的收获其实还不是这两个数,而是 `disagreements`:分歧了哪些题。
逐题看一遍分歧,通常比任何统计量都更能告诉你标注规范哪里没写清楚。
"""

from __future__ import annotations

from collections import Counter

from .. import stats as S

__all__ = ["pivot", "raw_agreement", "agreement_report", "disagreements",
           "render_report"]


def pivot(rows, item_key="item_id", rater_key="annotator", label_key="label"):
    """长表 → {题目: {标注者: 标签}}。未标注的项不会出现在内层字典里。"""
    table: dict = {}
    for row in rows:
        label = row.get(label_key)
        if label in (None, ""):
            continue
        table.setdefault(row[item_key], {})[row[rater_key]] = label
    return table


def raw_agreement(table) -> float:
    """逐题取"同意配对数 / 总配对数",再对题取平均。

    用配对数而不是"出现过半数的标签",是为了对 3 人以上的情形也成立,
    而且不引入"谁是多数"这个额外假设。
    """
    scores = []
    for labels in table.values():
        values = list(labels.values())
        if len(values) < 2:
            continue
        pairs = total = 0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total += 1
                if values[i] == values[j]:
                    pairs += 1
        scores.append(pairs / total)
    return sum(scores) / len(scores) if scores else float("nan")


def disagreements(rows, item_key="item_id", rater_key="annotator",
                  label_key="label") -> list[dict]:
    """列出所有出现分歧的题,按题目 id 排序。

    这是这份报告里最该被人读的部分。
    """
    table = pivot(rows, item_key, rater_key, label_key)
    out = []
    for item, labels in table.items():
        if len(set(labels.values())) > 1:
            out.append({"item_id": item, "labels": dict(sorted(labels.items()))})
    return sorted(out, key=lambda d: str(d["item_id"]))


def agreement_report(rows, item_key="item_id", rater_key="annotator",
                     label_key="label") -> dict:
    """一致率 + Kappa + 分歧清单 + 该不该信的判断。"""
    table = pivot(rows, item_key, rater_key, label_key)
    raters = sorted({r[rater_key] for r in rows})
    n_items = len(table)
    counts = {len(v) for v in table.values()}
    n_annotations = sum(len(v) for v in table.values())

    label_counts = Counter(v for labels in table.values() for v in labels.values())
    total_labels = sum(label_counts.values()) or 1
    dominant, dominant_n = (label_counts.most_common(1)[0] if label_counts
                            else (None, 0))
    dominant_share = dominant_n / total_labels

    ra = raw_agreement(table)
    kappa = float("nan")
    method = "无法计算"

    if len(raters) == 2 and counts == {2}:
        # 两人、每题都标了 → Cohen's Kappa
        items = sorted(table, key=str)
        a = [table[i][raters[0]] for i in items]
        b = [table[i][raters[1]] for i in items]
        kappa, method = S.cohen_kappa(a, b), "Cohen's Kappa"
    elif counts == {len(raters)} and len(raters) >= 2:
        # 人数固定且都标了 → Fleiss' Kappa(需要类别矩阵)
        items = sorted(table, key=str)
        categories = sorted(label_counts, key=str)
        index = {c: i for i, c in enumerate(categories)}
        matrix = [[0] * len(categories) for _ in items]
        for i, item in enumerate(items):
            for label in table[item].values():
                matrix[i][index[label]] += 1
        kappa, method = S.fleiss_kappa(matrix), f"Fleiss' Kappa({len(raters)} 人)"

    # ---- 结论 ----
    notes = []
    if dominant_share > 0.9:
        notes.append(
            f"标签高度偏斜:{dominant_share:.0%} 都是「{dominant}」。"
            f"这种情况下**原始一致率一定会虚高**,以 Kappa 为准。")
    if kappa == kappa:                       # not nan
        if kappa < 0.4:
            notes.append("Kappa < 0.4:标注规范需要重写,现在的标注不足以支撑结论。")
        elif kappa < 0.6:
            notes.append("Kappa 0.4~0.6:中等。可以用于粗筛,不宜用于精细比较。")
        else:
            notes.append("Kappa >= 0.6:一致性可以接受。")
    if method == "无法计算":
        notes.append("每题标注人数不一致或有人漏标,Kappa 算不出来 —— "
                     "先用 template.validate_annotation_sheet 补齐。")
    if n_items and len(disagreements(rows, item_key, rater_key, label_key)) > 0:
        notes.append("务必逐条读分歧题:规范里没写清楚的地方,全在那里。")

    return {
        "n_items": n_items,
        "n_raters": len(raters),
        "n_annotations": n_annotations,
        "raw_agreement": ra,
        "kappa": kappa,
        "kappa_method": method,
        "label_counts": dict(label_counts),
        "dominant_label": dominant,
        "dominant_share": dominant_share,
        "n_disagreements": len(disagreements(rows, item_key, rater_key, label_key)),
        "notes": notes,
    }


def render_report(rep: dict) -> str:
    lines = [
        f"题数 {rep['n_items']}    标注者 {rep['n_raters']} 人    "
        f"标注总数 {rep['n_annotations']}",
        f"原始一致率 {rep['raw_agreement']:.1%}",
        (f"{rep['kappa_method']} {rep['kappa']:.3f}" if rep["kappa"] == rep["kappa"]
         else f"{rep['kappa_method']}"),
        f"标签分布 {rep['label_counts']}",
        f"分歧题 {rep['n_disagreements']} 道",
        "",
    ]
    lines += [f"  ! {n}" for n in rep["notes"]]
    return "\n".join(lines)
