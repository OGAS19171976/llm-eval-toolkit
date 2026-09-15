"""生成层里**可以确定性判定**的那部分指标。

LLM 生成质量的大部分指标(忠实度、答案相关性、有没有幻觉)都需要一个评委 ——
LLM 或人工。这个模块**故意不实现它们**。原因不是难,而是:

    一个指标的可信度不会超过它的评委。

在评委本身被评估之前(它稳定吗?有位置偏差吗?和人工一致吗?),报出来的
数字没有意义。那条流程在别处:

    build_evalset.template              生成标注表
    build_evalset.agreement             算 Kappa,确认评委之间一致到什么程度
    stats.expected_calibration_error     量化评委的置信度校准

本模块只做"不需要任何判断"的部分:引用是否指向真实存在的来源、系统有没有拒答。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import stats as S

__all__ = [
    "citation_validity", "citation_report", "CitationReport",
    "refusal_rate", "looks_like_refusal", "validate_refusal_detector",
]

# 拒答启发式的默认短语。**这只是起点,不是结论** —— 用之前必须先用人工标注
# 验证它(见 validate_refusal_detector)。
DEFAULT_REFUSAL_PHRASES = (
    "不知道", "无法回答", "没有找到", "未找到", "资料中没有", "无法确定",
    "cannot answer", "i don't know", "not enough information", "no relevant",
)


def citation_validity(cited, retrieved) -> float:
    """引用里有多大比例指向**真实存在于检索结果中**的来源。

    ⚠️ 它检验的是"出处存在",**不是**"出处支持这句话"。

    这两个经常被混为一谈,所以值得说清楚:

      * "引用准确率 = 95%" 听起来像在说"答案有据可查";
      * 而它实际只说明"没引用不存在的编号";
      * 至于引用的那段文字到底支持不支持结论,那是**蕴含判断**,
        需要评委,属于上面说的"必须先被评估的评委"。

    所以本函数叫 `citation_validity` 而不是 `citation_accuracy`。名字里
    藏着的这点区别,决定了读者会不会过度解读这个数字。

    没有引用时返回 nan(不是 0,也不是 1):"没引用"既不该被奖励也不该被惩罚,
    它该由另一个指标(引用覆盖率)去说。
    """
    cited = list(cited)
    if not cited:
        return float("nan")
    pool = set(retrieved)
    return sum(1 for c in cited if c in pool) / len(cited)


@dataclass
class CitationReport:
    n_cases: int
    n_no_citation: int
    n_defined: int
    mean_validity: float
    problem_cases: list[dict] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        return [
            f"题数 {self.n_cases},其中 {self.n_no_citation} 道没有引用(已排除)",
            f"引用有效性(指向真实来源的比例) {self.mean_validity:.1%}",
            f"存在无效引用的题: {len(self.problem_cases)} 道",
            "注意:这是「出处存在」而不是「出处支持结论」。后者需要评委。",
        ]


def citation_report(cases) -> CitationReport:
    """`cases` 是若干 `{"cited": [...], "retrieved": [...]}` 或
    `{"cited": [...], "retrieved": [...], "id": ...}`。"""
    cases = list(cases)
    values: list[float] = []
    no_citation = 0
    problems: list[dict] = []
    for case in cases:
        cited = list(case.get("cited") or [])
        retrieved = list(case.get("retrieved") or [])
        v = citation_validity(cited, retrieved)
        if v != v:                                  # nan
            no_citation += 1
            continue
        values.append(v)
        invalid = [c for c in cited if c not in set(retrieved)]
        if invalid:
            problems.append({"id": case.get("id"), "invalid": invalid,
                             "n_cited": len(cited)})
    mean = sum(values) / len(values) if values else float("nan")
    return CitationReport(len(cases), no_citation, len(values), mean, problems)


def refusal_rate(flags) -> float:
    """拒答率。`flags` 是逐题布尔序列(True = 系统选择不作答)。

    ⚠️ 拒答率和错误率是**两件不同的事**,不要合并成一个"准确率":

      * 拒答是设计上的诚实行为(检索不到就说不知道),不是失败;
      * 但拒答率过高意味着系统没用。

    所以两边都要单独报,而且拒答率的分母是**全部题**,不是"作答题" ——
    只算作答题会把"大部分都拒答"这种严重情况掩盖成一个漂亮的准确率。
    """
    flags = [bool(f) for f in flags]
    if not flags:
        return float("nan")
    return sum(flags) / len(flags)


def looks_like_refusal(text: str, phrases=DEFAULT_REFUSAL_PHRASES) -> bool:
    """关键词启发式判断一段回答是不是拒答。

    ⚠️ **它只是启发式。** 直接拿它去算拒答率,得到的只是另一个猜测:

      * 它会漏掉"换个说法拒绝"(例如"这超出了我的知识范围");
      * 它也会误判 —— 回答里出现"不知道"三个字,可能是在复述题干。

    正确用法:先人工标 100~200 条,用 `validate_refusal_detector` 量出它的
    准确率和 Kappa,达标了再拿去跑全量。这正是本仓库反复出现的那条原则 ——
    **评委(这里是启发式)自己也得先被评估。**
    """
    if not text:
        return False
    lowered = text.lower()
    return any(p.lower() in lowered for p in phrases)


def validate_refusal_detector(predicted, labels) -> dict:
    """拿人工标注验证这个启发式到底能不能用。

    `predicted` / `labels` 都是逐题布尔。返回准确率、混淆计数、Kappa,
    以及"如果 Kappa 太低就别用"的结论。
    """
    predicted = [bool(p) for p in predicted]
    labels = [bool(l) for l in labels]
    if len(predicted) != len(labels):
        raise ValueError("预测与标注必须等长")
    if not predicted:
        raise ValueError("空序列")
    tp = sum(1 for p, l in zip(predicted, labels) if p and l)
    fp = sum(1 for p, l in zip(predicted, labels) if p and not l)
    fn = sum(1 for p, l in zip(predicted, labels) if not p and l)
    tn = sum(1 for p, l in zip(predicted, labels) if not p and not l)
    n = len(predicted)
    accuracy = (tp + tn) / n
    kappa = S.cohen_kappa(predicted, labels)
    if kappa < 0.4:
        verdict = "不可用 —— Kappa 过低,这个启发式不能当作指标"
    elif kappa < 0.6:
        verdict = "勉强 —— 只能做粗筛,不要用它下结论"
    else:
        verdict = "可用 —— 但仍要在报告里说明它是启发式"
    return {
        "n": n, "accuracy": accuracy, "kappa": kappa,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "verdict": verdict,
    }
