"""生成 examples/ 下的两份示例评估结果。

**为什么用脚本生成,而不是手写两个 JSON:**

手写的 JSON 没人知道它是怎么来的,改一个数字也无从判断会不会自相矛盾
(比如 summary 和逐题记录对不上 —— 这个坑本项目真的踩过:
summary 写着 51/51,逐题字段却全是 False)。这里从零构造,任何时候
`python examples/make_examples.py` 都能重现出一模一样的文件。

**构造出来的三种情形,正好覆盖评估时会遇到的三种结论:**

    ok            显著改善  60 题里 14 道从错变对、2 道从对变错
    family_hit    证据不足  净改善 5 道,刚好低于 6 道的检出门槛
    card_hit_top1 逐题相同  两版完全没有差异

第二种和第三种**必须**给出不同的结论,这是整个工具最想强调的一点:

  * "逐题相同"是**没有差异** —— 样本再多也没用;
  * "证据不足"是**没测出来** —— 扩样本可能有救。

把它们印成同一句"不显著",读者会去扩样本、白花标注预算。
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = 60


def _ids() -> list[str]:
    return [f"q{i:02d}" for i in range(N)]


def build() -> tuple[dict, dict]:
    ids = _ids()

    # ---- ok:显著改善 -------------------------------------------------
    # v1:前 42 道对。v2:把第 42~55 道(14 道)修好,但弄坏了第 0~1 道。
    # 净改善 +12,分歧对 16(14 好 / 2 坏)→ McNemar 精确 p ≈ 0.0042
    ok_v1 = [i < 42 for i in range(N)]
    ok_v2 = list(ok_v1)
    for i in range(42, 56):
        ok_v2[i] = True          # 改好 14 道
    for i in range(0, 2):
        ok_v2[i] = False         # 弄坏 2 道

    # ---- family_hit:证据不足 ------------------------------------------
    # 净改善 +5,反方向 0 道 → 精确 p = 2·(1/2)^5 = 0.0625,刚好不显著。
    # 而 60 题规模下要 6 道净改善才够 —— 这个例子就卡在门槛下面一道。
    fam_v1 = [i >= 10 for i in range(N)]
    fam_v2 = list(fam_v1)
    for i in range(0, 5):
        fam_v2[i] = True

    # ---- card_hit_top1:逐题相同 ---------------------------------------
    card_v1 = [i % 3 != 0 for i in range(N)]

    def report(ok, fam, card, seconds):
        return {
            "summary": {
                "total": N,
                "solved": sum(ok),
                "family_hit": sum(fam),
                "card_hit_top1": sum(card),
            },
            "seconds_total": seconds,
            "records": [
                {"id": ids[i], "ok": ok[i], "family_hit": fam[i],
                 "card_hit_top1": card[i]}
                for i in range(N)
            ],
        }

    old = report(ok_v1, fam_v1, card_v1, 12.5)
    new = report(ok_v2, fam_v2, card_v1, 13.1)
    return old, new


def main() -> int:
    old, new = build()
    for name, data in (("old.json", old), ("new.json", new)):
        path = HERE / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"写出 {path}  ({len(data['records'])} 题)")

    print("\n两版的原始比率:")
    for key in ("solved", "family_hit", "card_hit_top1"):
        a, b = old["summary"][key], new["summary"][key]
        print(f"  {key:<15} {a:>3}/{N}  →  {b:>3}/{N}")
    print("\n下一步:lev compare examples/old.json examples/new.json "
          "--label-a v1 --label-b v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
