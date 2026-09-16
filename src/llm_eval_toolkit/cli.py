"""命令行入口 `lev`。

三个子命令:

    lev compare A.json B.json      两份评估结果做配对比较
    lev view report.json ...       把某个逐题指标顶成主指标(为比较做准备)
    lev plan --delta 0.05          算"检出这个提升需要多少题"

分发的做法是先取第一个词再转交,而不是用 argparse 的 subparsers ——
因为 compare / view 的参数定义就在它们各自的模块里,用 subparsers 会把
同一份定义抄成两处,迟早对不上。
"""

from __future__ import annotations

import argparse
import sys

from .build_evalset import planning
from .report import compare, view
from . import stability

USAGE = """lev —— LLM 应用评估工具

用法:
  lev compare <基线.json> <候选.json> [选项]     两份结果做配对比较
  lev view <报告.json> --field F -o OUT          把逐题指标 F 顶成主指标
  lev plan --delta 0.05 [--baseline 0.7]         算需要多少题
  lev stability check --rule gd --spectrum "1,4" 训练步长体检（需要 stability-lens）
  lev stability predict --quick                  估 λ_max → 预测 η_max → 实测对拍
  lev stability note diag.json --md report.md    把稳定性结论并进评估报告

例子:
  lev compare old.json new.json --label-a 旧 --label-b 新 --md report.md
  lev view report.json --field card_hit_top3 --gate expect_card -o top3.json
  lev compare top1.json top3.json --metrics ok --label-a top-1 --label-b top-3
  lev plan --delta 0.05 --baseline 0.7

每个子命令都支持 -h 查看自己的选项。
"""


def _configure_stdout() -> None:
    """把 stdout/stderr 切成 UTF-8。

    **只在命令行入口做,不在 import 时做。**

    库模块在 import 阶段改全局 stdout 是一种污染 —— 使用者的程序可能
    另有安排。但 CLI 确实需要这一步:Windows 控制台默认 GBK,而报告里有
    中文、✅/❌ 之类的符号,一旦输出被重定向到文件或管道(自动化、CI、
    评审留痕都会这么做),就会抛 UnicodeEncodeError 或整片乱码 ——
    一份没人能读的评估报告等于没做。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _plan(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lev plan",
                                 description="算检出目标提升需要多少配对样本")
    ap.add_argument("--delta", type=float, required=True,
                    help="目标净提升,例如 0.05 表示 5 个百分点")
    ap.add_argument("--baseline", type=float, default=0.5,
                    help="基线的正确率(默认 0.5,最保守)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    args = ap.parse_args(argv)

    rows = planning.sample_size_table(args.delta, args.baseline,
                                      alpha=args.alpha, power=args.power)
    print(planning.render_table(args.delta, args.baseline, rows,
                                alpha=args.alpha, power=args.power))
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    command, rest = argv[0], argv[1:]
    if command == "compare":
        return compare.main(rest)
    if command == "view":
        return view.main(rest)
    if command == "plan":
        return _plan(rest)
    if command == "stability":
        return stability.main(rest)

    print(f"未知子命令:{command!r}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
