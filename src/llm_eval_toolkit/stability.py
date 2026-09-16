"""把「训练/微调的步长稳定性」接进评估流程。

评估报告回答的是「这个模型好不好」，但一个更前置的问题是
**「这个学习率到底能不能跑」**——它在跑之前就有解析答案：

* 前向 Euler 的绝对稳定域给出 ``η_max = min 2(−Re λ)/|λ|²``；
* 动量把上界线性放大到 ``2(1+β)/λ_max``；
* 真实网络里 ``λ_max`` 用 Hessian-向量积 + 幂迭代估出来（不构造 Hessian）。

这些计算在 ``stability-lens`` 包里（本工具**可选依赖**它，不装也能用其余功能）。
本模块只做两件事：

* ``lev stability check / predict``：透传并统一退出码，方便当 CI 的 gate；
* ``lev stability note``：把结论**追加**到已有的 Markdown 评估报告里，
  让"能不能跑"和"跑得好不好"出现在同一份文档里。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["main", "load_diagnosis", "render_note", "append_note"]

INSTALL_HINT = (
    "未找到 stability-lens。它是一个零依赖的独立包：\n"
    "    pip install -e ../stability-lens        # 或用 PYTHONPATH 指向它的目录\n"
    "缺少它时 `lev stability check/predict` 不可用，其余子命令不受影响。"
)


def _load():
    """延迟导入 stability-lens；不在 import 期污染依赖（这是可选依赖）。"""
    try:
        import stability_lens  # noqa: F401
    except ImportError:
        return None
    return stability_lens


# ===========================================================================
# 报告片段
# ===========================================================================
def _read_text(path: str | Path) -> str:
    """按 BOM 自动识别编码读取文本。

    Windows 上很常见的一种坑：``powershell ... --json > diag.json`` 写出来的是
    **UTF-16**，Notepad 另存为则带 **UTF-8 BOM**。一律按 utf-8 读，用户会在
    「文件明明存在」的情况下收到一个解码错误 —— 这里按字节 BOM 判断。
    """
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")     # utf-8-sig 同时兼容无 BOM 的普通 UTF-8


def load_diagnosis(path: str | Path) -> dict[str, Any]:
    """读入 ``stability-lens --json`` 的输出，两种形状都支持。

    * ``check`` 的输出：单次体检（含 ``verdict`` / ``eta_max`` / ``rows``）；
    * ``predict`` 的输出：真实训练研究（含 ``boundaries`` / ``lambda_max``）。
    """
    data = json.loads(_read_text(path))
    if not isinstance(data, dict):
        raise ValueError("期望一个 JSON 对象")
    if "verdict" not in data and "boundaries" not in data:
        raise ValueError("既不是 check 也不是 predict 的输出（缺少 verdict/boundaries）")
    return data


def render_note(data: dict[str, Any]) -> str:
    """把诊断渲染成一段可拼进评估报告的 Markdown。"""
    lines: list[str] = ["", "## 训练稳定性（步长体检）", ""]

    if "boundaries" in data:
        lam = data.get("lambda_max", {})
        ds = data.get("dataset", {})
        lines.append(f"- 数据：`{ds.get('name', '?')}`，n={ds.get('n', '?')}，"
                     f"参数量 p={ds.get('params', '?')}")
        lines.append(f"- λ_max：装配 {lam.get('assembled', float('nan')):.6g} / "
                     f"幂迭代 {lam.get('power_iteration', float('nan')):.6g}"
                     f"（相对误差 {lam.get('rel_err', float('nan')):.1e}）")
        checks = data.get("checks", {})
        lines.append(f"- θ* 是严格局部极小：{'是' if checks.get('is_strict_local_min') else '否'}"
                     f"（λ_min = {checks.get('lambda_min', float('nan')):.3e}）")
        lines += ["", "| β | 预测 η_max | 实测边界 | 相对偏差 |", "|---|---|---|---|"]
        for b in data["boundaries"]:
            lines.append(f"| {b.get('beta')} | {b.get('eta_pred'):.6f} | "
                         f"{b.get('eta_measured'):.6f} | {b.get('rel_err', 0) * 100:.2f}% |")
        worst = max((b.get("rel_err", 0.0) for b in data["boundaries"]), default=0.0)
        lines += ["", f"**结论**：预测与实测的最大偏差 {worst * 100:.2f}% —— "
                      f"{'线性化判据可用' if worst < 0.05 else '偏差偏大，需人工复核'}。"]
    else:
        lines.append(f"- 规则：`{data.get('rule', '?')}`")
        lines.append(f"- 判定：**{data.get('verdict', '?')}**")
        if data.get("eta") is not None:
            lines.append(f"- η = {data['eta']:.6g}，η_max = {data['eta_max']:.6g}")
        else:
            lines.append(f"- η_max = {data['eta_max']:.6g}")
        for key, value in data.get("rows", []):
            lines.append(f"  - {key}：{value}")
        if data.get("exit_code") == 2:
            lines.append("")
            lines.append("> 连续时间稳定但该步长下离散不稳定（缝隙）："
                         "把 η 降到 η_max 以下，或换 A-稳定的隐式格式。")

    lines.append("")
    return "\n".join(lines)


def append_note(md_path: str | Path, data: dict[str, Any]) -> Path:
    """把 =render_note= 的结果追加到 Markdown 报告末尾。"""
    path = Path(md_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + render_note(data), encoding="utf-8")
    return path


# ===========================================================================
# CLI
# ===========================================================================
def _check(argv: list[str]) -> int:
    sl = _load()
    if sl is None:
        print(INSTALL_HINT, file=sys.stderr)
        return 3
    ap = argparse.ArgumentParser(prog="lev stability check",
                                 description="对一组特征值/一个更新规则做步长体检")
    ap.add_argument("--rule", default="euler", choices=("euler", "gd", "heavy-ball", "lms"))
    ap.add_argument("--spectrum", default=None, help='特征值，逗号分隔，如 "1,4" 或 "-1+3i,-2"')
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--lambda-max", type=float, default=1.0, dest="lambda_max")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from stability_lens import report
    from stability_lens.diagnose import (diagnose_euler, diagnose_gd,
                                         diagnose_heavy_ball)

    if args.rule == "heavy-ball":
        d = diagnose_heavy_ball(eta=args.eta, beta=args.beta, lambda_max=args.lambda_max)
    elif args.spectrum is None:
        print("需要 --spectrum", file=sys.stderr)
        return 64
    elif args.rule == "gd":
        d = diagnose_gd(args.spectrum, eta=args.eta)
    else:
        d = diagnose_euler(args.spectrum, eta=args.eta)

    if args.json:
        print(json.dumps(report.to_json(d), ensure_ascii=False, indent=2))
    else:
        print(report.render(d))
    return d.exit_code


def _predict(argv: list[str]) -> int:
    sl = _load()
    if sl is None:
        print(INSTALL_HINT, file=sys.stderr)
        return 3
    ap = argparse.ArgumentParser(prog="lev stability predict",
                                 description="估 λ_max → 预测 η_max → 与实测边界对拍")
    ap.add_argument("--csv", default=None, help="数据表；默认用 stability-lens 自带的示例表")
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--betas", default="0,0.5,0.9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from stability_lens import datasets, experiment
    betas = tuple(float(x) for x in args.betas.replace(";", ",").split(",") if x.strip())
    if args.quick:
        ds = datasets.make_synthetic(n=400, d=6, seed=args.seed)
        res = experiment.run_study(dataset=ds, h=6, betas=betas, seed=args.seed,
                                   train_steps=60, test_steps=120, bisect_iters=10)
    else:
        ds = datasets.load_order_amount(path=args.csv, max_rows=args.rows, seed=args.seed)
        res = experiment.run_study(dataset=ds, h=args.hidden, betas=betas, seed=args.seed)

    if args.json:
        print(json.dumps(experiment.as_json(res), ensure_ascii=False, indent=2))
    else:
        print(experiment.render_study(res))
    ok = res.is_local_min and all(r.valid and r.rel_err < 0.05 for r in res.rows)
    return 0 if ok else 2


def _note(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lev stability note",
                                 description="把稳定性诊断追加进 Markdown 评估报告")
    ap.add_argument("diag", help="stability-lens check/predict 的 --json 输出文件")
    ap.add_argument("--md", required=True, help="要追加到的 Markdown 报告")
    ap.add_argument("--print-only", action="store_true", help="只打印片段，不写文件")
    args = ap.parse_args(argv)
    try:
        data = load_diagnosis(args.diag)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"读取诊断失败：{exc}", file=sys.stderr)
        return 64
    if args.print_only:
        print(render_note(data))
        return 0
    path = append_note(args.md, data)
    print(f"已把稳定性结论追加到 {path}")
    return 0


USAGE = """lev stability —— 训练/微调的步长稳定性

用法:
  lev stability check --rule gd --spectrum "1,4" --eta 0.6 [--json]
  lev stability predict [--quick] [--csv data.csv] [--json]
  lev stability note diag.json --md report.md

退出码:
  0 稳定 / 2 缝隙或预测不一致 / 3 缺少 stability-lens / 64 参数错误
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    sub, rest = argv[0], argv[1:]
    if sub == "check":
        return _check(rest)
    if sub == "predict":
        return _predict(rest)
    if sub == "note":
        return _note(rest)
    print(f"未知子命令:{sub!r}\n\n{USAGE}", file=sys.stderr)
    return 2
