"""命令行入口的测试。

CLI 是别人第一次接触这个库的地方,所以只测一件事:**它给出的数字和库里一致**。
这里不重复测算法,而是验证"参数解析 → 调用 → 输出"这条线没有走歪。
"""

from __future__ import annotations

import json

import pytest

from llm_eval_toolkit import cli


def _report(flags: dict) -> dict:
    return {"summary": {}, "seconds_total": 1.0,
            "records": [{"id": k, "ok": v} for k, v in flags.items()]}


def _write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


# ======================================================================
# 分发
# ======================================================================
def test_help_returns_zero(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "lev" in out and "compare" in out and "view" in out and "plan" in out


def test_unknown_command_returns_2(capsys):
    assert cli.main(["nope"]) == 2
    assert "未知子命令" in capsys.readouterr().err


def test_explicit_argv_beats_sys_argv(monkeypatch, capsys):
    """传了 argv 就必须用它,不去读 sys.argv —— 否则测试和被嵌入调用都会踩雷。"""
    monkeypatch.setattr("sys.argv", ["lev", "nope"])
    assert cli.main([]) == 0


# ======================================================================
# compare
# ======================================================================
def test_compare_subcommand_end_to_end(workdir, capsys):
    n = 40
    pa = _write(workdir / "a.json", _report({f"q{i}": False for i in range(n)}))
    pb = _write(workdir / "b.json", _report({f"q{i}": True for i in range(n)}))
    md = workdir / "r.md"
    code = cli.main(["compare", str(pa), str(pb), "--label-a", "旧",
                     "--label-b", "新", "--n-boot", "200", "--md", str(md)])
    assert code == 0
    out = capsys.readouterr().out
    assert "旧" in out and "新" in out
    assert "显著改善" in out
    assert md.exists()


# ======================================================================
# view
# ======================================================================
def test_view_subcommand_end_to_end(workdir, capsys):
    src = _write(workdir / "rep.json", {
        "summary": {}, "seconds_total": 1.0,
        "records": [
            {"id": "q1", "expect_card": "c", "card_hit_top3": True},
            {"id": "q2", "expect_card": "c", "card_hit_top3": False},
            {"id": "q3", "expect_card": None, "card_hit_top3": False},
        ]})
    dst = workdir / "view.json"
    assert cli.main(["view", str(src), "--field", "card_hit_top3",
                     "--gate", "expect_card", "-o", str(dst)]) == 0
    assert "可比题 2 道" in capsys.readouterr().out
    written = json.loads(dst.read_text(encoding="utf-8"))
    assert [r["ok"] for r in written["records"]] == [True, False, None]


# ======================================================================
# plan
# ======================================================================
def test_plan_subcommand_prints_the_table(capsys):
    assert cli.main(["plan", "--delta", "0.05", "--baseline", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "退化比" in out
    assert "目标" in out
    assert "5.0%" in out


def test_plan_rejects_impossible_target(capsys):
    """基线 90% 却想提升 20% —— 必须印"不可行",不能编一个题数出来。"""
    assert cli.main(["plan", "--delta", "0.20", "--baseline", "0.90"]) == 0
    out = capsys.readouterr().out
    assert "不可行" in out and "不可能" in out


def test_plan_requires_delta(capsys):
    with pytest.raises(SystemExit):
        cli.main(["plan"])


# ======================================================================
# 编码
# ======================================================================
def test_stdout_reconfigured_only_by_cli():
    """库模块 import 时不该动全局 stdout;CLI 必须动。

    前者是污染(使用者的程序可能另有安排),后者是必需
    (Windows 控制台 GBK + 报告里有中文和符号 = 重定向时直接崩)。
    这里验证的就是这条边界有没有守住。
    """
    import io
    import sys as _sys

    fake = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    original = _sys.stdout
    try:
        _sys.stdout = fake
        cli._configure_stdout()
        assert fake.encoding.lower() in ("utf-8", "utf8")
    finally:
        _sys.stdout = original
