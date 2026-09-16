"""``lev stability`` 的测试。

stability-lens 是**可选**依赖：缺了它，check/predict 必须给出清晰的安装提示，
而 note（纯读 JSON + 写 Markdown）仍然可用。
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
from pathlib import Path

import pytest

from llm_eval_toolkit import cli, stability

_COUNTER = itertools.count()


@pytest.fixture
def wd():
    """仓库内的临时目录。

    不用 pytest 的 ``tmp_path``：在受限沙箱里系统临时目录可能不可写。
    也不要用 ``tempfile.mkdtemp`` —— 它建出来的目录在某些沙箱策略下同样被拒写；
    普通的 ``Path.mkdir`` 才是稳的。
    """
    base = Path.cwd() / f"lev-stability-tmp-{os.getpid()}-{next(_COUNTER)}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _diag_check() -> dict:
    return {
        "rule": "gd",
        "title": "梯度下降",
        "verdict": "连续稳定但离散不稳定（缝隙）：η 是上界的 1.200 倍",
        "level": "gap",
        "exit_code": 2,
        "eta": 0.6,
        "eta_max": 0.5,
        "rho": 1.4,
        "rows": [["曲率 λ(A)", "1  4"], ["η_max", "0.5"]],
        "notes": [],
    }


def _diag_predict() -> dict:
    return {
        "dataset": {"name": "order_amount", "n": 1200, "params": 133},
        "checks": {"lambda_min": 4.46e-4, "is_strict_local_min": True},
        "lambda_max": {"assembled": 5.62984527, "power_iteration": 5.6298452,
                       "rel_err": 4.4e-13},
        "boundaries": [
            {"beta": 0.0, "eta_pred": 0.355250, "eta_measured": 0.355225, "rel_err": 7e-5},
            {"beta": 0.5, "eta_pred": 0.532874, "eta_measured": 0.532716, "rel_err": 3e-4},
        ],
        "notes": [],
    }


class TestRenderNote:
    def test_check_shape_mentions_verdict_and_gap(self):
        text = stability.render_note(_diag_check())
        assert "## 训练稳定性" in text
        assert "缝隙" in text
        assert "0.5" in text

    def test_predict_shape_has_table_and_conclusion(self):
        text = stability.render_note(_diag_predict())
        assert "λ_max" in text
        assert "| β | 预测 η_max | 实测边界 | 相对偏差 |" in text
        assert "线性化判据可用" in text

    def test_predict_shape_flags_large_deviation(self):
        data = _diag_predict()
        data["boundaries"][0]["rel_err"] = 0.5
        assert "需人工复核" in stability.render_note(data)


class TestAppendNote:
    def test_appends_and_preserves_existing(self, wd: Path):
        md = wd / "report.md"
        md.write_text("# 评估报告\n\n旧内容\n", encoding="utf-8")
        stability.append_note(md, _diag_check())
        text = md.read_text(encoding="utf-8")
        assert text.startswith("# 评估报告")
        assert "旧内容" in text
        assert "训练稳定性" in text

    def test_creates_file_when_missing(self, wd: Path):
        md = wd / "new.md"
        stability.append_note(md, _diag_check())
        assert md.exists()


class TestLoadDiagnosis:
    def test_rejects_unrelated_json(self, wd: Path):
        p = wd / "x.json"
        p.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        with pytest.raises(ValueError):
            stability.load_diagnosis(p)

    def test_reads_utf8_bom(self, wd: Path):
        """Notepad 之类的编辑器另存为常带 UTF-8 BOM。"""
        p = wd / "bom8.json"
        p.write_bytes(b"\xef\xbb\xbf" + json.dumps(_diag_check()).encode("utf-8"))
        assert stability.load_diagnosis(p)["rule"] == "gd"

    def test_reads_utf16(self, wd: Path):
        """`powershell ... --json > diag.json` 在 Windows 上写出的是 UTF-16。"""
        p = wd / "bom16.json"
        p.write_bytes(json.dumps(_diag_check()).encode("utf-16"))
        assert stability.load_diagnosis(p)["rule"] == "gd"


class TestCli:
    def test_note_roundtrip(self, wd: Path):
        diag = wd / "diag.json"
        diag.write_text(json.dumps(_diag_predict()), encoding="utf-8")
        md = wd / "report.md"
        md.write_text("# 报告\n", encoding="utf-8")
        code = cli.main(["stability", "note", str(diag), "--md", str(md)])
        assert code == 0
        assert "训练稳定性" in md.read_text(encoding="utf-8")

    def test_note_print_only(self, wd: Path, capsys):
        diag = wd / "diag.json"
        diag.write_text(json.dumps(_diag_check()), encoding="utf-8")
        code = cli.main(["stability", "note", str(diag), "--md", "ignored.md",
                         "--print-only"])
        assert code == 0
        assert "训练稳定性" in capsys.readouterr().out

    def test_note_bad_file(self, wd: Path, capsys):
        code = cli.main(["stability", "note", str(wd / "nope.json"),
                         "--md", str(wd / "r.md")])
        assert code == 64
        assert "读取诊断失败" in capsys.readouterr().err

    def test_unknown_subcommand(self, capsys):
        code = cli.main(["stability", "nope"])
        assert code == 2
        assert "未知子命令" in capsys.readouterr().err

    def test_help(self, capsys):
        assert cli.main(["stability"]) == 0
        assert "lev stability" in capsys.readouterr().out

    def test_missing_optional_dependency_is_graceful(self, monkeypatch, capsys):
        monkeypatch.setattr(stability, "_load", lambda: None)
        code = cli.main(["stability", "check", "--rule", "gd", "--spectrum", "1,4"])
        assert code == 3
        assert "stability-lens" in capsys.readouterr().err

    @pytest.mark.skipif(stability._load() is None, reason="需要 stability-lens")
    def test_check_passthrough_exit_code(self, capsys):
        code = cli.main(["stability", "check", "--rule", "gd", "--spectrum", "1,4",
                         "--eta", "0.6"])
        assert code == 2
        assert "缝隙" in capsys.readouterr().out

    @pytest.mark.skipif(stability._load() is None, reason="需要 stability-lens")
    def test_check_json_is_loadable(self, wd: Path, capsys):
        code = cli.main(["stability", "check", "--rule", "gd", "--spectrum", "1,4",
                         "--eta", "0.45", "--json"])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert stability.load_diagnosis  # 形状不变
        assert payload["exit_code"] == 0
