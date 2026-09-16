"""`_io` 的测试:写出的文本必须**跨平台字节一致**。

为什么值得单独测:Windows 上 `Path.write_text` 会把 `\\n` 翻译成 `\\r\\n`。
对一个"报告要提交进仓库、要被 diff、要被当基线比对"的工具来说,那意味着

  * git 每次提交都刷一屏 CRLF 警告;
  * 在 Windows 上生成的基线,拿到 Linux 上比对会显示"整篇都变了"。

所以这里锁两件事:工具函数本身用 LF,以及**各处调用确实走了工具函数**
(第二个才是容易漏的 —— 函数写对了但没人用,等于没写)。
"""

from __future__ import annotations

import json

from llm_eval_toolkit._io import write_json_lf, write_text_lf
from llm_eval_toolkit.report import view


def test_write_text_lf_has_no_carriage_return(workdir):
    p = write_text_lf(workdir / "a.txt", "第一行\n第二行\n")
    raw = p.read_bytes()
    assert b"\r\n" not in raw
    assert raw == "第一行\n第二行\n".encode("utf-8")


def test_write_text_lf_creates_parent_directories(workdir):
    p = write_text_lf(workdir / "deep" / "nest" / "a.txt", "x")
    assert p.exists()


def test_write_json_lf_keeps_chinese_readable(workdir):
    p = write_json_lf(workdir / "a.json", {"k": "中文"})
    raw = p.read_bytes()
    assert b"\r\n" not in raw
    assert "中文".encode("utf-8") in raw, "ensure_ascii=False,不该被转义成 \\uXXXX"
    assert json.loads(raw.decode("utf-8")) == {"k": "中文"}


def test_view_output_is_lf(workdir):
    """接线测试:`lev view` 写出来的文件必须是 LF。

    这个才是真正防回归的那条 —— 上面三个只证明工具函数对,
    证明不了调用方真的用了它。
    """
    src = workdir / "rep.json"
    write_json_lf(src, {"summary": {}, "records": [
        {"id": "q1", "ok": True, "card_hit_top1": True},
    ]})
    dst = workdir / "view.json"
    assert view.main([str(src), "--field", "card_hit_top1", "-o", str(dst)]) == 0
    assert b"\r\n" not in dst.read_bytes()


def test_compare_markdown_is_lf(workdir, capsys):
    from llm_eval_toolkit.report import compare

    def rep(flag):
        return {"summary": {}, "seconds_total": 1.0,
                "records": [{"id": f"q{i}", "ok": flag} for i in range(30)]}

    a = workdir / "a.json"
    b = workdir / "b.json"
    write_json_lf(a, rep(False))
    write_json_lf(b, rep(True))
    md = workdir / "r.md"
    assert compare.main([str(a), str(b), "--n-boot", "200", "--md", str(md)]) == 0
    capsys.readouterr()
    assert b"\r\n" not in md.read_bytes()


def test_template_jsonl_is_lf(workdir):
    from llm_eval_toolkit.build_evalset import template as T

    rows = T.build_annotation_sheet([{"id": "q1"}], ["A", "B"])
    p = T.to_jsonl(rows, workdir / "sheet.jsonl")
    assert b"\r\n" not in p.read_bytes()


def test_template_csv_is_lf_but_keeps_bom(workdir):
    from llm_eval_toolkit.build_evalset import template as T

    rows = T.build_annotation_sheet([{"id": "q1"}], ["A", "B"])
    p = T.to_csv(rows, workdir / "sheet.csv")
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "BOM 要留着,否则 Excel 打开中文乱码"
    assert b"\r\n" not in raw, "但行尾要统一成 LF"
