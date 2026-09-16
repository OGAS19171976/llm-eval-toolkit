"""``llm_eval_toolkit.llm`` 的测试。

默认**完全离线**:把 HTTP 层替掉,只测"我们怎么解析、怎么报错、密钥从哪来"。
真正打网络的用例要显式打开(``LEV_LIVE_LLM=1`` 且配了密钥),免得每次跑测试
都花用户的钱。
"""

from __future__ import annotations

import io
import json
import os
import urllib.error

import pytest

from llm_eval_toolkit import cli, llm

FAKE_KEY = "sk-test-DO-NOT-USE-0000"
REAL_KEY = os.environ.get(llm.KEY_ENV, "")

# 真打网络要**显式**打开,免得每次跑测试都花用户的钱。
LIVE = bool(REAL_KEY) and os.environ.get("LEV_LIVE_LLM") == "1"


# ===========================================================================
# 替身
# ===========================================================================
class _Response:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(payload=None, *, seen=None, sequence=None):
    """返回一个假的 urlopen。``sequence`` 用来模拟"前几次失败、之后成功"。"""
    calls: list = []

    def opener(req, timeout=None):
        calls.append(req)
        if seen is not None:
            seen.append(req)
        if sequence is not None:
            index = min(len(calls) - 1, len(sequence) - 1)
            item = sequence[index]
            if isinstance(item, Exception):
                raise item
            return _Response(item)
        return _Response(payload or {})

    opener.calls = calls
    return opener


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    """把密钥换成假的,保证离线用例绝不会误打真接口。"""
    monkeypatch.setenv(llm.KEY_ENV, FAKE_KEY)
    monkeypatch.delenv(llm.ALT_KEY_ENV, raising=False)


# ===========================================================================
# 密钥来源
# ===========================================================================
class TestApiKey:
    def test_reads_from_env(self):
        assert llm.api_key() == FAKE_KEY

    def test_missing_key_gives_a_human_message(self, monkeypatch):
        monkeypatch.delenv(llm.KEY_ENV, raising=False)
        monkeypatch.delenv(llm.ALT_KEY_ENV, raising=False)
        with pytest.raises(llm.MissingKeyError) as ctx:
            llm.api_key()
        message = str(ctx.value)
        assert llm.KEY_ENV in message
        assert "SetEnvironmentVariable" in message  # 给了可照抄的命令
        assert isinstance(ctx.value, llm.LLMError)

    def test_alt_env_wins(self, monkeypatch):
        monkeypatch.setenv(llm.ALT_KEY_ENV, "sk-alt")
        assert llm.api_key() == "sk-alt"

    def test_never_reads_from_files(self):
        """密钥只从环境变量来 —— 这一条是设计约束,用源码检查兜住。"""
        import inspect

        source = inspect.getsource(llm)
        assert "open(" not in source.replace("urlopen(", "")
        assert "read_text" not in source


# ===========================================================================
# 请求
# ===========================================================================
class TestRequest:
    def test_sends_bearer_header_and_json_body(self, monkeypatch):
        seen: list = []
        monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen(
            {"choices": [{"message": {"content": "好"}, "finish_reason": "stop"}],
             "model": "m", "usage": {"total_tokens": 3}}, seen=seen))
        llm.chat([{"role": "user", "content": "hi"}], model="m")
        req = seen[0]
        assert req.get_header("Authorization") == f"Bearer {FAKE_KEY}"
        assert req.get_header("Content-type") == "application/json"
        assert json.loads(req.data.decode("utf-8"))["messages"][0]["content"] == "hi"

    def test_retries_on_429_then_succeeds(self, monkeypatch):
        err = urllib.error.HTTPError("u", 429, "Too Many", {}, io.BytesIO(b"{}"))
        monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen(sequence=[
            err, {"choices": [{"message": {"content": "ok"}}], "model": "m"}]))
        monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
        assert llm.chat([{"role": "user", "content": "x"}]).text == "ok"

    def test_error_message_has_status_and_detail_and_no_key(self, monkeypatch):
        body = io.BytesIO(json.dumps(
            {"error": {"message": "Invalid API-key provided."}}).encode())
        err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, body)
        monkeypatch.setattr(llm.urllib.request, "urlopen",
                            _fake_urlopen(sequence=[err]))
        with pytest.raises(llm.LLMError) as ctx:
            llm.chat([{"role": "user", "content": "x"}])
        message = str(ctx.value)
        assert "HTTP 401" in message
        assert "Invalid API-key provided." in message
        assert FAKE_KEY not in message          # 绝不回显密钥

    def test_network_error_is_wrapped(self, monkeypatch):
        monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen(sequence=[
            urllib.error.URLError("连接被重置")]))
        monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
        with pytest.raises(llm.LLMError) as ctx:
            llm.chat([{"role": "user", "content": "x"}])
        assert "网络错误" in str(ctx.value)


# ===========================================================================
# 三种调用
# ===========================================================================
class TestCalls:
    def test_chat_parses_result(self, monkeypatch):
        monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen({
            "model": "qwen-x", "usage": {"total_tokens": 12},
            "choices": [{"message": {"content": " 答案 "}, "finish_reason": "stop"}]}))
        r = llm.chat([{"role": "user", "content": "q"}])
        assert r.text == "答案"          # 去掉首尾空白
        assert r.model == "qwen-x"
        assert r.usage["total_tokens"] == 12
        assert r.finish_reason == "stop"
        assert "qwen-x" in r.render()

    def test_chat_without_choices_raises(self, monkeypatch):
        monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen({"choices": []}))
        with pytest.raises(llm.LLMError):
            llm.chat([{"role": "user", "content": "q"}])

    def test_embed_orders_by_index(self, monkeypatch):
        monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen({
            "model": "emb", "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]}]}))
        out = llm.embed(["a", "b"])
        assert out.vectors == [[1.0, 0.0], [0.0, 1.0]]
        assert out.dim == 2

    def test_embed_empty_input_makes_no_call(self, monkeypatch):
        def boom(*_a, **_k):                      # pragma: no cover
            raise AssertionError("空输入不该发请求")

        monkeypatch.setattr(llm.urllib.request, "urlopen", boom)
        assert llm.embed([]).vectors == []

    def test_rerank_uses_native_endpoint_and_parses_hits(self, monkeypatch):
        seen: list = []
        monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen({
            "output": {"results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1}]}}, seen=seen))
        hits = llm.rerank("q", ["d0", "d1", "d2"], top_n=2)
        assert [h.index for h in hits] == [2, 0]
        assert hits[0].score == pytest.approx(0.9)
        assert "services/rerank/text-rerank" in seen[0].full_url
        payload = json.loads(seen[0].data.decode("utf-8"))
        assert payload["input"]["query"] == "q"
        assert payload["parameters"]["top_n"] == 2

    def test_list_models_sorted(self, monkeypatch):
        monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen({
            "data": [{"id": "b"}, {"id": "a"}]}))
        assert llm.list_models() == ["a", "b"]


# ===========================================================================
# CLI
# ===========================================================================
class TestCli:
    def test_help(self, capsys):
        assert cli.main(["llm"]) == 0
        assert "lev llm" in capsys.readouterr().out

    def test_unknown_subcommand(self, capsys):
        assert cli.main(["llm", "nope"]) == 2
        assert "未知子命令" in capsys.readouterr().err

    def test_models_json(self, monkeypatch, capsys):
        monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen({
            "data": [{"id": "qwen3.8-flash"}, {"id": "deepseek-v4.1-flash"}]}))
        assert cli.main(["llm", "models", "--grep", "qwen", "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == ["qwen3.8-flash"]

    def test_missing_key_exits_3(self, monkeypatch, capsys):
        monkeypatch.delenv(llm.KEY_ENV, raising=False)
        monkeypatch.delenv(llm.ALT_KEY_ENV, raising=False)
        assert cli.main(["llm", "models"]) == 3
        assert llm.KEY_ENV in capsys.readouterr().err

    def test_ask_prints_text_and_meta(self, monkeypatch, capsys):
        monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_urlopen({
            "model": "m", "usage": {"total_tokens": 5},
            "choices": [{"message": {"content": "你好"}, "finish_reason": "stop"}]}))
        assert cli.main(["llm", "ask", "打个招呼"]) == 0
        out = capsys.readouterr().out
        assert "你好" in out and "tokens=5" in out


# ===========================================================================
# 真打网络(默认关闭:要 DASHSCOPE_API_KEY 且 LEV_LIVE_LLM=1)
# ===========================================================================
@pytest.mark.skipif(not LIVE, reason="需要 DASHSCOPE_API_KEY 且 LEV_LIVE_LLM=1")
class TestLive:
    @pytest.fixture(autouse=True)
    def _real_key(self, _key, monkeypatch):
        """把真密钥放回去。

        显式声明依赖 ``_key``,是为了**用依赖表达执行顺序**,而不是赌"autouse
        fixture 谁先跑"。第一版靠条件分支让路,结果开着 live 跑时离线用例又挂了
        (它们断言的是假密钥)—— 同一处配置被两拨用例用相反的方式需要,
        就该把它写成一个可覆盖的依赖,而不是一个开关。
        """
        monkeypatch.setenv(llm.KEY_ENV, REAL_KEY)

    def test_check_end_to_end(self):
        info = llm.check()
        assert info["model_count"] > 0
        assert info["chat_reply"]
        assert info["embed_dim"] > 0

    def test_rerank_live(self):
        hits = llm.rerank("前向 Euler 的稳定域", ["是一个圆盘", "今天天气不错"])
        assert len(hits) == 2
        assert hits[0].index == 0
