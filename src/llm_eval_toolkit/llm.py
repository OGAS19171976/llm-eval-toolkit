"""llm —— 真实模型客户端(**只用标准库**,不引入 requests / openai)。

为什么自己写 HTTP
-----------------
本库的硬约束是"除 numpy 外零运行时依赖"。模型客户端正是最容易破例的地方
(requests / openai / httpx),但这里要做的只是"POST 一个 JSON 再读回 JSON",
不值得为它把依赖翻一倍。于是用 ``urllib.request``。

端点分两处(实测出来的,不是照文档猜的):

* **OpenAI 兼容模式** —— 对话与向量化:
  ``https://dashscope.aliyuncs.com/compatible-mode/v1``
* **DashScope 原生** —— 重排:
  ``https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank``
  (兼容模式没有 rerank;原生端点下的 ``text-generation/generation`` 对同一批模型
  返 400,所以对话一律走兼容模式。)

密钥只从环境变量来
------------------
``DASHSCOPE_API_KEY``(可用 ``LEV_LLM_API_KEY`` 覆盖)。**从不从仓库里的文件读** ——
这样"密钥进 git"这件事在设计上就不可能发生。缺失时给一句人话,不是 KeyError。

其余可覆盖的环境变量:``LEV_LLM_BASE_URL`` / ``LEV_LLM_NATIVE_BASE_URL`` /
``LEV_LLM_MODEL`` / ``LEV_LLM_EMBED_MODEL`` / ``LEV_LLM_RERANK_MODEL``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "LLMError",
    "MissingKeyError",
    "ChatResult",
    "EmbeddingResult",
    "RerankHit",
    "KEY_ENV",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_RERANK_MODEL",
    "api_key",
    "base_url",
    "native_base_url",
    "default_model",
    "chat",
    "ask",
    "list_models",
    "embed",
    "rerank",
    "check",
    "main",
]

KEY_ENV = "DASHSCOPE_API_KEY"
ALT_KEY_ENV = "LEV_LLM_API_KEY"

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_NATIVE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_MODEL = "qwen3.8-flash"
DEFAULT_EMBED_MODEL = "qwen3.7-text-embedding-flash"
DEFAULT_RERANK_MODEL = "qwen3.7-text-rerank"

MISSING_KEY_HINT = (
    f"没有找到 API 密钥。请把它放进环境变量(**不要**写进仓库文件):\n"
    f'    # PowerShell(用户级,永久)\n'
    f'    [Environment]::SetEnvironmentVariable("{KEY_ENV}","sk-...","User")\n'
    f'    # bash\n'
    f'    export {KEY_ENV}=sk-...\n'
    f"设置后需要重开终端(环境变量只对新进程生效)。"
)


class LLMError(RuntimeError):
    """调用模型失败。消息里**不含**密钥。"""


class MissingKeyError(LLMError):
    """没配密钥 —— 与"调用失败"区分开,便于调用方决定是否跳过。"""


# ===========================================================================
# 配置
# ===========================================================================
def api_key() -> str:
    """从环境变量读密钥。两个名字都认,``LEV_LLM_API_KEY`` 优先。"""
    for name in (ALT_KEY_ENV, KEY_ENV):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise MissingKeyError(MISSING_KEY_HINT)


def base_url() -> str:
    return os.environ.get("LEV_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def native_base_url() -> str:
    return os.environ.get("LEV_LLM_NATIVE_BASE_URL", DEFAULT_NATIVE_BASE_URL).rstrip("/")


def default_model() -> str:
    return os.environ.get("LEV_LLM_MODEL", DEFAULT_MODEL)


# ===========================================================================
# 结果
# ===========================================================================
@dataclass
class ChatResult:
    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    finish_reason: str = ""

    def render(self) -> str:
        toks = self.usage.get("total_tokens", "?")
        return (f"{self.text}\n"
                f"    [model={self.model} tokens={toks} "
                f"latency={self.latency_s:.2f}s finish={self.finish_reason}]")


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


@dataclass
class RerankHit:
    index: int
    score: float
    document: str | None = None


# ===========================================================================
# HTTP(标准库)
# ===========================================================================
def _error_text(exc: urllib.error.HTTPError, limit: int = 300) -> str:
    """把服务端返回的错误体摘出来 —— 只留可读信息,不回显请求头。"""
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:                                   # pragma: no cover - 兜底
        return "（无法读取错误体）"
    try:
        data = json.loads(raw)
        msg = data.get("error", {}).get("message") or data.get("message") or raw
    except Exception:
        msg = raw
    msg = " ".join(str(msg).split())
    return msg[:limit] + ("…" if len(msg) > limit else "")


def _post(url: str, payload: dict[str, Any], *, key: str | None = None,
          timeout: float = 90.0, retries: int = 2) -> dict[str, Any]:
    """POST 一个 JSON 并返回解析后的 JSON。429/5xx 会重试。"""
    key = key or api_key()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last: LLMError | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err = LLMError(f"HTTP {exc.code}：{_error_text(exc)}")
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                last = err
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise err from None
        except urllib.error.URLError as exc:
            last = LLMError(f"网络错误：{exc.reason}")
            if attempt < retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise last from None
    raise last or LLMError("未知错误")                       # pragma: no cover


def _get(url: str, *, key: str | None = None, timeout: float = 30.0) -> dict[str, Any]:
    key = key or api_key()
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LLMError(f"HTTP {exc.code}：{_error_text(exc)}") from None
    except urllib.error.URLError as exc:
        raise LLMError(f"网络错误：{exc.reason}") from None


# ===========================================================================
# 对话 / 向量 / 重排
# ===========================================================================
def chat(
    messages: Sequence[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    timeout: float = 90.0,
    retries: int = 2,
) -> ChatResult:
    """一次对话补全。``messages`` 是 ``[{"role": "user", "content": "..."}]``。"""
    payload: dict[str, Any] = {
        "model": model or default_model(),
        "messages": list(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    t0 = time.time()
    data = _post(f"{base_url()}/chat/completions", payload,
                 timeout=timeout, retries=retries)
    latency = time.time() - t0
    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"响应里没有 choices：{str(data)[:200]}")
    message = choices[0].get("message") or {}
    return ChatResult(
        text=(message.get("content") or "").strip(),
        model=data.get("model", payload["model"]),
        usage=data.get("usage") or {},
        latency_s=latency,
        finish_reason=choices[0].get("finish_reason", ""),
    )


def ask(prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
    """单轮提问的便捷写法,只返回文本。"""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, **kwargs).text


def list_models(*, timeout: float = 30.0) -> list[str]:
    """可用模型 id 列表(实测能列出两百多个,含多个模型家族)。"""
    data = _get(f"{base_url()}/models", timeout=timeout)
    return sorted(str(m.get("id")) for m in (data.get("data") or []) if m.get("id"))


def embed(
    texts: Sequence[str],
    *,
    model: str | None = None,
    timeout: float = 90.0,
    retries: int = 2,
) -> EmbeddingResult:
    """文本向量化(兼容模式的 ``/embeddings``)。"""
    if not texts:
        return EmbeddingResult(vectors=[], model=model or DEFAULT_EMBED_MODEL)
    payload = {"model": model or os.environ.get("LEV_LLM_EMBED_MODEL", DEFAULT_EMBED_MODEL),
               "input": list(texts)}
    data = _post(f"{base_url()}/embeddings", payload, timeout=timeout, retries=retries)
    rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
    return EmbeddingResult(
        vectors=[list(r.get("embedding") or []) for r in rows],
        model=data.get("model", payload["model"]),
        usage=data.get("usage") or {},
    )


def rerank(
    query: str,
    documents: Sequence[str],
    *,
    model: str | None = None,
    top_n: int | None = None,
    return_documents: bool = False,
    timeout: float = 90.0,
    retries: int = 2,
) -> list[RerankHit]:
    """重排。走 DashScope **原生**端点 —— 兼容模式没有 rerank。"""
    payload: dict[str, Any] = {
        "model": model or os.environ.get("LEV_LLM_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        "input": {"query": query, "documents": list(documents)},
        "parameters": {"return_documents": return_documents},
    }
    if top_n is not None:
        payload["parameters"]["top_n"] = top_n
    url = f"{native_base_url()}/services/rerank/text-rerank/text-rerank"
    data = _post(url, payload, timeout=timeout, retries=retries)
    results = ((data.get("output") or {}).get("results")) or []
    hits: list[RerankHit] = []
    for r in results:
        doc = r.get("document")
        if isinstance(doc, dict):
            doc = doc.get("text")
        hits.append(RerankHit(index=int(r.get("index", -1)),
                              score=float(r.get("relevance_score", 0.0)),
                              document=doc))
    return hits


# ===========================================================================
# 自检
# ===========================================================================
def check(*, model: str | None = None, timeout: float = 60.0) -> dict[str, Any]:
    """端到端自检:列模型 + 一次最小生成 + 一次向量化。密钥缺失时抛 MissingKeyError。"""
    out: dict[str, Any] = {"base_url": base_url(), "key_present": True}
    models = list_models(timeout=timeout)
    out["model_count"] = len(models)
    out["models_sample"] = models[:6]

    target = model or default_model()
    if target not in models:
        out["model_warning"] = f"默认模型 {target} 不在列表里,请用 --model 指定"
        target = models[0] if models else target
    reply = chat([{"role": "user", "content": "只回复两个字:可用"}],
                 model=target, max_tokens=16, timeout=timeout)
    out["chat_model"] = reply.model
    out["chat_reply"] = reply.text
    out["chat_tokens"] = reply.usage.get("total_tokens")

    emb = embed(["自检"], timeout=timeout)
    out["embed_model"] = emb.model
    out["embed_dim"] = emb.dim
    return out


# ===========================================================================
# CLI：lev llm ...
# ===========================================================================
USAGE = """lev llm —— 真实模型客户端(标准库实现)

用法:
  lev llm check [--model M] [--json]     端到端自检(列模型 + 生成 + 向量化)
  lev llm models [--grep PAT] [--json]   列出可用模型
  lev llm ask "问题" [--model M] [--system S] [--temperature T] [--max-tokens N]

密钥:从环境变量 DASHSCOPE_API_KEY 读,不从任何文件读。
可覆盖:LEV_LLM_BASE_URL / LEV_LLM_NATIVE_BASE_URL / LEV_LLM_MODEL
"""


def _models(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lev llm models", description="列出可用模型")
    ap.add_argument("--grep", default=None, help="只显示匹配的子串(不区分大小写)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        ids = list_models()
    except LLMError as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 3
    if args.grep:
        ids = [i for i in ids if args.grep.lower() in i.lower()]
    if args.json:
        print(json.dumps(ids, ensure_ascii=False, indent=2))
    else:
        print(f"共 {len(ids)} 个模型：")
        for i in ids:
            print(f"  {i}")
    return 0


def _check(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lev llm check", description="端到端自检")
    ap.add_argument("--model", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        info = check(model=args.model)
    except MissingKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except LLMError as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 3
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"端点      {info['base_url']}")
        print(f"模型数    {info['model_count']}")
        print(f"样例      {', '.join(info['models_sample'])}")
        print(f"生成      {info['chat_model']} -> 「{info['chat_reply']}」"
              f" ({info['chat_tokens']} tokens)")
        print(f"向量化    {info['embed_model']} dim={info['embed_dim']}")
        if info.get("model_warning"):
            print(f"注意      {info['model_warning']}")
    return 0


def _ask(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lev llm ask", description="单轮提问")
    ap.add_argument("prompt")
    ap.add_argument("--model", default=None)
    ap.add_argument("--system", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=None, dest="max_tokens")
    args = ap.parse_args(argv)
    try:
        result = chat([{"role": "user", "content": args.prompt}], model=args.model,
                      temperature=args.temperature, max_tokens=args.max_tokens)
    except MissingKeyError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except LLMError as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 3
    print(result.render())
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    sub, rest = argv[0], argv[1:]
    if sub == "check":
        return _check(rest)
    if sub == "models":
        return _models(rest)
    if sub == "ask":
        return _ask(rest)
    print(f"未知子命令:{sub!r}\n\n{USAGE}", file=sys.stderr)
    return 2
