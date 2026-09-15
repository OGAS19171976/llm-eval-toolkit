"""让 `python -m llm_eval_toolkit` 等价于 `lev`。

装包(`pip install -e .`)之后会用 `lev` 这个命令;没装包、只是把 src 挂到
PYTHONPATH 上时,这个入口是唯一的调用方式 —— 所以它必须在。
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
