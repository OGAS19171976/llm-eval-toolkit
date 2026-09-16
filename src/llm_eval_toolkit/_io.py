"""统一的文本写出。

**一律 LF,与平台无关。**

Windows 上 `Path.write_text` 会把 `\\n` 翻译成 `\\r\\n`,于是同一份报告在
Windows 和 Linux 上**字节不同**。对一个"报告要提交进仓库、要被 diff、要被当
基线比对"的工具来说,这是实打实的麻烦:

  * `.gitattributes` 虽然能把仓库里存的规范化成 LF,但 git 每次都会刷一屏
    "CRLF will be replaced by LF" 警告 —— 警告刷多了就没人看了;
  * 更糟的是:你在 Windows 上生成的基线,拿到 Linux 上比对会显示
    "整篇都变了",而其实一个字都没改。

所以凡是本工具写出的文本,都走这两个函数。**顺带的好处**是写文件这件事
只有一处实现,以后要换成原子写(先写临时文件再 rename)也只用改这里。
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["write_text_lf", "write_json_lf"]


def write_text_lf(path: str | Path, text: str) -> Path:
    """按 UTF-8 + LF 写出文本,必要时创建父目录。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return out


def write_json_lf(path: str | Path, data, *, indent: int = 2) -> Path:
    """写出 JSON。`ensure_ascii=False` —— 报告里全是中文,转义了就没法读。"""
    return write_text_lf(path, json.dumps(data, ensure_ascii=False, indent=indent))
