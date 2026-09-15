"""测试环境。

两件事:

1. 把 `src/` 放进 sys.path —— 这样不装包也能直接跑测试(CI 与本地都省一步)。
2. 提供 `workdir` 夹具替代 pytest 的 `tmp_path`。

关于第 2 点:某些受限环境里,pytest 的 tmpdir 插件会在 basetemp 下建
"编号目录 + 锁文件",而那套机制可能让 basetemp 变成**读不回来也删不掉**的
状态(实测复现过两次,连 takeown / icacls 都被拒)。`workdir` 只用最朴素的
mkdir,并且每次用独立的随机目录名,某一次留下的坏目录不会殃及后续运行。

顺手的好处:临时产物落在仓库里,排查失败用例时不用去 %TEMP% 里翻。

受限环境下的运行方式(本机沙箱实测):

    pytest tests/ -p no:cacheprovider -p no:tmpdir

正常环境直接 `pytest tests/` 即可 —— 所以不把它写进 pytest.ini。
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def workdir():
    """一次性的工作目录,用例结束后自动清理。"""
    d = REPO / f".pytest-work-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=False)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
