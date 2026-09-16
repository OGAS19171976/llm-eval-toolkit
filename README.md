# llm-eval-toolkit

> 把「点估计」升级成「带不确定性的结论」。

它只回答一个问题,但这个问题决定其它所有事:

**我们怎么知道它真的变好了?**

大多数团队的答案是"看几个样例"或者"指标从 0.72 涨到 0.75"。这个库要做的是把
这种说法换成能被追问得住的说法:带置信区间的估计、配对显著性检验、多重比较
校正,以及一句"在现有样本量下,**你能检出的最小差异是多少**"。

---

## 先看一个真实例子

一个 56 道题的评估集,原报告是这么写的:

```
求解率        56/56    (100.0%)
top-1 族命中  55/56    (98.2%)
与标注一致    51/51    (100.0%)
```

三个数字,没有一个带区间。而 `56/56` 的教科书 Wald 区间宽度是 **0** ——
它会告诉你"成功率 = 100% ± 0",把一个"我没测出问题"说成了"没有问题"。

换成 Wilson 区间是 `[93.6%, 100%]`。这才是诚实的说法。

再看一个更实际的场景。同一套题,比较"只看检索排名第 1 的结果"和"放宽到前 3":

```bash
lev view report_definite.json --field card_hit_top1 --gate expect_card -o top1.json
lev view report_definite.json --field card_hit_top3 --gate expect_card -o top3.json
lev compare top1.json top3.json --metrics ok --label-a "top-1" --label-b "top-3"
```

```
指标                    top-1      95% Wilson      top-3      95% Wilson
求解 / 给出答案         75.5%   [61.9%,85.4%]      93.9%   [83.5%,97.9%]

指标                       Δ     95% CI (配对)   分歧对  净改善   精确 p   Holm p    结论
求解 / 给出答案        +18.4%   [+8.2%,+28.6%]      9     +9   0.0039   0.0039  显著改善
```

于是结论是:**top-1 的排序是真正的短板,值得投一个 reranker** ——
最多能拿回 9 道题(18.4%),而且这个提升在统计上站得住。

同样这套方法在**另一个**评估集上给出的是相反的结论:
top-1 97.8% → top-3 100.0%,Δ 只有 2.2%,分歧对只有 1 道,精确 p = 1.0000。
按观测到的分歧率,要检出这个效果需要 **351 道题**(现在只有 45 道)。
所以那边**不值得**投任何工程。

**这就是"看指标涨了"和"知道该不该动手"之间的差别。**

---

## 安装

```bash
pip install -e .            # 只需要 numpy
pip install -e ".[dev]"     # 跑测试还需要 pytest 和 scipy
```

命令行:

```bash
lev --help
python -m llm_eval_toolkit --help    # 没装包时(把 src 挂到 PYTHONPATH 上)
```

---

## 先跑起来看看

```bash
python examples/make_examples.py                              # 生成一对示例报告
lev compare examples/old.json examples/new.json --label-a v1 --label-b v2
```

输出会同时展示三种结论 —— **显著改善 / 证据不足 / 逐题相同** —— 以及为什么
后两者必须说成不同的话。逐条解释见 [`examples/README.md`](examples/README.md)。

方法论长文(可直接对外发布):[`docs/point-estimates-lie.md`](docs/point-estimates-lie.md)。

---

## 六个模块

| 模块 | 做什么 |
|---|---|
| `stats` | Wilson 区间、Bootstrap(BCa + percentile)、McNemar(精确 / mid-p / 校正卡方)、Cochran's Q、Holm、BH、Cohen's & Fleiss' Kappa、ECE、分层估计、检出能力门槛 |
| `metrics` | 检索层 Hit@k / Recall@k / Precision@k / MRR / MAP / NDCG@k;生成层里**可确定性判定**的部分(引用有效性、拒答率) |
| `build_evalset` | 评估集**设计**:要多少题、标注表生成与校验、标注一致性分析 |
| `report` | 两份结果的配对比较、显著性标注、文本与 markdown 报告 |
| `stability` | 训练/微调的**步长体检**（可选依赖 `stability-lens`）：预测 `η_max`、与实测边界对拍、把结论并进评估报告 |
| `llm` | 真实模型客户端：对话 / 向量化 / 重排 / 列模型。**标准库实现**，密钥只从环境变量读 |

> ⚠️ `stability-lens` 是**同级的另一个本地包,没有发布到 PyPI** —— 从 GitHub
> clone 本仓库的人是装不上它的。本仓库里 `lev stability note`(纯读写
> JSON / Markdown)不依赖它,只有 `check` / `predict` 需要;缺失时命令会
> 给出明确的安装提示,而不是抛一个看不懂的 ImportError。

---

## 顺带回答「这个学习率能不能跑」

评估报告回答的是「模型好不好」，但还有个更前置的问题。它在**跑之前**就有解析答案：

```bash
lev stability check --rule gd --spectrum "1,4" --eta 0.6    # 退出码 2 = 缝隙
lev stability note diag.json --md report.md                 # 并进评估报告
```

`--spectrum` 解释为曲率 `λ(A)`（`gd` 规则内部取 `J = −A`）。
需要更真实的场景时，`lev stability predict` 会在真实宽表上训一个小网络，
用 Hessian-向量积 + 幂迭代估 `λ_max`，再与实测稳定边界对拍：

```
β      预测 η_max = 2(1+β)/λ_max  实测边界  相对偏差
β=0                     0.355250  0.355225     0.01%
β=0.5                   0.532874  0.532716     0.03%
β=0.9                   0.674974  0.674773     0.03%
```

`stability-lens` 是**可选依赖**：没装它时，`check`/`predict` 会给出安装提示并返回退出码 3，
其余子命令完全不受影响。

---

## 接真实模型（`lev llm`）

评估总得有个模型可打。这部分用**标准库**实现，不引入 `requests` / `openai`：

```bash
export DASHSCOPE_API_KEY=sk-...        # Windows 见下面
lev llm check                          # 端到端自检：列模型 + 生成 + 向量化
lev llm models --grep qwen             # 看有哪些模型可用
lev llm ask "用一句话说明前向 Euler 的绝对稳定域"
```

实测输出（同一个 key 背后不止一个模型家族）：

```
端点      https://dashscope.aliyuncs.com/compatible-mode/v1
模型数    252
样例      MiniMax-M2.1, MiniMax-M2.5, MiniMax/MiniMax-M2.7, ...
生成      qwen3.8-flash -> 「可用」 (104 tokens)
向量化    qwen3.7-text-embedding-flash dim=1024
```

**密钥只从环境变量读**（`DASHSCOPE_API_KEY`，可用 `LEV_LLM_API_KEY` 覆盖），
从不从仓库文件读 —— 这样"密钥进 git"在设计上就不可能发生。缺失时报一句人话
并返回退出码 3，不是 `KeyError`：

```powershell
# Windows：用户级环境变量（永久），然后**重开终端**
[Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY","sk-...","User")
```

Python 里同样直接可用：

```python
from llm_eval_toolkit import llm

llm.ask("你好")                                   # 单轮
llm.chat([{"role": "user", "content": "..."}])    # 多轮，返回 ChatResult(含 usage/latency)
llm.embed(["文本 A", "文本 B"]).vectors           # 1024 维
llm.rerank("查询", ["文档1", "文档2"], top_n=2)   # 重排
llm.list_models()                                 # 可用模型
```

两处端点不一样，是实测出来的：对话与向量化走 **OpenAI 兼容模式**
（`/compatible-mode/v1`），而重排只有 **DashScope 原生端点**有
（兼容模式没有 rerank；原生端点下的 `text-generation/generation` 对同一批模型返 400）。
同一个 key 打**国际站**会返 401，所以默认端点用的是国内站。

> 测试默认**完全离线**（HTTP 层被替身替换）；真打网络的用例要显式打开：
> `LEV_LIVE_LLM=1 python -m pytest tests/test_llm.py`。免得每次跑测试都花你的钱。

---

## 回答「要多少样本才能检出 5% 的提升」

这是评估里最常被问到、也最容易答砸的问题。不算的话你只能回答"越多越好"。

```bash
lev plan --delta 0.05 --baseline 0.7
```

```
退化比 r     改坏 π01     改好 π10      分歧率       需要题数
     0.00       0.0%       5.0%     5.0%        155
     0.25       1.2%       6.2%     7.5%        234
     0.50       2.5%       7.5%    10.0%        312
     1.00       5.0%      10.0%    15.0%        469
```

**注意 r 越大需要越多样本。** 直觉上"改好 5 道、同时改坏 5 道"听起来是扯平了,
实际上它对样本量的要求最高 —— 因为净差异虽然不变,分歧对却翻倍了,
差异更难从噪声里分辨出来。

### 一个反直觉但很重要的推论

McNemar 精确检验**以分歧对的数量为条件**,所以:

> 单纯扩大评估集并不会提高判别力。

从 56 题扩到 200 题,如果新增的题两版表现都一样,检出能力**一点都不会提高**,
纯粹白花标注预算。要提高判别力,只能**针对性地补那些"两版确实会分歧"的题**。

`compare` 会把这个门槛直接算出来打印:「49 道题时,净改善至少要 6 道(12.2%)
才可能被判为显著。」

---

## 设计上的几个决定

**除 numpy 外零运行时依赖。** 评估基础设施最不该出现的情况就是"装不上包所以
跑不了"。scipy **只在测试里**出现,当独立裁判做交叉验证 —— 见
`tests/test_stats.py`,里面逐位比对了 Wilson、McNemar、BH 的结果,还有
bootstrap 区间**覆盖率**的实证检查(公式抄错了但看起来正常,只有覆盖率能抓住)。

**连模型客户端也是标准库写的。** `llm.py` 用 `urllib.request` 发那个 POST,
而不是引入 `requests` / `openai`。理由是同一个:一个"发 JSON、读 JSON"的调用
不值得把依赖翻一倍 —— 何况 `pip install openai` 会连带拖进一整套你并不需要的
SDK 表面。密钥只从环境变量读(**从不从仓库文件读**),所以"密钥进 git"在
设计上就不可能发生。

**"不显著"和"没差异"是两回事。** 两版逐题完全相同时,报告会说"逐题相同",
而不是"证据不足" —— 前者是差异为 0,样本再多也没用;后者是没测出来,
扩样本可能有救。把这两句话混成一句,读者会去扩样本,白花预算。

**缺失标注一律排除,不算 0 分。** 一道题没有相关项标注,它的 Recall@k 就是
没有定义的。算成 0 会把"标注缺失"变成"系统漏掉了",凭空制造指标下降。
被排除了多少题会明确写在报告里。

**生成层只放能确定性判定的指标。** 忠实度、答案相关性这类需要评委的指标
**故意没有实现** —— 一个指标的可信度不会超过它的评委。在评委本身被评估之前
(它稳定吗?有位置偏差吗?和人工一致吗?),报出来的数字没有意义。
那条流程是:`build_evalset.template` 生成标注表 → `build_evalset.agreement`
算 Kappa → `stats.expected_calibration_error` 量化校准。

**不做多重比较校正就等于没做检验。** 一次报 6 个指标、每个都用 α=0.05,
那么"至少有一个假阳性"的概率接近 `1 - 0.95^6 ≈ 26%`。所以 `compare` 默认对
同一批指标同时给 Holm(强控制 FWER,对外下结论用)和 BH(控制 FDR,
探索性分析用)。

---

## 当库用

`compare.analyze_pair` 是纯函数 —— 不碰文件系统、不打印、不读 `sys.argv`,
所以可以直接嵌进你自己的流程:

```python
from llm_eval_toolkit.report import compare

res = compare.analyze_pair(
    {r["id"]: r for r in old_report["records"]},
    {r["id"]: r for r in new_report["records"]},
    label_a="旧", label_b="新",
)
for row in res.rows:
    print(f"{row.label:<20} {row.est_a.rate:.1%} → {row.est_b.rate:.1%}  "
          f"Δ{row.mcnemar.delta:+.1%}  Holm p={row.holm_p:.4f}  {row.verdict}")
```

输入只需要 `{"records": [{"id": ..., <逐题布尔字段>}, ...]}` 这一层结构,
不绑定任何特定项目。

---

## 测试

```bash
pytest tests/                     # 160 项,约 1.3 秒
```

受限环境(容器、沙箱、被管控的机器)里,如果系统临时目录不可用,用:

```bash
pytest tests/ -p no:cacheprovider -p no:tmpdir
```

测试套件里的 `workdir` 夹具就是为这种情况准备的:它只用最朴素的 `mkdir`,
不依赖 pytest 的 tmpdir 插件。

---

## 目录

```
src/llm_eval_toolkit/
├── stats.py              统计基础(零第三方依赖,除 numpy)
├── cli.py                lev 命令
├── llm.py                真实模型客户端(标准库 urllib;密钥只读环境变量)
├── metrics/
│   ├── retrieval.py      Hit@k / Recall@k / MRR / NDCG@k
│   └── generation.py     引用有效性 / 拒答率(可确定性判定的部分)
├── build_evalset/
│   ├── planning.py       要多少题
│   ├── template.py       标注表生成与校验
│   └── agreement.py      标注一致性(Kappa)
└── report/
    ├── compare.py        配对比较(analyze_pair 是纯函数)
    └── view.py           把任意逐题指标顶成主指标
```

---

## 许可

MIT,见 `LICENSE`。

> 这个库脱胎于一个符号积分系统的评估实践(那个项目里,它把一个只靠肉眼比点
> 估计的流程,换成了带置信区间与显著性判定的流程,并顺手查出了三处统计数据
> 缺陷)。所有设计决定背后都有一次具体的踩坑,不是照搬教科书。
