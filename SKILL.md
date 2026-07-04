---
name: research-code-skills
description: 科研代码撰写工具：生成模块化、可复现、可扩展的 PyTorch 研究项目脚手架，用 uv 管理环境，用 yaml 管理实验配置与消融，读取日志/CSV/JSON 自动产出表格/曲线/消融图与论文风格 caption，并把论文方法/公式/伪代码拆解成 PyTorch 模块、训练流程、损失与配置。Use when the user wants to start or organize a research/ML codebase, scaffold a deep-learning project (models/ datasets/ losses/ trainers/ configs/ scripts/ utils/ experiments/ results/), set up train.py/eval.py/config.yaml/logging/checkpoints/seed/experiment-naming, manage hyperparameters or ablations via yaml, analyze and visualize experiment results (curves, ablation plots, PSNR-vs-SNR, 3D ablation surfaces, comparison figures) with paper-ready captions, or turn a paper method / formula / pseudocode / idea into a clean PyTorch implementation. Emphasis on simple, readable, well-commented, decoupled code.
---

# Research Code Skills（科研代码撰写）

帮助研究者写出**模块化、可复现、可扩展**的实验代码。科研代码最怕后期混乱，
所以本技能默认强制：配置与代码分离、随机性可控、组件解耦、产物可追溯。

## 何时用

- 起一个新的深度学习/科研项目，需要标准目录与训练/评估骨架。
- 把已有的散乱脚本规整成模块化结构。
- 管理超参与消融实验（生成一组配置、逐一变量或网格扫描）。
- 实验跑完后，把日志/CSV/JSON 变成论文能用的表格、曲线、消融图与 caption。
- 把论文的方法描述/公式/伪代码/你的想法，落成 PyTorch 模块、损失、训练流程与配置。

## 动手前先对齐（重要）

**在写任何实现代码前，先和用户确认算法的流程与 idea**，达成一致再编码。至少确认清楚：

- 要解决的问题与核心想法（idea）是什么；
- 端到端的数据流与各阶段张量形状；
- 关键模块的划分、目标函数（损失）由哪几项构成、哪些量需要做成可配置/可消融的超参；
- 训练流程与评估指标。

把上述要点用简短的流程描述或伪代码先讲清楚、请用户确认（或纠正）后，再开始落代码。
脚手架/工具类、用户已明确给出完整方案的情况可直接执行；但凡涉及"论文方法→代码"或
新算法实现，**先对齐，后编码**，避免反复返工。

## 代码留痕：维护 DEVLOG.md（重要）

每个项目都有一份 `DEVLOG.md`（脚手架已自带模板），它是代码的"留痕"文件。要求：

- **对齐之后**，把确认好的核心 idea、算法流程（含数据流/shape/模块/损失）、代码计划
  写进 DEVLOG 的第 1–3 节，作为实现的依据。
- **每次写或改代码后**，在第 4 节「变更记录」追加一行：`日期 | 改了什么 | 为什么 | 涉及文件`。
  "为什么"最重要——它记录了设计决策，便于复盘、交接和论文写作时回溯。
- 改动较大时同步更新第 2–3 节（算法流程/计划），保持 DEVLOG 与代码一致。

无论是脚手架生成、落实论文方法，还是后续修改，都不要"改完就走"——**改完必记一笔**。

## 五条不可破坏的原则

1. **配置即唯一事实来源**：超参全部进 yaml，代码里不写死数字。改参用 `--set key=value` 或新配置文件，不改代码。
2. **可复现优先**：固定随机种子；每次运行保存当时的完整配置与配置哈希；实验命名带时间戳，绝不互相覆盖；锁依赖（`uv.lock`）。
3. **解耦**：模型/数据/损失/训练循环各管一摊。换其中任何一个，其余代码一行不改。
4. **简约可读**：小函数、单一职责、命名达意；注释解释"为什么"而非复述代码；不引入不必要的抽象与框架。
5. **产物可追溯**：每次实验自成一个目录，内含配置、日志、指标、checkpoint，"看到结果就能找回是怎么跑出来的"。

> 关于"解耦"的实现方式：用**显式的 `build_model/build_dataset/build_loss` 函数 + 一个字典**
> 按名字实例化组件即可，不要上注册表/装饰器那类隐式机制——对科研代码反而增加理解成本。
> 新增一个组件 = 写一个文件 + 在字典里加一行，一眼能看懂在哪改。

## 能力 1 · 项目脚手架生成

用 `scripts/new_project.py` 从 `assets/project-template/` 生成一个**完整可运行**的项目：

```bash
python scripts/new_project.py <项目名> --dest <父目录> [--git] [--uv]
```

生成的结构（与用户预期一致）：

```
<项目名>/
├── train.py  eval.py            # 入口（薄，只负责装配）
├── DEVLOG.md                    # 开发日志：算法流程/代码计划/改动留痕
├── configs/default.yaml         # 基线配置 + ablations/（自动生成的消融配置）
├── models/ datasets/ losses/    # 三类可替换组件，各有 build_* 函数
├── trainers/                    # 训练循环
├── utils/                       # 配置/种子/日志/checkpoint/指标
├── scripts/make_ablation.py analyze.py
├── experiments/  results/       # 训练产物 / 分析图表（gitignore）
└── pyproject.toml requirements.txt README.md .gitignore
```

模板自带：CSV+文本日志、best/last/周期 checkpoint（周期档**滚动保留**最近
`ckpt_keep` 个、best 只存权重，不做全量累积，磁盘可控）、**断点续训**（`--resume`，
RNG 状态随档保存，续训与一口气跑完结果一致）、优化器/学习率调度器从配置构建、
**tqdm 进度条**、种子固定、`<name>_<时间戳>_<配置哈希>` 命名规则、合成数据让框架
**开箱即跑**。生成后即可：`uv run python train.py --config configs/default.yaml`；
中断后 `uv run python train.py --resume experiments/<某次实验>` 续训；评估
`eval.py` 默认在与训练一致的验证集划分上算指标。

布局、命名与复现约定的细节见 `references/scaffold-layout.md`。
**优先用 `new_project.py` 生成再按需改**，不要手敲整套目录。

## 能力 2 · 用 uv 管理环境

默认 uv（Rust 实现，快）：

```bash
uv sync                          # 按 pyproject.toml / uv.lock 精确还原
uv run python train.py ...        # 在锁定环境里运行
uv add <pkg>                      # 增依赖（自动更新 lock）
uv python pin 3.11                # 锁 Python 版本
```

模板的 `pyproject.toml` 已配好依赖与 ruff。务必提交 `uv.lock`。给不用 uv 的环境留了
`requirements.txt` 兜底。

## 能力 3 · 实验配置与消融管理

- 所有超参在 `configs/default.yaml`；临时改用 `--set`，固定改法新建配置文件。
- 用 `scripts/make_ablation.py` 从基线批量生成配置：
  - `--mode oat`（逐一变量）= 标准消融，每次只改一个超参、其余保持基线，干净归因；
  - `--mode grid`（网格）= 超参搜索，取值的笛卡尔积。

```bash
python scripts/make_ablation.py --base configs/default.yaml --mode oat \
    --set loss.name=CombinedLoss --grid loss.alpha=0.0,0.1,0.5,1.0
```

生成的配置文件名编码了被改的超参，便于 `analyze.py` 按超参聚合画图。
详见 `references/config-and-ablation.md`。

## 能力 4 · 结果分析与可视化

用 `scripts/analyze.py` 读实验目录（config.yaml / metrics.csv / metrics.json / eval.json），
产出三类论文常用图表，每张图配一段**论文风格 caption**（数字真实算出，文字给初稿）：

```bash
python scripts/analyze.py summary --exp-root experiments              # 跨实验最终指标汇总表
python scripts/analyze.py curves  --exp-root experiments --metric val_loss   # 曲线 vs epoch
python scripts/analyze.py sweep   --exp-root experiments --x loss.alpha --y best_val_mse  # 消融曲线
python scripts/analyze.py sweep   --exp-root experiments --x model.depth --x2 model.hidden_dim --y best_val_mse  # 3D 消融曲面
```

`sweep` 正是"PSNR vs SNR""LPIPS vs token budget"这类图；给两个超参（如 α 与 L）即出
3D 消融曲面。它**不依赖脚手架**——指向任何含上述文件的目录都能用。重建图像/样本对比类
可视化、配色与排版规范见 `references/result-analysis.md`。

## 能力 5 · 论文方法到代码实现

输入论文方法描述、公式、伪代码或想法，按固定流程拆解，落进脚手架对应的槽位：

1. **数据流**：输入张量形状 → 各阶段形状 → 输出（先把 shape 标清楚）。
2. **模块分解**：每个公式/子结构对应一个 `nn.Module`，放进 `models/`。
3. **损失**：目标函数逐项拆成可加权的 loss，放进 `losses/`，权重作为配置项（天然可消融）。
4. **训练流程**：前向/反向/优化器/调度，落进 `trainers/`。
5. **配置项**：所有超参（维度、层数、权重、学习率…）进 `configs/`。

把公式符号与代码变量名对齐、为每个模块写"它做什么/为什么在/去掉会怎样"。
完整的拆解模板与公式→模块映射示例见 `references/paper-to-code.md`。

## 代码风格（贯穿全部能力）

- 简约：能用小函数就别上类层级；不为"将来可能"提前抽象（YAGNI）。
- 可读：命名自解释；与周围代码同密度地写注释；注释讲"为什么"。
- 解耦：组件间只通过明确的输入输出与配置交互，不互相 import 内部细节。
- 注释全面但不啰嗦：复杂逻辑、shape 约定、复现相关的点（种子/划分）一定要注。
- 默认中文注释（贴合使用者）；标识符用英文。

## 典型流程

起项目 → `new_project.py` 生成并 `uv sync` → 把方法落进 models/losses（能力5）→
跑基线 `train.py` → `make_ablation.py` 出一组消融 → 逐个训练 → `analyze.py` 出表格/曲线/caption。
端到端示例见 `examples/walkthrough.md`。

## 衔接

需要时可配合其他技能：文献/笔记（paper-notes-skills）、论文写作、图表规划与绘制、
引用管理等。本技能聚焦"把研究跑起来并产出可分析、可写进论文的结果"。
