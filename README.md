# research-code-skills

面向科研代码撰写的 [Claude Code](https://claude.com/claude-code) / Codex Skill。
帮助研究者写出**模块化、可复现、可扩展**的实验代码——科研代码最怕后期混乱，本技能默认
强制：配置与代码分离、随机性可控、组件解耦、产物可追溯、改动留痕。

## 适用场景

- 起一个新的深度学习/科研项目，需要标准目录与训练/评估骨架。
- 把已有的散乱脚本规整成模块化结构。
- 用 yaml 管理超参与消融实验（逐一变量或网格扫描）。
- 实验跑完后，把日志/CSV/JSON 变成论文能用的表格、曲线、消融图与 caption。
- 把论文的方法描述/公式/伪代码/想法，落成 PyTorch 模块、损失、训练流程与配置。

## 六大能力

1. **项目脚手架生成** — 一键生成标准结构（`models/ datasets/ losses/ trainers/ configs/
   scripts/ utils/ experiments/ results/`）外加 `train.py / eval.py / config.yaml /
   README.md / requirements.txt / DEVLOG.md`，内置日志、checkpoint、随机种子固定、
   断点续训、tqdm 进度条、`<name>_<时间戳>_<配置哈希>` 实验命名。开箱即跑。
2. **uv 管理环境** — 默认 `uv`，`uv sync` 精确还原、`uv run` 运行、提交 `uv.lock`。
3. **配置与消融管理** — 超参全进 yaml（逐项注明含义）；`make_ablation.py` 从基线批量生成
   逐一变量（消融）或网格（搜索）配置。
4. **结果分析与可视化** — `analyze.py` 读实验产物，自动出汇总表、训练曲线、消融曲线
   （如 PSNR vs SNR、LPIPS vs token budget）、3D 消融曲面（如 α 与 L），并配论文风格 caption。
5. **论文方法到代码** — 按"数据流→模块→损失→训练→配置"五步，把方法/公式/伪代码拆解落地。
6. **简约可读的代码风格** — 小函数、单一职责、注释讲"为什么"、模块解耦，不引入隐式魔法。

## 设计原则（不可破坏）

- **配置即唯一事实来源**：代码里不写死超参。
- **可复现优先**：固定种子、保存配置与哈希、带时间戳命名、锁依赖。
- **解耦**：模型/数据/损失/训练循环各管一摊，换其一不动其余；用显式 `build_*` 函数 + 字典
  按名字装配组件，**不使用注册表/装饰器这类隐式机制**（对科研代码反而增加理解成本）。
- **先对齐后编码**：涉及论文方法/新算法实现，先与用户确认 idea 与算法流程再动手。
- **改动留痕**：每个项目维护 `DEVLOG.md`，记录算法流程、代码计划与每次改动的原因。

## 安装

```bash
git clone <this-repo>
cd research-code-skills

./install.sh              # 同时安装到 Claude Code 与 Codex（默认软链接）
./install.sh claude       # 仅 Claude Code
./install.sh codex        # 仅 Codex
./install.sh both --copy  # 复制而非软链接
./install.sh both --dry-run  # 仅打印计划安装的目标路径
```

安装后重启 Claude Code / Codex 即可发现该 skill。

## 直接使用脚手架（不经对话）

```bash
python scripts/new_project.py my-experiment --dest ~/research --git --uv
cd ~/research/my-experiment
uv run python train.py --config configs/default.yaml
```

## 目录

```
research-code-skills/
├── SKILL.md                       # 技能主指令
├── scripts/new_project.py         # 脚手架生成器
├── assets/project-template/       # 完整可运行的项目模板（脚手架的来源）
├── references/                    # 布局、配置消融、结果分析、方法→代码 四份指南
├── examples/walkthrough.md        # 端到端示例
└── install.sh
```

详细用法见 `SKILL.md` 与 `references/`，端到端示例见 `examples/walkthrough.md`。
