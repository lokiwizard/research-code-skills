# {{PROJECT_NAME}}

科研实验项目，由 `research-code-skills` 脚手架生成。默认强制 **模块化、可复现、可扩展**。

## 目录结构

```
{{PROJECT_NAME}}/
├── train.py              # 训练入口（只负责装配，薄）
├── eval.py               # 评估入口（从实验目录加载 best.pt）
├── DEVLOG.md             # 开发日志：算法流程/代码计划/每次改动留痕
├── configs/
│   ├── default.yaml      # 基线配置：所有超参的唯一来源
│   └── ablations/        # make_ablation.py 生成的消融配置（自动）
├── models/               # 模型（build_model 按名字实例化）
├── datasets/             # 数据集（build_dataset）
├── losses/               # 损失（build_loss）
├── trainers/             # 训练循环
├── utils/                # 配置/种子/日志/checkpoint/指标
├── scripts/
│   ├── make_ablation.py  # 批量生成消融/超参配置
│   └── analyze.py        # 读结果 → 表格/曲线/消融图 + 论文 caption
├── experiments/          # 每次训练的输出（自动，gitignore）
└── results/              # 分析图表输出（自动，gitignore）
```

## 环境（uv）

```bash
uv sync                       # 按 pyproject.toml/uv.lock 还原环境
uv run python train.py --config configs/default.yaml
```

> 不用 uv 时：`pip install -r requirements.txt`，再去掉命令里的 `uv run`。

## 快速开始

```bash
# 1) 训练基线（开箱即跑，用合成数据，无需下载）
uv run python train.py --config configs/default.yaml

# 2) 临时改超参，不动配置文件
uv run python train.py --config configs/default.yaml --set train.optimizer.lr=5e-4 model.depth=4

# 3) 生成一组消融配置（逐一变量），再逐条训练
uv run python scripts/make_ablation.py --base configs/default.yaml --mode oat \
    --set loss.name=CombinedLoss --grid loss.alpha=0.0,0.1,0.5,1.0
uv run python train.py --config configs/ablations/<生成的配置>.yaml   # 对每个配置跑一次

# 4) 汇总与画图
uv run python scripts/analyze.py summary --exp-root experiments
uv run python scripts/analyze.py curves  --exp-root experiments --metric val_loss
uv run python scripts/analyze.py sweep   --exp-root experiments --x loss.alpha --y best_val_mse

# 5) 评估某次实验
uv run python eval.py --exp-dir experiments/<实验目录>

# 6) 断点续训：从某次实验的 last.pt 接着训（中断/加 epoch 都可用）
uv run python train.py --resume experiments/<实验目录>
```

训练时用 **tqdm 进度条**实时显示每个 epoch 的 batch 进度与运行平均 loss；
每个 epoch 自动覆盖一份 `checkpoints/last.pt`，断了用 `--resume` 即可续上。

checkpoint 不做全量累积，磁盘占用可控：`last.pt` 每 epoch 覆盖（含优化器，续训用）、
`best.pt` 指标刷新时更新（只存权重，评估用）、`epoch_*.pt` 周期档滚动保留最近
`train.ckpt_keep` 个（旧的自动删，设 0 可全保留）。

## 怎么扩展（关键在解耦）

加一个新模型只需两步，**训练代码一行不用改**：

1. 在 `models/your_model.py` 写好 `nn.Module`；
2. 在 `models/__init__.py` 里 `from .your_model import YourModel`，并往 `_MODELS`
   字典加一行 `"YourModel": YourModel`。

然后把 `configs/default.yaml` 的 `model.name` 改成你的名字、填上构造参数即可。
数据集、损失同理（`datasets/`、`losses/`）——都是同一个"字典 + build 函数"的写法，
没有任何隐式魔法，一眼能看懂在哪加、加什么。

## 复现约定

- 所有超参只在 yaml 里；代码不写死数字。
- 每次运行自动固定种子、保存当时的 `config.yaml` 和配置哈希。
- 实验命名：`<name>_<时间戳>_<配置哈希>`，看到结果就能找回配置。
- 提交 `uv.lock`，用 `uv sync` 在任何机器还原同一环境。
