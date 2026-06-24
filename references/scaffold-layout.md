# 项目布局与复现约定

脚手架（`assets/project-template/`）的设计依据。用 `scripts/new_project.py` 生成后，
按本文调整即可。

## 目录职责

| 目录/文件 | 职责 | 改动频率 |
|---|---|---|
| `train.py` / `eval.py` | **入口**，只做装配（读配置 → 建组件 → 跑），保持薄 | 低 |
| `configs/default.yaml` | 超参的唯一来源 | 每次实验 |
| `configs/ablations/` | `make_ablation.py` 生成的消融配置（自动、gitignore） | 自动 |
| `models/` | 模型 `nn.Module`；`build_model(cfg)` 按名字实例化 | 高 |
| `datasets/` | `torch.utils.data.Dataset`；`build_dataset(cfg)` | 中 |
| `losses/` | 损失 `nn.Module`；`build_loss(cfg)` | 中 |
| `trainers/` | 训练循环（前向/反向/日志/存档），与具体模型无关 | 低 |
| `utils/` | 配置、种子、日志、checkpoint、指标 | 低 |
| `scripts/` | 辅助工具：`make_ablation.py`、`analyze.py` | 低 |
| `experiments/` | 每次训练一个子目录（产物，gitignore） | 自动 |
| `results/` | 分析图表输出（gitignore） | 自动 |

## 为什么这样分

- **入口薄、逻辑沉**：`train.py` 不写训练细节，只把零件拼起来。读代码的人先看入口就懂全貌。
- **三类组件平行**：模型、数据、损失是研究中最常被替换的三样，各自独立成包，互不 import 内部。
- **训练循环与模型解耦**：`Trainer` 只接收"已经建好的"模型/数据/损失，所以换模型不动 `trainers/`。

## 组件如何按名字装配（不用注册表）

每个包的 `__init__.py` 里放一个**显式字典 + build 函数**：

```python
# models/__init__.py
from .mlp import MLP
from .your_model import YourModel        # 1) 新模型 import 进来

_MODELS = {"MLP": MLP, "YourModel": YourModel}   # 2) 加一行

def build_model(cfg):
    cfg = dict(cfg); name = cfg.pop("name")
    return _MODELS[name](**cfg)            # name 之外的字段直接当构造参数
```

配置里 `model: {name: YourModel, ...构造参数}` 即可。这样做的好处：**没有隐式魔法**，
"有哪些可选组件"和"怎么加新组件"都在这一处看得见，比注册表/装饰器更适合科研代码。

## 实验命名与可复现

- 实验目录名 = `<experiment.name>_<时间戳>_<配置哈希>`：分别回答"哪个实验/何时跑/什么配置"。
- 每次运行落盘：`config.yaml`（当时生效的完整配置）、`train.log`、`metrics.csv`、
  `metrics.json`、`checkpoints/best.pt`（+ 可选周期 checkpoint）。
- 复现三件套：**固定种子**（`utils/seed.py` 覆盖 Python/NumPy/Torch）、**保存配置**、**锁依赖**（`uv.lock`）。
- 数据划分也用独立种子，保证 train/val 切分可复现。

## 改造为真实任务的步骤

1. 把 `datasets/synthetic.py` 换成你的数据集（实现 `__len__`/`__getitem__`），更新 `_DATASETS`。
2. 把 `models/mlp.py` 换成你的模型，更新 `_MODELS`。
3. 在 `utils/metrics.py` 加你任务的指标（如 PSNR/SSIM/Accuracy），签名保持 `f(pred, target)->float`。
4. 按需在 `trainers/trainer.py` 调整训练循环（如加学习率调度、梯度裁剪）。
5. 更新 `configs/default.yaml` 的各 `name` 与超参。
