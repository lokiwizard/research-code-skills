# 端到端示例：从空目录到一组消融图

下面演示用本技能把一个研究跑通的完整路径。命令均为真实可执行（已在模板上验证）。

## 0. 动手前先对齐（最重要）

涉及新算法/论文方法实现时，先和用户确认：**核心 idea、数据流与各阶段 shape、模块划分、
损失构成、可消融的超参、评估指标**。确认后把要点写进项目的 `DEVLOG.md` 第 1–3 节，再编码。

## 1. 生成脚手架

```bash
python scripts/new_project.py snr-recon --dest ~/research --git
cd ~/research/snr-recon
uv sync
```

得到标准结构（models/ datasets/ losses/ trainers/ configs/ scripts/ utils/ experiments/ results/）
以及 train.py / eval.py / configs/default.yaml / README.md / requirements.txt / DEVLOG.md。

## 2. 跑通基线（开箱即跑，用合成数据）

```bash
uv run python train.py --config configs/default.yaml
```

训练时有 tqdm 进度条；产物落在 `experiments/baseline_<时间戳>_<哈希>/`：
`config.yaml`、`train.log`、`metrics.csv`、`metrics.json`、`checkpoints/{best,last}.pt`。

中途断了？直接续训：

```bash
uv run python train.py --resume experiments/baseline_<时间戳>_<哈希>
```

## 3. 把方法落进脚手架（能力 5）

按 `references/paper-to-code.md` 的五步：数据流 → 模块（`models/`）→ 损失（`losses/`，
权重做成配置项）→ 训练流程（`trainers/`）→ 超参（`configs/`）。每改一处，在 `DEVLOG.md`
第 4 节追加一行 `日期 | 改动 | 为什么 | 文件`。

## 4. 设计并生成消融（能力 3）

把"想研究的量"做成配置项（如 `dataset.snr`、`loss.alpha`），再批量生成配置：

```bash
# 逐一变量消融：每次只改一个，干净归因
uv run python scripts/make_ablation.py --base configs/default.yaml --mode oat \
    --set loss.name=CombinedLoss --grid loss.alpha=0.0,0.1,0.5,1.0

# 两个超参的网格（用于 3D 曲面）
uv run python scripts/make_ablation.py --base configs/default.yaml --mode grid \
    --grid model.depth=1,2,3 model.hidden_dim=16,64
```

逐个训练：

```bash
for cfg in configs/ablations/*.yaml; do uv run python train.py --config "$cfg"; done
```

## 5. 分析与出图（能力 4）

```bash
# 跨实验最终指标汇总表
uv run python scripts/analyze.py summary --exp-root experiments

# 训练曲线（多实验叠加）
uv run python scripts/analyze.py curves --exp-root experiments --metric val_loss

# 消融曲线：指标 vs 单个超参（如 PSNR vs SNR、LPIPS vs token budget）
uv run python scripts/analyze.py sweep --exp-root experiments --x loss.alpha --y best_val_mse

# 3D 消融曲面：指标关于两个超参（如 α 与 L）
uv run python scripts/analyze.py sweep --exp-root experiments \
    --x model.depth --x2 model.hidden_dim --y best_val_mse
```

每张图都会在 `results/` 下生成 PNG + 一段论文风格 caption（.md）。把表格、曲线和 caption
直接用于论文，必要时按论文语气微调文字。

> 提示：一次 `sweep` 最好指向"只包含该研究相关实验"的目录，避免把无关实验的点混进同一张图。

## 6. 评估

```bash
uv run python eval.py --exp-dir experiments/<某次实验>
```

至此：脚手架 → 基线 → 方法实现 → 消融 → 分析出图，全程模块化、可复现、有留痕。
