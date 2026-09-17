# {{PROJECT_NAME}}

单设备回归示例：MLP 拟合合成数据 `y = Wx + n`，默认损失为 MSE。替换数据、模型和指标后可用于真实任务，接口变化时需调整训练循环。

## 环境与训练

```bash
uv sync
uv run python train.py --config configs/default.yaml
uv run python train.py --config configs/default.yaml --set train.optimizer.lr=5e-4
```

首次安装后提交 `uv.lock`；复现时用 `uv sync --locked`。不用 uv 时可安装 `requirements.txt`，但它不是精确版本锁。

产物位于 `experiments/<实验名>_<日期_时间>/`：有效配置、日志、CSV、指标摘要和 checkpoint。正式实验另需记录代码版本、环境设备、数据版本及划分。

## 恢复与评估

```bash
uv run python train.py --resume experiments/<实验目录>
uv run python train.py --resume experiments/<实验目录> --set train.epochs=40
uv run python eval.py --exp-dir experiments/<实验目录>
```

`last.pt` 保存最近完成轮次的训练状态，`best.pt` 保存验证选出的模型，周期档保留最近 `train.ckpt_keep` 个。续训只允许延长总轮数；调度器周期不随之修改。恢复一致性需在目标环境验证。

默认评估 best.pt 的验证集，结果写入 `eval.json`。其他 split/checkpoint 另存文件；重复同一评估会更新对应文件。此示例没有独立测试集，验证结果不能作为最终测试结果。

## 消融与分析

```bash
uv run python scripts/make_ablation.py --base configs/default.yaml --mode oat \
    --set loss.name=CombinedLoss --grid loss.alpha=0.0,0.1,0.5,1.0
uv run python train.py --config configs/ablations/<生成的配置>.yaml
uv run python scripts/analyze.py summary --exp-root experiments
uv run python scripts/analyze.py curves --exp-root experiments --metric val_loss
uv run python scripts/analyze.py sweep --exp-root experiments --x loss.alpha --y best_val_mse
```

分析前确认实验协议一致。脚本提供单次运行汇总和基础图注，不做多种子统计；正式图表需核对指标来源、重复次数和结论。

模型、数据、损失分别在 `models/`、`datasets/`、`losses/`；训练循环在 `trainers/`，超参在 `configs/`。方法、实验协议与验证记录写入 `DEVLOG.md`。

## 训练产物滚动保留

权重位于 `checkpoints/`：`last.pt`、`best.pt` 及最近 `train.ckpt_keep` 个周期档。
验证结果位于 `evaluations/`：`last.json`、`best.json` 及最近 `train.eval_keep` 个周期档；周期由 `train.eval_interval` 控制。
两类默认各保留 3 个周期档，另保留 best/last；keep=0 显式保留全部，interval=0 不生成周期档。

新档写入成功后才清理旧档。CSV 标量历史不删除；正式测试结果单独归档。模板未输出逐样本预测或重建图，真实任务增加这些大文件时需沿用同样的保留规则。
