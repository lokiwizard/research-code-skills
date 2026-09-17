# 示例：合成回归与损失权重扫描

本例验证脚手架流程，不代表真实任务的实验结论。新方法的研究假设、数据协议与预算应先写入 DEVLOG，关键缺项才向用户确认。

## 创建与检查

```bash
python scripts/new_project.py regression --dest /tmp
cd /tmp/regression
uv sync
uv run python train.py --config configs/default.yaml --set train.epochs=2
```

两轮训练用于检查接口，不能替代正确性验证。正式任务还需检查梯度、指标、小数据过拟合或任务不变量。

## 单独运行损失扫描

```bash
uv run python scripts/make_ablation.py --base configs/default.yaml --mode oat \
    --out-dir configs/loss-sweep \
    --set loss.name=CombinedLoss experiment.output_root=experiments/loss-sweep \
    --grid loss.alpha=0.0,0.1,0.5,1.0
for cfg in configs/loss-sweep/*.yaml; do
    uv run python train.py --config "$cfg"
done
uv run python scripts/analyze.py summary --exp-root experiments/loss-sweep
uv run python scripts/analyze.py sweep --exp-root experiments/loss-sweep \
    --x loss.alpha --y best_val_mse
```

`alpha=0` 是对照。此例每个条件仅一个训练种子，正式比较需重复并报告波动；当前分析脚本不做多种子聚合。

## 恢复与复查

将下列占位路径替换为实际实验目录：

```bash
uv run python train.py --resume experiments/loss-sweep/<实验目录> --set train.epochs=40
uv run python eval.py --exp-dir experiments/loss-sweep/<实验目录>
```

默认评估验证集。配置确定后，真实项目应使用独立测试数据评估，并保留代码、依赖锁、数据划分、原始结果及重建表图的命令。图注需核对后使用。
