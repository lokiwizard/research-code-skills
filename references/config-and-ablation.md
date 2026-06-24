# 配置与消融管理

## 配置原则

- **一切超参进 yaml**，代码里不出现魔法数字。看到一个数，应该能在配置里找到它。
- 配置结构按"组件"分块：`experiment / model / dataset / loss / train`。每块的 `name`
  指向一个组件，其余字段就是该组件的构造参数——配置项和代码参数一一对应。
- 三种改参方式，按"是否要留痕"选择：
  - 临时试一下：`--set train.lr=5e-4 model.depth=4`（不改文件）。
  - 要长期保留：复制 `default.yaml` 成新文件改。
  - 成组实验：用 `make_ablation.py` 批量生成。

## 消融 vs 超参搜索

| | 目的 | 工具模式 | 取值方式 |
|---|---|---|---|
| 消融（ablation） | 干净归因每个因素的贡献 | `--mode oat` | 每次只改一个超参，其余保持基线 |
| 超参搜索 | 找最优组合 | `--mode grid` | 多个超参取值的笛卡尔积 |

**消融要点**：一次只动一个变量。若同时改两个，结果差异就无法归因到具体哪个——
这也是 `oat` 模式存在的原因。

## make_ablation.py 用法

```bash
# 消融损失权重 alpha（先把损失换成带 alpha 的 CombinedLoss）
python scripts/make_ablation.py --base configs/default.yaml --mode oat \
    --set loss.name=CombinedLoss --grid loss.alpha=0.0,0.1,0.5,1.0

# 对学习率 × 深度做网格搜索
python scripts/make_ablation.py --base configs/default.yaml --mode grid \
    --grid train.lr=1e-3,5e-4 model.depth=2,4
```

- `--set`：先施加到基线上的固定改动（对所有生成的配置都生效）。
- `--grid`：要扫描的项，`key.path=v1,v2,...`。
- 生成的文件名编码了被改的超参（如 `baseline__alpha0.5.yaml`），且写进了
  `experiment.name`，因此 `analyze.py` 能按超参值聚合作图。

## 配置设计建议

- **把"想消融的东西"做成配置项**。例如想研究损失某项的权重影响，就让权重从配置进，
  而不是写死——这样消融只是改 yaml，不碰代码。
- 维度、层数、heads、dropout、学习率、batch、epoch、调度参数等都应可配。
- 数据相关的随机性（生成种子、划分种子）单独配置，与训练种子分开。
- 配置层级别太深（一般 2 层够用），太深反而难读难覆盖。

## 跑完一组消融

```bash
for cfg in configs/ablations/*.yaml; do
    uv run python train.py --config "$cfg"
done
uv run python scripts/analyze.py sweep --x loss.alpha --y best_val_mse --exp-root experiments
```
