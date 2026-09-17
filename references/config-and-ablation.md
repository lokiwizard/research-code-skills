# 配置与消融

配置保存实验可变项，按组件分组。维护的基线配置注明参数含义、单位和范围；自动生成的 YAML 是有效值快照，PyYAML 不保留原注释。

临时覆盖用 `--set key=value`，稳定方案保存新配置，批量实验用生成脚本。保存生成命令、基线版本及最终配置；文件名不能替代这些记录。

## 消融与搜索

- `oat` 每次改变一个因素，用于相对固定基线的比较。移除模块可能需要同步调整接口或维度，应记录为同一干预及其配套变化。
- `grid` 枚举组合，可用于超参搜索，也可检验因素交互。单因素结果不能说明不存在交互。
- 搜索使用验证集，选定方案后再测测试集。消融固定其余条件及种子集合，必要时控制参数量、算力或训练预算。

```bash
uv run python scripts/make_ablation.py --base configs/default.yaml --mode oat \
    --set loss.name=CombinedLoss --grid loss.alpha=0.0,0.1,0.5,1.0

uv run python scripts/make_ablation.py --base configs/default.yaml --mode grid \
    --grid train.optimizer.lr=1e-3,5e-4 model.depth=2,4
```

`--set` 先修改共同基线；`--grid` 指定扫描值。保留 `alpha=0` 等对照条件，确认基线已运行。不同研究问题使用不同配置和结果目录，避免混合汇总。

脚本只生成配置，不负责调度、失败重试或统计聚合。运行前核对组合数与预算；重复实验保持数据生成与划分种子固定，仅改变训练种子。

生成文件名包含参数完整路径，避免不同组件的同名参数碰撞。输出目录存在同名配置或生成名重复时，脚本报错并拒绝覆盖；使用新的输出目录重新生成。
