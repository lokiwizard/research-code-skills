# 论文方法 → PyTorch 实现

输入：论文方法描述、公式、伪代码或一个想法。输出：落进脚手架的模型、损失、训练流程
与配置。关键是**有顺序地拆解**，不要一上来就写大类。

## 五步拆解法

### 1. 先把数据流的 shape 画清楚

在写任何模块前，写下端到端的张量形状：

```
输入  x: [B, C, H, W]
  → Encoder      → z:    [B, D]
  → Quantizer    → z_q:  [B, D]      (+ 码本损失)
  → Decoder      → x_hat:[B, C, H, W]
```

shape 对不上是实现 bug 的最大来源，先对齐再写代码。

### 2. 一个公式/子结构 → 一个 nn.Module

把方法拆成最小的可独立测试单元，每个放进 `models/`。对每个模块回答三问：
**它做什么 / 为什么需要它 / 去掉会怎样**（这也是注释要写的）。

公式符号尽量和代码变量名对齐：

> 论文：$z_q = \arg\min_{e_k} \lVert z - e_k \rVert_2$

```python
class VectorQuantizer(nn.Module):
    """把连续向量 z 量化到最近的码本向量 e_k（公式 (3)）。"""
    def __init__(self, num_codes: int, dim: int):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, dim)   # e_k

    def forward(self, z):                               # z: [B, D]
        # 到每个码本向量的距离，取最近
        dist = torch.cdist(z, self.codebook.weight)    # [B, K]
        idx = dist.argmin(dim=1)                        # [B]
        z_q = self.codebook(idx)                        # [B, D]
        return z_q, idx
```

### 3. 目标函数逐项拆成可加权的 loss

把总损失按"项"拆开，每项一个可读的计算，权重做成配置项（于是天然可消融）：

> $\mathcal{L} = \mathcal{L}_{rec} + \beta\,\mathcal{L}_{commit}$

```python
class VQLoss(nn.Module):
    """重建损失 + β·承诺损失（commitment）。β 从配置进，便于消融。"""
    def __init__(self, beta: float = 0.25):
        super().__init__()
        self.beta = beta

    def forward(self, x_hat, x, z, z_q):
        rec = F.mse_loss(x_hat, x)
        commit = F.mse_loss(z, z_q.detach())
        return rec + self.beta * commit
```

### 4. 训练流程落进 trainers/

前向 → 算损失 → 反向 → 优化器 → （调度/裁剪/EMA 等）。和具体模型解耦：
`Trainer` 只调用 `model(x)` 和 `loss_fn(...)`，不关心内部。需要特殊前向（如返回多个量）时，
约定模型 `forward` 的返回结构，并在 Trainer 里按约定解包。

### 5. 把所有超参提进 configs/

维度、码本大小、层数、β、学习率、调度参数……全部进 yaml：

```yaml
model:   {name: VQVAE, dim: 64, num_codes: 512}
loss:    {name: VQLoss, beta: 0.25}
train:   {lr: 2.0e-4, epochs: 100, batch_size: 128}
```

## 落位对照表

| 论文里的东西 | 放进 | 说明 |
|---|---|---|
| 网络结构、子模块 | `models/` | 每个子结构一个 `nn.Module` |
| 目标函数的每一项 | `losses/` | 权重做成配置项 |
| 数据预处理、采样 | `datasets/` | `Dataset.__getitem__` |
| 训练循环、优化细节 | `trainers/` | 与模型解耦 |
| 所有超参、维度、权重 | `configs/` | yaml，唯一来源 |
| 评估指标 | `utils/metrics.py` | `f(pred,target)->float` |

## 常见坑

- **shape 不对**：先写 shape 注释，必要时用一条假数据 `assert` 形状。
- **把超参写死**：任何"可能要调/要消融"的数都进配置。
- **模块耦合**：模型不要 import 训练逻辑，损失不要依赖具体模型内部属性。
- **一步到位写大类**：先拆成小模块各自跑通，再组装。
- **与原文不一致**：把公式编号写进 docstring，方便对照核验；不确定处明确标注假设。
