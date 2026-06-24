"""一个最小可跑的示例模型：多层感知机（MLP）。

它存在的意义是把"模型该怎么写、怎么接进框架"演示清楚——
你真正的模型按同样的套路新建一个文件、写成 `nn.Module` 即可。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """全连接网络。

    参数
    ----
    in_dim:   输入特征维度
    hidden_dim: 隐藏层宽度
    out_dim:  输出维度
    depth:    隐藏层层数（>=1）
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
