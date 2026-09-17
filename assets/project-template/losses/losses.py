"""损失函数：配置里按名字选用、按字段传超参。

标准 MSE 直接用 torch 自带的 nn.MSELoss，不用自己包一层。示例的
CombinedLoss 演示"多项加权损失 + 可调权重超参"怎么写——论文里的
自定义损失（带正则项、多项加权）照这个套路扩展即可。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """L_total = mse + alpha * L1，演示"多项加权损失 + 可调权重超参"。

    把 alpha 暴露成配置项，正好可以拿来做"不同 alpha"的消融。
    """

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target) + self.alpha * self.l1(pred, target)
