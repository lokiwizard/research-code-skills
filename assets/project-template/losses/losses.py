"""损失函数：配置里按名字选用、按字段传超参。

示例给了 MSE 和一个带权重的组合损失，演示"损失也可以有自己的超参"。
论文里的自定义损失（带正则项、多项加权）按 CombinedLoss 的写法扩展即可。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """标准均方误差损失。"""

    def __init__(self) -> None:
        super().__init__()
        self.fn = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fn(pred, target)


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
