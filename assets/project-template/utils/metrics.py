"""评估指标：放在一处，训练和评估共用，避免两边算法不一致。

这里给的是通用回归指标作为占位；换成你的任务后，把图像任务的
PSNR/SSIM/LPIPS、分类任务的 Accuracy/F1 等加进来即可，签名保持
`metric(pred, target) -> float`，下游代码无需改动。
"""

from __future__ import annotations

import torch


@torch.no_grad()
def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """均方误差。"""
    return torch.mean((pred - target) ** 2).item()


@torch.no_grad()
def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """平均绝对误差。"""
    return torch.mean(torch.abs(pred - target)).item()


@torch.no_grad()
def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """决定系数 R²，越接近 1 越好。"""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2) + 1e-12
    return (1.0 - ss_res / ss_tot).item()


# 名字 -> 函数，方便按配置选指标。
METRICS = {"mse": mse, "mae": mae, "r2": r2_score}
