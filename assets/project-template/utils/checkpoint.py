"""Checkpoint 保存与恢复：训练中断不丢进度，最优模型可回溯，磁盘不爆。

存档策略（三类文件，各司其职）：
- last.pt   —— 每个 epoch 覆盖，含优化器状态，断点续训的唯一入口；
- best.pt   —— 监控指标刷新时更新，只存模型权重（评估用，不需要优化器）；
- epoch_*.pt —— 周期存档，**滚动保留最近 N 个**（prune_checkpoints），
  用于回溯中间阶段，同时避免长训练把磁盘占满。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], ckpt_dir: str | Path, filename: str) -> Path:
    """保存一个 checkpoint，返回写入路径。"""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    torch.save(state, path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """读取 checkpoint 字典（默认加载到 CPU，避免设备不匹配报错）。

    显式 weights_only=False：torch 2.6 起默认翻转为 True，会拒绝加载
    checkpoint 里的 RNG 状态等非张量对象。这里加载的是**本项目自己产出**
    的文件，可以信任；不要用它加载来路不明的 .pt。
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def prune_checkpoints(ckpt_dir: str | Path, keep: int,
                      pattern: str = "epoch_*.pt") -> list[Path]:
    """滚动清理周期 checkpoint：只保留最近 keep 个，返回被删除的路径。

    keep <= 0 表示不清理（全部保留）。周期档文件名带零填充 epoch 号
    （如 epoch_0010.pt），按文件名排序即按训练先后排序；best.pt / last.pt
    不匹配 pattern，永远不会被删。
    """
    if keep <= 0:
        return []
    old = sorted(Path(ckpt_dir).glob(pattern))[:-keep]
    for path in old:
        path.unlink()
    return old


class BestTracker:
    """跟踪监控指标，决定何时刷新 best.pt。

    mode='min' 表示越小越好（如 loss）；mode='max' 表示越大越好（如准确率/PSNR）。
    """

    def __init__(self, mode: str = "min") -> None:
        assert mode in {"min", "max"}
        self.mode = mode
        self.best: float | None = None

    def is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    def update(self, value: float) -> bool:
        """若刷新了最优则记录并返回 True。"""
        if self.is_better(value):
            self.best = value
            return True
        return False
