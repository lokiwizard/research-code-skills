"""Checkpoint 保存与恢复：训练中断不丢进度，最优模型可回溯。

只存"复现训练所需的最小集合"：模型权重、优化器状态、当前 epoch、最优指标。
另外维护一个 best.pt（按监控指标自动更新），评估时直接用它。
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
    """读取 checkpoint 字典（默认加载到 CPU，避免设备不匹配报错）。"""
    return torch.load(path, map_location=map_location)


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
