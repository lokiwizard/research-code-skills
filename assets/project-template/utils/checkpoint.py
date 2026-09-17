"""保存、恢复与滚动清理 checkpoint。best.pt 和 last.pt 不参与周期档清理。"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

from .artifacts import prune_snapshots


def save_checkpoint(state: Dict[str, Any], ckpt_dir: str | Path, filename: str) -> Path:
    """保存一个 checkpoint，返回写入路径。"""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    fd, temporary = tempfile.mkstemp(dir=ckpt_dir, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
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

    keep=0 表示显式保留全部，负数非法。按文件名中的 epoch 数值排序；best.pt / last.pt
    不匹配 pattern，永远不会被删。
    """
    return prune_snapshots(ckpt_dir, keep, pattern)


class BestTracker:
    """跟踪监控指标，决定何时刷新 best.pt。

    mode='min' 表示越小越好（如 loss）；mode='max' 表示越大越好（如准确率/PSNR）。
    """

    def __init__(self, mode: str = "min") -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode 必须为 min 或 max")
        self.mode = mode
        self.best: float | None = None

    def is_better(self, value: float) -> bool:
        if not math.isfinite(value):
            raise ValueError("监控指标为 NaN 或 Inf，拒绝更新 best")
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    def update(self, value: float) -> bool:
        """若刷新了最优则记录并返回 True。"""
        if self.is_better(value):
            self.best = value
            return True
        return False
