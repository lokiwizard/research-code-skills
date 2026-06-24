"""损失包：按名字把配置变成损失实例。

新增损失：建文件 + 在下面 import + 往 `_LOSSES` 加一行。
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .losses import CombinedLoss, MSELoss

_LOSSES = {
    "MSELoss": MSELoss,
    "CombinedLoss": CombinedLoss,
}


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    """按配置实例化损失：cfg 形如 {"name": "...", **构造参数}。"""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in _LOSSES:
        raise KeyError(f"未知损失 '{name}'，可选：{list(_LOSSES)}")
    return _LOSSES[name](**cfg)
