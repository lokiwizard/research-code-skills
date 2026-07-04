"""优化器与学习率调度器：按配置构建，与 model/dataset/loss 同一"字典 + build 函数"写法。

优化器类型、学习率、调度策略都是常被消融的超参，所以和模型一样从 yaml 构建，
不在训练循环里写死。新增可选项 = 往对应字典加一行。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch

_OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}

_SCHEDULERS = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
}


def _coerce_floats(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """把数值型字符串转成 float：yaml 把 `5e-4` 这类写法解析成字符串，直接传给
    torch 会报错，这里统一兜底（`1.0e-4` 则本来就是 float，不受影响）。"""
    out = {}
    for k, v in cfg.items():
        if isinstance(v, str):
            try:
                v = float(v)
            except ValueError:
                pass
        out[k] = v
    return out


def build_optimizer(cfg: Dict[str, Any], params: Iterable) -> torch.optim.Optimizer:
    """按配置实例化优化器：cfg 形如 {"name": "Adam", "lr": 1e-3, **其余构造参数}。"""
    cfg = _coerce_floats(dict(cfg))
    name = cfg.pop("name")
    if name not in _OPTIMIZERS:
        raise KeyError(f"未知优化器 '{name}'，可选：{list(_OPTIMIZERS)}")
    return _OPTIMIZERS[name](params, **cfg)


def build_scheduler(cfg: Optional[Dict[str, Any]],
                    optimizer: torch.optim.Optimizer):
    """按配置实例化学习率调度器；cfg 缺省或 name 为 none 时返回 None（不调度）。"""
    if not cfg or str(cfg.get("name")).lower() in {"none", "null"}:
        return None
    cfg = _coerce_floats(dict(cfg))
    name = cfg.pop("name")
    if name not in _SCHEDULERS:
        raise KeyError(f"未知调度器 '{name}'，可选：{list(_SCHEDULERS)}")
    return _SCHEDULERS[name](optimizer, **cfg)
