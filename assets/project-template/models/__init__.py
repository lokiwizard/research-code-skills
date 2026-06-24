"""模型包：在这里按名字把配置变成模型实例。

新增模型只需两步：
1. 在本目录新建 `your_model.py`，写好 `nn.Module`；
2. 在下面 import 进来，并往 `_MODELS` 字典加一行。

这样训练代码只调用 `build_model(cfg)`，永远不用改。
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .mlp import MLP

# 配置里 model.name 用的就是这里的 key。一目了然，增删改都在这一处。
_MODELS = {
    "MLP": MLP,
}


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """按配置实例化模型：cfg 形如 {"name": "MLP", **构造参数}。

    `name` 之外的字段原样作为关键字参数传给模型，所以"配置项"和
    "构造参数"一一对应，新增超参只改 yaml 即可。
    """
    cfg = dict(cfg)  # 拷贝，别改到调用方的字典
    name = cfg.pop("name")
    if name not in _MODELS:
        raise KeyError(f"未知模型 '{name}'，可选：{list(_MODELS)}")
    return _MODELS[name](**cfg)
