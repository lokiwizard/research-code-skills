"""设置 Python、NumPy、Torch 随机源，保存与恢复 RNG 状态。

固定种子不保证跨版本、设备或任意数据管线逐位一致。"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """固定所有常见随机源。

    deterministic=True 时进一步要求 cuDNN 走确定性算法：更可复现，但卷积
    可能略慢；追求速度的大规模训练可设 False。
    """
    # 对当前进程的 hash 随机化已不起作用，设置是为了让 DataLoader 子进程继承
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> Dict[str, Any]:
    """收集所有随机源的当前状态（存进 checkpoint 供续训恢复）。"""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any] | None) -> None:
    """恢复 get_rng_state 存下的状态；state 为空时静默跳过（兼容旧 checkpoint）。"""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    # GPU 数量变了（如 2 卡训练 → 1 卡续训）就跳过，不硬塞
    if "cuda" in state and torch.cuda.is_available() \
            and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])
