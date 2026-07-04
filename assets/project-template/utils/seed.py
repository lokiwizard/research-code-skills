"""随机种子固定 + RNG 状态存取：可复现的第一道闸门。

复现失败十有八九是种子没固定全。这里一次性把 Python / NumPy / PyTorch
（CPU 和 GPU）的随机源都种上同一个数。

另提供 get_rng_state / set_rng_state：把所有随机源的**当前状态**打包进
checkpoint，续训时恢复——这样"中断后续训"和"一口气跑完"产生完全相同的
随机序列（数据 shuffle、dropout 等），曲线可以逐位对上。
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict


def set_seed(seed: int, deterministic: bool = True) -> None:
    """固定所有常见随机源。

    参数
    ----
    seed:
        随机种子。同一个 seed + 同一份代码 + 同一份配置应当得到同样的结果。
    deterministic:
        是否进一步要求 cuDNN 走确定性算法。开了更可复现，但卷积可能略慢；
        纯调试/复现时建议 True，追求速度的大规模训练可设 False。
    """
    # 注意：进程启动后设 PYTHONHASHSEED 对当前进程的 hash 随机化已不起作用，
    # 这里设置只是让 DataLoader worker 等子进程继承同一个值。
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass  # 没装 numpy 也不致命

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng_state() -> Dict[str, Any]:
    """收集所有随机源的当前状态（存进 checkpoint 供续训恢复）。"""
    state: Dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def set_rng_state(state: Dict[str, Any] | None) -> None:
    """恢复 get_rng_state 存下的状态；state 为空时静默跳过（兼容旧 checkpoint）。"""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "torch" in state:
        import torch

        torch.set_rng_state(state["torch"])
        # GPU 数量变了（如 2 卡训练 → 1 卡续训）就跳过，不硬塞
        if "cuda" in state and torch.cuda.is_available() \
                and len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
