"""随机种子固定：可复现的第一道闸门。

复现失败十有八九是种子没固定全。这里一次性把 Python / NumPy / PyTorch
（CPU 和 GPU）的随机源都种上同一个数。
"""

from __future__ import annotations

import os
import random


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
