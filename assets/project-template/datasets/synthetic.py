"""示例数据集：合成回归数据，让框架开箱即跑、无需下载任何东西。

真实项目里把它换成你的数据集（读图像/文本/张量），只要继承
`torch.utils.data.Dataset`、实现 `__len__` 和 `__getitem__`，并在配置里
写好它的参数即可。
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticRegression(Dataset):
    """y = W·x + 噪声 的合成回归数据。

    参数
    ----
    n_samples: 样本数
    in_dim:    输入维度
    noise:     高斯噪声标准差
    seed:      数据生成种子（与训练种子分开，保证数据本身可复现）
    """

    def __init__(self, n_samples: int = 2000, in_dim: int = 16, noise: float = 0.1,
                 seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n_samples, in_dim, generator=g)
        true_w = torch.randn(in_dim, 1, generator=g)
        self.y = self.x @ true_w + noise * torch.randn(n_samples, 1, generator=g)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
