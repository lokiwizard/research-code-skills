"""训练器：把"训练循环"这件事收敛到一个类里，train.py 只管装配。

职责边界（刻意保持窄）：
- 接收已经建好的 model / 数据加载器 / loss，自己只负责"循环 + 进度 + 日志 + 存档"。
- 不关心模型内部结构、不关心数据从哪来——这正是解耦的好处：
  换模型、换数据、换损失都不需要改这个文件。

支持断点续训：每个 epoch 存一份 last.pt（含模型/优化器/epoch/best），
`resume_from` 读回后从下一个 epoch 接着跑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.checkpoint import BestTracker, load_checkpoint, save_checkpoint
from utils.logger import CSVLogger
from utils.metrics import METRICS


def resolve_device(name: str) -> torch.device:
    """把配置里的 'auto'/'cpu'/'cuda' 解析成真正的 device。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    def __init__(
        self,
        cfg: Dict[str, Any],
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        exp_dir: Path,
        logger,
    ) -> None:
        self.cfg = cfg
        tcfg = cfg["train"]
        self.device = resolve_device(tcfg.get("device", "auto"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(tcfg["lr"]))

        self.epochs = int(tcfg["epochs"])
        self.log_interval = int(tcfg.get("log_interval", 1))
        self.ckpt_interval = int(tcfg.get("ckpt_interval", 0))
        self.metric_name = tcfg.get("monitor_metric", "mse")
        self.monitor_mode = tcfg.get("monitor_mode", "min")

        self.exp_dir = Path(exp_dir)
        self.logger = logger
        self.csv = CSVLogger(self.exp_dir / "metrics.csv")
        self.best = BestTracker(mode=self.monitor_mode)
        self.start_epoch = 1  # 续训时会被 resume_from 改写

    # ---- 断点续训：从 last.pt 恢复模型/优化器/进度 ----
    def resume_from(self, ckpt_path: str | Path) -> None:
        state = load_checkpoint(ckpt_path, map_location=str(self.device))
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.best.best = state.get("best")
        self.start_epoch = int(state["epoch"]) + 1  # 从下一个 epoch 接着跑
        self.logger.info(f"已从 {ckpt_path} 恢复，将从 epoch {self.start_epoch} 继续")

    # ---- 单个 epoch 的两个半场，拆开写更易读 ----
    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        # tqdm 进度条：实时显示 batch 进度与当前平均 loss；leave=False 让它每个 epoch 结束自动消失
        bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False)
        for x, y in bar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            bar.set_postfix(loss=f"{total / max(n, 1):.4f}")  # 进度条尾部显示运行平均 loss
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        loss_sum, n = 0.0, 0
        preds, targets = [], []
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss_sum += self.loss_fn(pred, y).item() * x.size(0)
            n += x.size(0)
            preds.append(pred.cpu())
            targets.append(y.cpu())
        pred_all = torch.cat(preds)
        target_all = torch.cat(targets)
        metric_fn = METRICS[self.metric_name]
        return {
            "val_loss": loss_sum / max(n, 1),
            f"val_{self.metric_name}": metric_fn(pred_all, target_all),
        }

    def fit(self) -> Dict[str, Any]:
        """跑完整训练，返回最终摘要（也会写到 metrics.json）。"""
        monitor_key = f"val_{self.metric_name}"
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_stats = self._validate()
            row = {"epoch": epoch, "train_loss": train_loss, **val_stats}
            self.csv.log(row)

            if epoch % self.log_interval == 0:
                msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in row.items())
                self.logger.info(msg)

            # 每个 epoch 都覆盖一份 last.pt，作为断点续训的入口
            save_checkpoint(self._state(epoch), self.exp_dir / "checkpoints", "last.pt")

            # 刷新最优就存 best.pt
            if self.best.update(val_stats[monitor_key]):
                save_checkpoint(self._state(epoch), self.exp_dir / "checkpoints", "best.pt")

            # 周期性存档，便于回溯任意阶段
            if self.ckpt_interval and epoch % self.ckpt_interval == 0:
                save_checkpoint(self._state(epoch), self.exp_dir / "checkpoints",
                                f"epoch_{epoch:04d}.pt")

        summary = {
            "best_" + monitor_key: self.best.best,
            "epochs": self.epochs,
        }
        self.logger.info(f"训练完成，最优 {monitor_key} = {self.best.best:.4f}")
        return summary

    def _state(self, epoch: int) -> Dict[str, Any]:
        """组装 checkpoint 内容：恢复训练所需的最小集合。"""
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best": self.best.best,
            "config": self.cfg,
        }
