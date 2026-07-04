"""训练器：把"训练循环"这件事收敛到一个类里，train.py 只管装配。

职责边界（刻意保持窄）：
- 接收已经建好的 model / 数据加载器 / loss，自己只负责"循环 + 进度 + 日志 + 存档"。
- 不关心模型内部结构、不关心数据从哪来——这正是解耦的好处：
  换模型、换数据、换损失都不需要改这个文件。

支持断点续训：每个 epoch 存一份 last.pt（模型/优化器/调度器/RNG 状态/epoch/best），
`resume_from` 读回后从下一个 epoch 接着跑。RNG 状态也随档保存并恢复，
所以"中断后续训"与"一口气跑完"的随机序列一致，结果可逐位对上。

存档策略（避免长训练把磁盘占满，不做全量累积）：
- last.pt：每 epoch 覆盖，含完整续训状态，续训入口；
- best.pt：指标刷新时更新，只存模型权重（评估不需要优化器，省一大半体积）；
- epoch_*.pt：每 ckpt_interval 个 epoch 存一份，滚动保留最近 ckpt_keep 个。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainers.optim import build_optimizer, build_scheduler
from utils.checkpoint import BestTracker, load_checkpoint, prune_checkpoints, save_checkpoint
from utils.logger import CSVLogger
from utils.metrics import METRICS
from utils.seed import get_rng_state, set_rng_state


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
        # 优化器/调度器也从配置构建（它们本身就是常消融的超参），不在这里写死
        self.optimizer = build_optimizer(tcfg["optimizer"], self.model.parameters())
        self.scheduler = build_scheduler(tcfg.get("scheduler"), self.optimizer)

        self.epochs = int(tcfg["epochs"])
        self.log_interval = int(tcfg.get("log_interval", 1))
        self.ckpt_interval = int(tcfg.get("ckpt_interval", 0))
        self.ckpt_keep = int(tcfg.get("ckpt_keep", 3))
        self.metric_name = tcfg.get("monitor_metric", "mse")
        self.monitor_mode = tcfg.get("monitor_mode", "min")

        self.exp_dir = Path(exp_dir)
        self.logger = logger
        self.csv = CSVLogger(self.exp_dir / "metrics.csv")
        self.best = BestTracker(mode=self.monitor_mode)
        self.best_epoch: int | None = None  # best.pt 对应的 epoch，进 metrics.json
        self.start_epoch = 1  # 续训时会被 resume_from 改写

    # ---- 断点续训：从 last.pt 恢复模型/优化器/调度器/RNG/进度 ----
    def resume_from(self, ckpt_path: str | Path) -> None:
        state = load_checkpoint(ckpt_path, map_location=str(self.device))
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        set_rng_state(state.get("rng"))  # 恢复随机序列，续训 == 一口气跑完
        self.best.best = state.get("best")
        self.best_epoch = state.get("best_epoch")
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
        t0 = time.monotonic()
        for epoch in range(self.start_epoch, self.epochs + 1):
            # 记录本 epoch 实际使用的 lr（调度器 step 之前取），进 CSV 便于画 lr 曲线
            row = {"epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"]}
            row["train_loss"] = self._train_one_epoch(epoch)
            val_stats = self._validate()
            row.update(val_stats)

            # 先更新 best/调度器再存档，last.pt 里的状态才是本 epoch 结束后的最新值
            best_improved = self.best.update(val_stats[monitor_key])
            if best_improved:
                self.best_epoch = epoch
            if self.scheduler is not None:
                self.scheduler.step()

            # 每个 epoch 都覆盖一份 last.pt（latest，含完整续训状态），断点续训的入口
            save_checkpoint(self._state(epoch), self.exp_dir / "checkpoints", "last.pt")

            # 刷新最优就存 best.pt。只存权重不存优化器——best 用于评估而非续训，
            # 而 Adam 的动量/方差状态约是权重体积的两倍，省下它磁盘占用小得多
            if best_improved:
                save_checkpoint(self._state(epoch, for_resume=False),
                                self.exp_dir / "checkpoints", "best.pt")

            # 周期性存档 + 滚动清理：只保留最近 ckpt_keep 个周期档，
            # 长训练也不会无限累积占满磁盘（best/last 不受影响）
            if self.ckpt_interval and epoch % self.ckpt_interval == 0:
                save_checkpoint(self._state(epoch), self.exp_dir / "checkpoints",
                                f"epoch_{epoch:04d}.pt")
                removed = prune_checkpoints(self.exp_dir / "checkpoints", self.ckpt_keep)
                if removed:
                    self.logger.info(
                        "滚动清理旧周期档：" + ", ".join(p.name for p in removed))

            # CSV 放在存档之后：若中途被杀，宁可 CSV 少一行（曲线缺个点），
            # 也不要 checkpoint 落后于 CSV 导致续训后出现重复 epoch 行
            self.csv.log(row)
            if epoch % self.log_interval == 0:
                msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in row.items())
                self.logger.info(msg)

        summary = {
            "best_" + monitor_key: self.best.best,
            "best_epoch": self.best_epoch,
            "epochs": self.epochs,
            # 只计本次运行的墙钟时间；续训时是"本段"而非累计
            "train_minutes": round((time.monotonic() - t0) / 60, 2),
        }
        self.logger.info(f"训练完成，最优 {monitor_key} = {self.best.best:.4f}"
                         f"（epoch {self.best_epoch}）")
        return summary

    def _state(self, epoch: int, for_resume: bool = True) -> Dict[str, Any]:
        """组装 checkpoint 内容。

        for_resume=True（last.pt / 周期档）额外带上优化器/调度器/RNG 状态，
        保证续训无缝；for_resume=False（best.pt）只存权重，体积约缩到 1/3。
        """
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "best": self.best.best,
            "best_epoch": self.best_epoch,
            "config": self.cfg,
        }
        if for_resume:
            state["optimizer"] = self.optimizer.state_dict()
            state["scheduler"] = (self.scheduler.state_dict()
                                  if self.scheduler is not None else None)
            state["rng"] = get_rng_state()
        return state
