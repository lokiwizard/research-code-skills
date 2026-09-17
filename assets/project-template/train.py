"""训练入口：加载配置、构建组件并运行 Trainer。支持从 last.pt 恢复。"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from datasets import build_dataset, split_train_val
from losses import build_loss
from models import build_model
from trainers import Trainer
from utils.config import validate_config
from utils.artifacts import write_json_atomic
from utils.checkpoint import load_checkpoint
from utils import (
    apply_overrides,
    load_config,
    make_experiment_dir,
    save_config,
    set_seed,
    setup_logger,
)


def build_dataloaders(cfg, seed):
    """建数据集并按 val_split 切出训练/验证两个 DataLoader。

    切分走 datasets.split_train_val（固定种子），eval.py 用同一函数，
    保证训练和评估看到完全相同的划分。
    """
    tcfg = cfg["train"]
    dataset = build_dataset(cfg["dataset"])
    train_set, val_set = split_train_val(dataset, float(tcfg.get("val_split", 0.2)),
                                         int(tcfg.get("split_seed", seed)))

    loader_kwargs = dict(
        batch_size=int(tcfg["batch_size"]),
        num_workers=int(tcfg.get("num_workers", 0)),   # 重预处理任务（图像等）调大
        pin_memory=bool(tcfg.get("pin_memory", False)),  # cuda 下开了能加速 H2D 拷贝
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def prepare_experiment(args):
    """根据是否续训，准备好 (配置, 实验目录, 是否续训)。

    - 新实验：读配置 + 命令行覆盖，新建带时间戳的实验目录并存下配置。
    - 续训：复用已有实验目录与其中的 config.yaml（仅允许 --set 延长 epochs）。
    """
    if args.resume:
        exp_dir = Path(args.resume)
        if not (exp_dir / "config.yaml").exists():
            raise SystemExit(f"续训目录里没有 config.yaml：{exp_dir}")
        original = load_config(exp_dir / "config.yaml")
        for item in args.overrides:
            if item.split("=", 1)[0] != "train.epochs":
                raise SystemExit("续训只允许覆盖 train.epochs；改变训练条件请新建实验")
        cfg = apply_overrides(original, args.overrides)
        validate_config(cfg)
        epochs = cfg["train"]["epochs"]
        if type(epochs) is not int or epochs < original["train"]["epochs"]:
            raise SystemExit("续训 train.epochs 必须为整数且不小于原总轮数")
        if not (exp_dir / "checkpoints" / "last.pt").is_file():
            raise SystemExit("续训目录缺少 checkpoints/last.pt")
        state = load_checkpoint(exp_dir / "checkpoints" / "last.pt")
        saved = state["config"]
        comparable = apply_overrides(cfg, [f"train.epochs={saved['train']['epochs']}"])
        if comparable != saved or epochs < state["epoch"]:
            raise SystemExit("配置与 checkpoint 不一致；恢复原配置或新开实验")
        return cfg, exp_dir, True

    if not args.config:
        raise SystemExit("请提供 --config（新实验）或 --resume（续训）")
    cfg = apply_overrides(load_config(args.config), args.overrides)
    validate_config(cfg)
    root = cfg["experiment"].get("output_root", "experiments")
    exp_dir = make_experiment_dir(root, cfg["experiment"]["name"])
    save_config(cfg, exp_dir / "config.yaml")  # 存下当时生效的完整配置
    return cfg, exp_dir, False


def main() -> None:
    parser = argparse.ArgumentParser(description="训练入口")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="yaml 配置路径（新实验必填）")
    source.add_argument("--resume", default=None,
                        help="断点续训：传入已有实验目录，从 last.pt 接着训")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides",
                        help="命令行覆盖，如 train.optimizer.lr=1e-4 model.depth=4")
    args = parser.parse_args()

    # 1) 配置与实验目录（区分新实验 / 续训）
    cfg, exp_dir, resuming = prepare_experiment(args)
    logger = setup_logger(exp_dir / "train.log")
    logger.info(f"{'续训' if resuming else '新实验'}：{exp_dir}")

    # 2) 复现：固定种子
    set_seed(int(cfg["experiment"]["seed"]))

    # 3) 按配置构建组件
    train_loader, val_loader = build_dataloaders(cfg, int(cfg["experiment"]["seed"]))
    model = build_model(cfg["model"])
    loss_fn = build_loss(cfg["loss"])
    logger.info(f"模型：{cfg['model']['name']} | 损失：{cfg['loss']['name']}")

    # 4) 训练（续训时先恢复进度）
    trainer = Trainer(cfg, model, train_loader, val_loader, loss_fn, exp_dir, logger)
    if resuming:
        trainer.resume_from(exp_dir / "checkpoints" / "last.pt")
        if args.overrides:
            initial = exp_dir / "config.initial.yaml"
            if not initial.exists():
                save_config(load_config(exp_dir / "config.yaml"), initial)
            save_config(cfg, exp_dir / "config.yaml")
    summary = trainer.fit()

    # 5) 落盘最终摘要，供 scripts/analyze.py 汇总
    write_json_atomic(exp_dir / "metrics.json", summary)


if __name__ == "__main__":
    main()
