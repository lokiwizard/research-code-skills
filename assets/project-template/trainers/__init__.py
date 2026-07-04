"""训练器包。"""

from .optim import build_optimizer, build_scheduler
from .trainer import Trainer, resolve_device

__all__ = ["Trainer", "resolve_device", "build_optimizer", "build_scheduler"]
