import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    max_lr: float = 3e-5,
    min_lr: float = 1e-7,
    total_steps: int = 100000,
    warmup_steps: int = 2000
):
    def lr_lambda(current_step: int):
        # 1) warmup阶段：线性上升
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # 2) cosine衰减阶段
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return (min_lr / max_lr) + (1 - min_lr / max_lr) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)