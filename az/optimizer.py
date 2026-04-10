"""
DirectML-safe Adam optimizer.

torch-directml does not implement ``aten::lerp.Scalar_out``, which is used
by both the foreach and scalar code paths of ``torch.optim.Adam``:

  foreach path  → ``torch._foreach_lerp_``   (adam.py ~534)
  scalar path   → ``exp_avg.lerp_(grad, ...)`` (adam.py ~379)

Both fall back to CPU with a UserWarning.

This implementation replaces every ``lerp_`` call with the mathematically
equivalent ``mul_`` + ``add_``, which DirectML supports natively:

  x.lerp_(y, w)  ≡  x.mul_(1 - w).add_(y, alpha=w)

All ops used here (mul_, add_, addcmul_, sqrt, add_, addcdiv_) run on
the DirectML backend without fallback.
"""

import math
import torch
from torch.optim import Optimizer


class DirectMLSafeAdam(Optimizer):
    """
    Adam optimizer with no lerp_ calls — compatible with torch-directml.

    Matches the behaviour of ``torch.optim.Adam`` with L2 weight decay
    (weight_decay acts as a coefficient on a gradient penalty, same as the
    PyTorch default; use weight_decay=0 for decoupled / AdamW style).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # L2 weight decay: add wd * p to the gradient (matches PyTorch Adam default)
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # First moment  (replaces exp_avg.lerp_(grad, 1 - beta1))
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # Second moment (replaces exp_avg_sq.lerp_(grad*grad, 1 - beta2))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias-corrected step size
                step_size = lr * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)

                # Parameter update: p -= step_size * exp_avg / (sqrt(exp_avg_sq) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
