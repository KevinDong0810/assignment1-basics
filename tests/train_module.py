from __future__ import annotations

from typing import Any

import torch
from torch import nn
import numpy as np
import math
from collections.abc import Callable, Iterable
from typing import Optional
import torch.nn.functional as F
import numpy.typing as npt

class CrossEntropyLoss(nn.Module):

    def __init__(self,):
        super().__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        max_logits = torch.max(logits, dim=-1, keepdim=True).values
        logits = logits - max_logits

        targets = targets.unsqueeze(dim=-1)
        exp_sum = torch.sum(torch.exp(logits), dim=-1, keepdim=True)
        log_exp_sum = torch.log(exp_sum)  # shape [..., batch_size, 1]

        target_logits = torch.gather(logits, -1, targets)
        loss = -target_logits + log_exp_sum  # shape [..., batch_size, 1]
        avg_loss = torch.mean(loss)
        return avg_loss


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        }
        super().__init__(params, defaults)
    
    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data
                t = state.get("t", 1)
                lrt = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr * weight_decay * p.data

                m = state.get("m", 0)
                v = state.get("v", 0)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                p.data -= lrt * m / (torch.sqrt(v) + eps)

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        
        return loss
                

def lr_cosine_func(it, max_lr, min_lr, warm_it, cos_it):
    if it < warm_it:
        return it / warm_it * max_lr
    elif it < cos_it:
        angle = (it - warm_it) / (cos_it - warm_it) * math.pi
        delta = 0.5 * (1 + math.cos(angle)) * (max_lr - min_lr)
        return min_lr + delta
    else:
        return min_lr


def gradient_clipping(params, max_norm):
    total_norm = 0.0
    for param in params:
        if param.grad is not None:
            total_norm += torch.linalg.vector_norm(param.grad)**2
    
    total_norm = math.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for param in params:
            if param.grad is not None:
                param.grad.mul_(scale)


def sample_from_dataset(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    data_length = len(dataset)

    rng = np.random.default_rng()
    start_index = rng.integers(0, data_length - context_length - 1, batch_size, endpoint=True)
    result_array = []
    target_array = []
    for index in start_index:
        tmp = dataset[index:index + context_length]
        target = dataset[index + 1 : index + context_length + 1]
        result_array.append(tmp)
        target_array.append(target)
    result_array = np.stack(result_array, axis=0)
    target_array = np.stack(target_array, axis=0)

    return torch.from_numpy(result_array).to(device), torch.from_numpy(target_array).to(device)

def test_cross_entropy_loss():
    logits = torch.tensor([
        [2.0, 1.0, 0.1],
        [0.5, 2.5, -1.0],
    ], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    criterion = CrossEntropyLoss()
    actual = criterion(logits, targets)
    expected = F.cross_entropy(logits, targets)

    print(f"custom loss: {actual.item()}")
    print(f"torch loss : {expected.item()}")
    print(f"close match: {torch.allclose(actual, expected, atol=1e-6)}")


if __name__ == "__main__":
    test_cross_entropy_loss()
