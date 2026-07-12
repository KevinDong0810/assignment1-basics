from __future__ import annotations

from typing import Any

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

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
