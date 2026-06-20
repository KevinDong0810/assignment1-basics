import torch
from torch import nn
import numpy as np
from einops import rearrange, einsum, reduce

class Linear(nn.Module):

    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()

        weight = torch.zeros((out_features, in_features), device=device, dtype=dtype)
        std = np.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a = -3 * std, b = 3 * std)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x: torch.Tensor):
        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return y


class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        weight = torch.zeros((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a = -3, b = 3)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        gain = torch.ones((d_model), device=device, dtype=dtype)
        self.gain = nn.Parameter(gain, requires_grad=True)
        self.eps = eps
        
    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (reduce(x**2, "... d_model -> ... 1", "mean") + self.eps).sqrt()
        output = x * self.gain / rms
        return output.to(in_dtype)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = ((8.0 * d_model / 3.0) // 64 ) * 64
        else:
            self.d_ff = d_ff
        weights = torch.zeros((3, self.d_ff, self.d_model), device=device, dtype=dtype)
        nn.init.trunc_normal_(weights, mean=0.0, std=1.0, a = -3, b = 3)
        self.w1 = nn.Parameter(weights[0])
        self.w2 = nn.Parameter(weights[1].transpose(1, 0))
        self.w3 = nn.Parameter(weights[2])
    
    def forward(self, x: torch.Tensor):
        res1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        silu_res = res1 * torch.sigmoid(res1)
        res3 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        input2 = silu_res * res3
        output = einsum(input2, self.w2, "... d_ff, d_model d_ff -> ... d_model")
        return output


if __name__ == "__main__":
    module = RMSNorm(2)
    token_ids = torch.tensor(np.array([1.0, 2.0]), dtype=torch.float32)
    print(module(token_ids))