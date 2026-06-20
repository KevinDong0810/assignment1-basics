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


class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.d_k = d_k
        position_array = torch.arange(max_seq_len, device=device)
        dimension_array = torch.pow(torch.tensor(theta), -2.0 / d_k * torch.arange(d_k // 2, device=device))
        indices = torch.outer(position_array, dimension_array)

        cos_mat = torch.cos(indices)
        sin_mat = torch.sin(indices)

        self.register_buffer("cos_mat", cos_mat, persistent=False)
        self.register_buffer("sin_mat", sin_mat, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        cos_vector = self.cos_mat[token_positions]  # [..., d_k // 2]
        sin_vector = self.sin_mat[token_positions]  # [..., d_k // 2]

        temp = rearrange(x, "... (h w) -> ... h w", w = 2)
        even = temp[..., 0]
        odd = temp[..., 1]

        even_res = rearrange(torch.stack([cos_vector * even, sin_vector * even], dim = -1), "... h w -> ... (h w)")
        odd_res = rearrange(torch.stack([ -1.0 * sin_vector * odd, cos_vector * odd], dim = -1), "... h w -> ... (h w)")
        result = even_res + odd_res
        return result
    

class SoftMax(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, dim: int):
        max_v = torch.max(x, dim=dim, keepdim=True).values
        v = x - max_v  # automatically broadcasted
        exp_v = torch.exp(v)
        exp_v_sum = torch.sum(exp_v, dim=dim, keepdim=True)
        result = exp_v / exp_v_sum
        return result


class Attention(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.softmax_func = SoftMax()
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
        score = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (Q.shape[-1] ** 0.5)
        if mask is not None:
            mask_score = torch.zeros_like(score).masked_fill(~mask, float('-inf'))
            score = score + mask_score
        softmax_score = self.softmax_func(score, -1)

        result = einsum(softmax_score, V, "... queries keys, ...  keys d_v -> ... queries d_v")
        return result

if __name__ == "__main__":
    import torch
    x = torch.tensor([1.0, 2.0])
    soft_func = SoftMax()
    print(soft_func(x, 0))

