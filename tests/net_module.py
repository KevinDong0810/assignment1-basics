from typing import Any

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
    

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, device=None):
        super().__init__()
        atten_matrices = torch.zeros((3, d_model, d_model), device=device)
        out_mat = torch.zeros((d_model, d_model), device=device)
        std = np.sqrt(1.0 / d_model)
        nn.init.trunc_normal_(atten_matrices, mean=0.0, std=std, a = -3 * std, b = 3 * std)
        nn.init.trunc_normal_(out_mat, mean=0.0, std=std, a = -3 * std, b = 3 * std)
        self.atten_matrices = nn.Parameter(atten_matrices)
        self.out_mat = nn.Parameter(out_mat)

        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads
        assert d_model == self.h * self.d_k
        self.atten_func = Attention()
        self.rope = None

    def build_rope(self, max_seq_len: int, theta: float):
        self.rope = RoPE(theta, self.d_k, max_seq_len)

    def forward(self, x: torch.Tensor, token_positions=None):
        # x: [..., seq_len, d_model]
        seq_len = x.size()[-2]
        mask = ~torch.triu(torch.ones(seq_len, seq_len).to(dtype=torch.bool), diagonal=1)

        atten_matrices_proc = rearrange(self.atten_matrices, "num (h d_k) d_model -> num h d_k d_model", h = self.h)
        result_matrices = einsum(x, atten_matrices_proc, "... seq_len d_model, num h d_k d_model -> ... num h seq_len d_k")
        q = result_matrices[..., 0, :, :, :]  # result dim: [..., head seq_len d_k]
        k = result_matrices[..., 1, :, :, :]
        v = result_matrices[..., 2, :, :, :]

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        atten_result = self.atten_func(q, k, v, mask)

        stacked_result = rearrange(atten_result, "... h seq_len d_k -> ... seq_len (h d_k)")
        projected_result = einsum(stacked_result, self.out_mat, "... d_model, d_model_2 d_model -> ... d_model_2")

        return projected_result
    

class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len, theta, device=None):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model=d_model, device=device)
        self.rmsnorm2 = RMSNorm(d_model=d_model, device=device)
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, device=device)
        self.mha.build_rope(max_seq_len, theta)

        self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device)
    
    def forward(self, x: torch.Tensor):
        token_positions = torch.arange(x.shape[-2])
        mha_output = self.mha(self.rmsnorm1(x), token_positions) + x
        output = self.swiglu(self.rmsnorm2(mha_output)) + mha_output
        return output
    
    def load_weights(self, weights: dict):
        atten_matrics = torch.stack([
            weights['attn.q_proj.weight'], 
            weights['attn.k_proj.weight'],
            weights['attn.v_proj.weight']], dim=0)
        self.mha.load_state_dict({
            "atten_matrices": atten_matrics,
            "out_mat": weights['attn.output_proj.weight']
        })

        self.rmsnorm1.load_state_dict({
            'gain': weights['ln1.weight']
        })

        self.swiglu.load_state_dict({
            "w1": weights['ffn.w1.weight'],
            "w2": weights['ffn.w2.weight'],
            "w3": weights['ffn.w3.weight']
        })

        self.rmsnorm2.load_state_dict({
            'gain': weights['ln2.weight']
        })       


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta) -> None:
        super().__init__()

        self.emb = Embedding(vocab_size, d_model)
        self.transformer_layers = []
        for _ in range(num_layers):
            transformer_block = TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            self.transformer_layers.append(transformer_block)
        self.rms_norm = RMSNorm(d_model)
        self.output_proj = Linear(d_model, vocab_size)
        self.softmax = SoftMax()

    def forward(self, x:torch.Tensor):
        res = self.emb(x)
        for layer in self.transformer_layers:
           res = layer(res)
        output = self.output_proj(self.rms_norm(res))

        return output

    def load_weights(self, weights):
        self.emb.load_state_dict({"weight": weights['token_embeddings.weight']})
        self.rms_norm.load_state_dict({"gain": weights['ln_final.weight']})
        self.output_proj.load_state_dict({"weight": weights['lm_head.weight']})
        
        for i in range(len(self.transformer_layers)):
            input_weights = {
                "attn.q_proj.weight": weights[f"layers.{i}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{i}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{i}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[f"layers.{i}.attn.output_proj.weight"],
                "ln1.weight": weights[f"layers.{i}.ln1.weight"],
                "ln2.weight": weights[f"layers.{i}.ln2.weight"],
                "ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
                "ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
            }
            self.transformer_layers[i].load_weights(input_weights)

if __name__ == "__main__":
    import torch
    x = torch.tensor([1.0, 2.0])
    soft_func = SoftMax()
    print(soft_func(x, 0))

