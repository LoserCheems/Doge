import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange


class DMHA(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int = 1,
        num_v: int = 4,
        **kwargs
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_v = num_v
        self.Q_proj = nn.Linear(d_model, d_model)
        self.K_proj = nn.Linear(d_model, d_model)
        self.V_queries = nn.Linear(self.d_model, self.head_dim)
        self.V_keys = nn.Parameter(torch.zeros(self.num_v, self.head_dim))
        self.V_embed = nn.Embedding(self.num_v, self.head_dim * self.num_heads)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def compute_value_states(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        V_queries = self.V_queries(x)
        # sim = torch.einsum('b t n, k n -> b t k', V_queries, self.V_keys)
        sim = torch.matmul(V_queries, self.V_keys.T)
        V_embed = self.V_embed(sim.topk(1, dim=-1).indices)
        # y = torch.einsum('b t d, b t k d -> b t d', x, V_embed)
        y = x * V_embed.sum(dim=-2)
        return y

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq_len, d_model = x.size()
        min_type = torch.finfo(x.dtype).min
        Q = self.Q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.K_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.compute_value_states(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.full((seq_len, seq_len), min_type, device=x.device).triu(1)
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
        y = torch.matmul(attn_weights, V)
        y = y.transpose(1, 2).reshape(bsz, seq_len, d_model)
        y = self.out_proj(y)
        return y

        