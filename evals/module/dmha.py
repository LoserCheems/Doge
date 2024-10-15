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
        dynamic_value_num_heads: int = 1,
        **kwargs
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dynamic_value_num_heads = dynamic_value_num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_queries = nn.Sequential(
                nn.Linear(
                    self.d_model,
                    self.head_dim * self.dynamic_value_num_heads,
                ),
                Rearrange('b t (h n) -> b t h n', h = self.dynamic_value_num_heads)
            )
        self.num_v_keys = int(math.sqrt(self.num_heads))
        self.v_keys = nn.Parameter(
            torch.zeros(
                self.dynamic_value_num_heads, 
                self.num_v_keys, 
                self.head_dim
            )
        )
        self.v_embed = nn.Embedding(
            self.num_heads,
            self.head_dim * self.num_heads
        )
        self.out_proj = nn.Linear(d_model, d_model)
    
    def compute_value_states(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        v_queries = self.v_queries(x)
        sim = torch.einsum('b t h n, h k n -> b t h k', v_queries, self.v_keys)
        indices = sim.topk(self.dynamic_value_num_heads * 2, dim=-1).indices
        v_embed = self.v_embed(indices)
        x = torch.einsum('b t d, b t h k d -> b t d', x, v_embed)
        return x

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq_len, hidden_size = x.size()
        min_type = torch.finfo(x.dtype).min
        q_states = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_states = self.compute_value_states(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q_states, k_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.full((seq_len, seq_len), min_type, device=x.device).triu(1)
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
        attn_output = torch.matmul(attn_weights, v_states)
        attn_output = attn_output.transpose(1, 2).view(bsz, seq_len, hidden_size)
        attn_output = self.out_proj(attn_output)
        return attn_output

        