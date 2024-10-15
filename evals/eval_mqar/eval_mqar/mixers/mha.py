import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MHA(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int = 4,
        **kwargs
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        bsz, seq_len, hidden_size = x.size()
        min_type = torch.finfo(x.dtype).min
        q_states = self.q_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_states = self.k_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_states = self.v_proj(x).reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q_states, k_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.full((seq_len, seq_len), min_type, device=x.device).triu(1)
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
        attn_output = torch.matmul(attn_weights, v_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, hidden_size)
        attn_output = self.out_proj(attn_output)
        return attn_output

        