import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        **kwargs
    ) -> None:
        super().__init__()
        in_features, out_features = d_model, d_model
        hidden_features = d_model * hidden_mult
        self.up_proj = nn.Linear(in_features, hidden_features)
        self.act_fn = nn.SiLU()
        self.down_proj = nn.Linear(hidden_features, out_features)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        y = self.up_proj(x)
        y = self.act_fn(y)
        y = self.down_proj(y)
        return y
        