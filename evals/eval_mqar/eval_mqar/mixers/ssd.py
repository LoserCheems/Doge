import torch
from torch import nn
from einops import rearrange


from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    # Pad seq_len to be multiple of chunk_size
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)
    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)

def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.
    """
    # b t ... -> b (l c) ...
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return rearrange(input_tensor, 'b (l c) h -> b l c h', c=chunk_size)
    else:
        return rearrange(input_tensor, 'b (l c) h d -> b l c h d', c=chunk_size)


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


class SSD(nn.Module):

    def __init__(
        self, 
        d_model: int,
        n_heads: int = 1,
        d_ssm_state: int = 64,
        n_groups: int = 1,
        chunk_len: int = 256,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ssm_state = d_ssm_state
        self.n_groups = n_groups
        self.d_head = d_model // n_heads
        self.chunk_len = chunk_len

        A = torch.arange(1, self.n_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.B_proj = nn.Linear(self.d_model, self.n_groups * self.d_ssm_state)
        self.C_proj = nn.Linear(self.d_model, self.n_groups * self.d_ssm_state)
        self.dt_proj = nn.Linear(self.d_model, self.n_heads)
        self.D = nn.Parameter(torch.ones(self.n_heads))

        self.out_proj = nn.Linear(self.d_model, self.d_model)


    def forward(
        self, 
        x: torch.Tensor,
        **kwargs
    ):
        """
        Notations:
            b - batch size
            t - target sequence length
            s - source sequence length
            d - d_model
            h - n_heads
            p - d_head
            c - n_chunks
            l - chunk_len
            n - d_state
            g - n_groups
        """
        seq_len, dtype = x.size(1), x.dtype
        A = -torch.exp(self.A_log.float())
        B = rearrange(self.B_proj(x), 'b t (g n) -> b t g n', g=self.n_groups, n=self.d_ssm_state)
        B = B.repeat(1, 1, self.n_heads // self.n_groups, 1)
        C = rearrange(self.C_proj(x), 'b t (g n) -> b t g n', g=self.n_groups, n=self.d_ssm_state)
        C = C.repeat(1, 1, self.n_heads // self.n_groups, 1)
        dt = self.dt_proj(x)
        x = rearrange(x, 'b t (h p) -> b t h p', h=self.n_heads)
        dt = nn.functional.softplus(dt)
    
        # pad_size = (self.chunk_len - seq_len % self.chunk_len) % self.chunk_len
        # D_residual = rearrange(self.D, '... -> ... 1') * pad_tensor_by_size(x, pad_size)
        
        # # Discretize x and A
        # x, A = x * rearrange(dt, '... -> ... 1'), A.to(x.dtype) * dt
        
        # # Rearrange into blocks/chunks
        # x, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_len) for t in (x, A, B, C)]

        # # Compute cumulative sum of A
        # A = rearrange(A, 'b c l h -> b h c l', l=self.chunk_len)
        # A_cumsum = torch.cumsum(A, dim=-1)

        # # 1. Compute the output for each intra-chunk (diagonal blocks)
        # # This is the analog of a causal mask
        # L = torch.exp(segment_sum(A))
        
        # # First, contraction of C and B to get G (attention-weights like)
        # G = (rearrange(C, 'b l c h n -> b l c 1 h n') * rearrange(B, 'b l c h n -> b l 1 c h n')).sum(dim=-1) # shape: (b, c, l, s, h)
    
        # # Step 2: Compute M, equivalent to applying attention mask to weights
        # M_intermediate = rearrange(G, '... -> ... 1') * rearrange(L, 'b h c s t -> b c s t h 1')
        # M = M_intermediate.sum(dim=-1)

        # # Step 3: Compute Y_diag (apply to values)
        # Y_diag = (rearrange(M, '... -> ... 1') * rearrange(x, 'b l c h p -> b l 1 c h p')).sum(3)
        
        # # (right term of low-rank factorization of off-diagonal blocks; B terms)
        # decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        # B_decay_contraction = B * rearrange(decay_states, 'b h c l -> b c l h 1')
        # # permute back B * decay states
        # states = (rearrange(B_decay_contraction, 'b c l h s -> b c h l s 1') * rearrange(x, 'b l c h p -> b l h c 1 p')).sum(dim=3).permute(0, 1, 2, 4, 3)
        # previous_states = torch.zeros_like(states[:, :1])
        # states = torch.cat([previous_states, states], dim=1)
        # decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
        # states_permuted = states.permute(0, 2, 1, 3, 4)
        # result = (decay_chunk[..., None, None] * states_permuted[:, :, None, ...]).sum(dim=2)
        # new_states = result.permute(0, 2, 1, 3, 4)
        # states = new_states[:, :-1]

        # # Compute state -> output conversion per chunk
        # # (left term of low-rank factorization of off-diagonal blocks; C terms)
        # # compute Yoff
        # C_times_states = rearrange(C, 'b c l h n -> b c l h 1 n') * rearrange(states, 'b c h p n -> b c 1 h p n')
        # Y_off = (C_times_states.sum(-1) * rearrange(torch.exp(A_cumsum), 'b h c l -> b c l h 1'))
        # # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        # y = rearrange(Y_diag + Y_off, 'b c l h p -> b (c l) h p') + D_residual
        # # Cutting off padded chunks
        # if pad_size > 0:
        #     y = y[:, :seq_len, :, :]
        y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=self.chunk_len, D=self.D)
        y = self.out_proj(rearrange(y, 'b t h p -> b t (h p)').to(dtype))

        return y