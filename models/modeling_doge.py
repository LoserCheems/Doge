# coding=utf-8
# Copyright 2024 Jingze Shi and the HuggingFace Inc. team.    All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Doge model."""

import inspect
import math
from einops.layers.torch import Rearrange
import einx
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    AttentionMaskConverter,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
    MoeCausalLMOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    logging,
)

from configuration_doge import DogeConfig


logger = logging.get_logger(__name__)


def _prepare_4d_causal_attention_mask_with_cache_position_and_dynamic_mask(
    attention_mask: torch.Tensor = None,
    dynamic_mask: torch.Tensor = None,
    sequence_length: int = None,
    target_length: int = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
    min_dtype: float = None,
    cache_position: torch.Tensor = None,
    batch_size: int = None,
):
    mask_length = attention_mask.size(-1)
    if dynamic_mask is not None:
        num_heads = dynamic_mask.size(0)
    else:
        num_heads = 1

    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, num_heads, -1, -1)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(-1, num_heads, 1, -1)
            if dynamic_mask is not None:
                dynamic_mask = dynamic_mask[None, :, None, :mask_length].expand(batch_size, -1, 1, -1)
                attention_mask = attention_mask.clone() * dynamic_mask
    
            causal_mask = causal_mask.clone()
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask == 0, min_dtype
            )

    return causal_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这是torch.repeat_interleave(x, dim=1, repeats=n_rep)的等效版本. 隐藏状态从(batch, num_query_key_heads, seqlen, head_dim)变为(batch, num_attention_heads, seqlen, head_dim)

    This is an equivalent version of torch.repeat_interleave(x, dim=1, repeats=n_rep). Hidden states go from (batch, num_query_key_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """
    旋转输入的一半隐藏维度.
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_QK_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, config: Optional[DogeConfig] = None):
        super().__init__()
        self.rope_kwargs = {}
     
        if config.rope_scaling is None:
            self.rope_type = "default"
        else:
            self.rope_type = config.rope_scaling
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DogeAttention(nn.Module):

    def __init__(self, config: DogeConfig, layer_id: Optional[int] = None):
        super().__init__()

        self.config = config
        self.layer_idx = layer_id
        self.dtype = config.torch_dtype

        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads
        self.num_attention_groups = config.num_attention_groups
        self.attention_head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = self.num_attention_heads // self.num_attention_groups
        self.dynamic_value = config.dynamic_value
        self.dynamic_value_num_heads = config.dynamic_value_num_heads

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.attention_head_dim * self.num_attention_heads,
            bias=config.hidden_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.attention_head_dim * self.num_key_value_heads,
            bias=config.hidden_bias,
        )
        if self.dynamic_value:
            self.v_queries = nn.Sequential(
                nn.Linear(
                    self.hidden_size,
                    self.attention_head_dim * self.dynamic_value_num_heads,
                    bias=config.hidden_bias,
                ),
                Rearrange('b t (h n) -> b t h n', h = self.dynamic_value_num_heads)
            )
            self.num_v_keys = int(math.sqrt(self.num_key_value_heads))
            self.v_keys = nn.Parameter(
                torch.zeros(
                    self.dynamic_value_num_heads, 
                    self.num_v_keys, 
                    self.attention_head_dim
                )
            )
            self.v_embed = nn.Embedding(
                self.num_key_value_heads,
                self.attention_head_dim * self.num_key_value_heads
            )
        else:
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.attention_head_dim * self.num_key_value_heads,
                bias=config.hidden_bias,
            )

        self.out_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=config.hidden_bias,
        )
    

    def compute_value_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        v_queries = self.v_queries(hidden_states)
        sim = torch.einsum('b t h n, h k n -> b t h k', v_queries, self.v_keys)
        indices = sim.topk(self.dynamic_value_num_heads * 2, dim=-1).indices
        v_embed = self.v_embed(indices)
        hidden_states = torch.einsum('b t d, b t h k d -> b t d', hidden_states, v_embed)
        return hidden_states


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        if self.dynamic_value:
            value_states = self.compute_value_states(hidden_states)
        else:
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.attention_head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.attention_head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, query_states = apply_QK_rotary_pos_emb(query_states, query_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_attention_groups)
        value_states = repeat_kv(value_states, self.num_attention_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.attention_head_dim)
        
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : seq_len]
            attn_weights = attn_weights + causal_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output, past_key_value


class DogeSdpaAttention(DogeAttention):
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        if self.dynamic_value:
            value_states = self.compute_value_states(hidden_states)
        else:
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_attention_heads, self.attention_head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.attention_head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.attention_head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, query_states = apply_QK_rotary_pos_emb(query_states, query_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = repeat_kv(key_states, self.num_attention_groups)
        value_states = repeat_kv(value_states, self.num_attention_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.size(2)]
        
        is_causal = True if causal_mask is None and seq_len > 1 else False

        attn = F.scaled_dot_product_attention(
            query_states.contiguous(), 
            key_states.contiguous(), 
            value_states.contiguous(), 
            attn_mask=causal_mask, 
            is_causal=is_causal
        )

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(bsz, seq_len, self.hidden_size)
        out = self.out_proj(attn)
        
        return out, past_key_value


ATTENTION_CLASSES = {
    "eager": DogeAttention,
    "sdpa": DogeSdpaAttention,
}


# 门控MLP
# Gate MLP
class DogeGateMLP(nn.Module):
    def __init__(self, config: DogeConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.shared_expert_intermediate_size

        self.gate_proj = nn.Linear(
            self.hidden_dim,
            self.intermediate_dim,
            bias=config.hidden_bias,
        )
        self.up_proj = nn.Linear(
            self.hidden_dim,
            self.intermediate_dim,
            bias=config.hidden_bias,
        )
        self.down_proj = nn.Linear(
            self.intermediate_dim,
            self.hidden_dim,
            bias=config.hidden_bias,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.down_proj(self.up_proj(hidden_states) * self.act_fn(self.gate_proj(hidden_states)))
        return hidden_states


# 交叉领域混合百万专家
# Cross Domain Mixture of Million Experts
class DogeCDMoME(nn.Module):

    def __init__(self, config: DogeConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.shared_expert_intermediate_dim = config.shared_expert_intermediate_size
        self.private_expert_intermediate_dim = config.private_expert_intermediate_size

        self.num_cdmmoe_experts = config.num_cdmmoe_experts
        self.num_cdmmoe_heads = config.num_cdmmoe_heads
        self.num_cdmmoe_experts_per_head = config.num_cdmmoe_experts_per_head

        self.act_fn = ACT2FN[config.hidden_act]
        
        # 共享参数的Up Linear
        # Shared parameter Up Linear
        self.shared_up_proj = nn.Linear(
            self.hidden_dim, 
            self.shared_expert_intermediate_dim, 
            bias=config.hidden_bias, 
        )
        # 共享参数的Down Linear
        # Shared parameter Down Linear
        self.shared_down_proj = nn.Linear(
            self.shared_expert_intermediate_dim,
            self.private_expert_intermediate_dim, 
            bias=config.hidden_bias, 
        )

        # 查询与键
        # Queries and Keys
        self.queries = nn.Sequential(
            nn.Linear(
                self.private_expert_intermediate_dim,
                self.private_expert_intermediate_dim * self.num_cdmmoe_heads,
                bias=False,
            ),
            Rearrange('b t (p h d) -> p b t h d', p = 2, h = self.num_cdmmoe_heads)
        )
        self.num_keys = int(math.sqrt(self.num_cdmmoe_experts))
        self.keys = nn.Parameter(
            torch.zeros(
                self.num_cdmmoe_heads, 
                self.num_keys, 
                2,
                self.private_expert_intermediate_dim // 2
            )
        )

        # 私有专家
        # Private Experts
        self.up_embed = nn.Embedding(
            self.num_cdmmoe_experts,
            self.private_expert_intermediate_dim,
        )
        self.down_embed = nn.Embedding(
            self.num_cdmmoe_experts,
            self.hidden_dim,
        )


    def forward(
        self, 
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 交叉领域
        # Cross-domain
        hidden_states = self.shared_down_proj(self.act_fn(self.shared_up_proj(hidden_states)))
   
        # 查询
        # Queries
        queries = self.queries(hidden_states)
        # 获取与键的相似度
        # Get similarity with keys
        sim = torch.einsum('p b t h d, h k p d -> p b t h k', queries, self.keys)
        # 获取相似度最大的专家分数与索引
        # Get expert scores and indices with the highest similarity
        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.num_cdmmoe_experts_per_head, dim=-1)
        all_scores = einx.add('... i, ... j -> ... (i j)', scores_x, scores_y)
        all_indices = einx.add('... i, ... j -> ... (i j)', indices_x * self.num_keys, indices_y)
        scores, pk_indices = all_scores.topk(self.num_cdmmoe_experts_per_head, dim=-1)
        indices = all_indices.gather(-1, pk_indices)

        # 根据索引获取相关的专家嵌入
        # Get related expert embeddings based on indices
        up_embed = self.up_embed(indices)
        down_embed = self.down_embed(indices)

        # 专家计算
        # Expert computation
        hidden_states = torch.einsum('b t d, b t h k d -> b t h k', hidden_states, up_embed)
        hidden_states = self.act_fn(hidden_states) * scores.softmax(dim=-1)
        hidden_states = torch.einsum('b t h k, b t h k d -> b t d', hidden_states, down_embed)
        return hidden_states


# Attention 层
# Attention Layer
class DogeAttentionLayer(nn.Module):
    def __init__(self, config: DogeConfig, layer_id: Optional[int] = None):
        super().__init__()

        self.in_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = ATTENTION_CLASSES[config.attn_implementation](config, layer_id)
        self.in_ffn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = DogeCDMoME(config)

        self.hidden_dropout = config.hidden_dropout


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # 自注意力
        # Self-attention
        residual = hidden_states
        hidden_states = self.in_attn_layernorm(hidden_states)
        hidden_states, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        self_attn_weights = None
        # 失活
        # Dropout
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # Attn 后的残差连接
        # Residual connection after Attn
        hidden_states = residual + hidden_states

        # 前馈
        # Feed forward
        residual = hidden_states
        hidden_states = self.in_ffn_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        # 失活
        # Dropout
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # 前馈后的残差连接
        # Residual connection after feed forward
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class DogePreTrainedModel(PreTrainedModel):
    config_class = DogeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DogeAttentionLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True


    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Embedding(torch.nn.Module):
    def __init__(self, config: DogeConfig):
        super(Embedding, self).__init__()
        
        self.hidden_size = config.hidden_size
        # 词嵌入(并行).
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            self.hidden_size, 
            padding_idx=config.pad_token_id, 
        )


    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 词嵌入.
        # Word embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        return words_embeddings


class DogeModel(DogePreTrainedModel):

    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.word_embed = Embedding(config)
        self.rotary_emb = RotaryEmbedding(config)
        if config.dynamic_mask:
            self.dynamic_mask = nn.Parameter(torch.round(torch.ones(config.num_attention_heads, config.max_position_embeddings)))
        else:
            self.dynamic_mask = None

        # 添加解码器层
        # Add decoder layers
        self.layers = nn.ModuleList(
            [DogeAttentionLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # 最终的LayerNorm
        # Final LayerNorm
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.word_embed

    def set_input_embeddings(self, value):
        self.word_embed = value
    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检索input_ids和inputs_embeds
        # Retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("你不能同时指定input_ids和inputs_embeds You cannot specify both input_ids and inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True`与梯度检查点不兼容. 设置`use_cache=False`..."

                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.word_embed(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
      
        hidden_states = inputs_embeds

        # 生成位置嵌入以在解码器层之间共享
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 解码器层
        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # 添加来自最后一个解码器层的隐藏状态
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position_and_dynamic_mask(
            attention_mask,
            self.dynamic_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class DogeForCausalLM(DogePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config
        self.model = DogeModel(config)
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False, 
        )

        # 初始化权重并应用最终处理
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embed

    def set_input_embeddings(self, value):
        self.model.word_embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position_and_dynamic_mask(
                attention_mask,
                self.model.dynamic_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                用于计算掩码语言建模损失的标签. 索引应在 `[0, ..., config.vocab_size]` 范围内. 设置为 `-100` 的索引被忽略(掩码), 仅为标签为 `[0, ..., config.vocab_size]` 的标记计算损失.

            num_logits_to_keep (`int` or `None`, `optional`):
                计算最后 `num_logits_to_keep` 个标记的对数. 如果为 `None`, 则计算所有 `input_ids` 的对数. 仅对生成的最后一个标记的对数进行计算, 并且仅为该标记节省内存, 对于长序列来说这是非常重要的.


        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int` or `None`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `None`, calculate logits for all
                `input_ids`. Only last token logits are needed for generation, and calculating them only for that token
                can save memory, which becomes pretty significant for long sequences.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 解码器输出由(dec_features, layer_state, dec_hidden, dec_attn)组成
        # Decoder output consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # Shift 使得 tokens < n 预测 n
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 扁平化 the tokens
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # 开启模型并行
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DogeForSequenceClassification(DogePreTrainedModel):
    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.model = DogeModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()
    
    def get_input_embeddings(self):
        return self.model.word_embed
    
    def set_input_embeddings(self, value):
        self.model.word_embed = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.classifier(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )