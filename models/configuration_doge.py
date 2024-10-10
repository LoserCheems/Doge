from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DogeConfig(PretrainedConfig):
    model_type = "doge"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,

        # 基础配置
        # Basic Configuration
        num_hidden_layers=8,
        vocab_size=32768,
        hidden_size=64,
        hidden_bias=False,
        hidden_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=16384,
        rope_theta=10000.0,
        rope_scaling=None,
  
        # 初始化配置
        # Initialization Configuration
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,

        # Attention 配置
        # Attention Configuration
        num_attention_heads=4,
        num_attention_groups=1,
        attn_implementation="eager",
        dynamic_mask=True,
        dynamic_value=True,
        dynamic_value_num_heads=1,

        # 交叉领域混合百万专家配置
        # Cross Domain Mixture of Million Experts Configuration
        shared_expert_intermediate_size=256,
        private_expert_intermediate_size=64,
        num_cdmmoe_experts=256, 
        num_cdmmoe_heads=1, 
        num_cdmmoe_experts_per_head=2, 
        **kwargs
    ):

        # 基础配置
        # Basic Configuration
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.hidden_bias = hidden_bias
        self.hidden_dropout = hidden_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # 初始化配置
        # Initialization Configuration
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        # Attention 配置
        # Attention Configuration
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.attn_implementation = attn_implementation
        self.dynamic_mask = dynamic_mask
        self.dynamic_value = dynamic_value
        self.dynamic_value_num_heads = dynamic_value_num_heads

        # 交叉领域混合百万专家配置
        # Cross Domain Mixture of Million Experts Configuration
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.private_expert_intermediate_size = private_expert_intermediate_size
        self.num_cdmmoe_experts = num_cdmmoe_experts
        self.num_cdmmoe_heads = num_cdmmoe_heads
        self.num_cdmmoe_experts_per_head = num_cdmmoe_experts_per_head

        super().__init__(
            attn_implementation = attn_implementation,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
