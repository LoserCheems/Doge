import uuid
import numpy as np
from eval_mqar.config import TrainConfig, ModelConfig, DataConfig, DataSegmentConfig, LoggerConfig
from eval_mqar.data.associative_recall import MQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "mqar" + sweep_id


VOCAB_SIZE = 8_192


configs = []
for input_seq_len, num_kv_pairs in [
    (256, 8),
    (512, 32),
    (1024, 128),
    (2048, 512),
]:  
    if input_seq_len == 8192:
        batch_size = 2
    elif input_seq_len == 4096:
        batch_size = 4
    elif input_seq_len == 2048:
        batch_size = 32
    elif input_seq_len == 1024:
        batch_size = 64
    elif input_seq_len == 512:
        batch_size = 128
    elif input_seq_len == 256:
        batch_size = 256
    else:
        batch_size = 1


    factory_kwargs = {
        "num_kv_pairs": num_kv_pairs,
        "train_power_a": 0.01,
        "test_power_a": 0.01,
        "random_non_queries": False
    }

    data = DataConfig(
        train_configs=[MQARConfig(num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        test_configs=[MQARConfig(num_examples=1_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
        seed=123,
        batch_size=batch_size,
        cache_dir="./cache",
    )

    for d_model, lr in [
        (64, 4e-3),
        (128, 2e-3),
        (256, 1e-3),
        (512, 8e-4),
    ]:
            
        MIXERS = {
            "attention": dict(
                name="eval_mqar.mixers.mha.MHA",
                kwargs={
                    "num_heads": 4,
                },
            ),
            "dynamic_attention": dict(
                name="eval_mqar.mixers.dmha.DMHA",
                kwargs={
                    "num_heads": 4,
                    "dynamic_value_num_heads": 2,
                }
            )
        }

        for sequence_mixer in [
            "attention",
            "dynamic_attention",  
        ]:

        
            block_type = "TransformerBlock"

            model = ModelConfig(
                d_model=d_model,
                n_layers=2,
                block_type=block_type,
                max_position_embeddings=input_seq_len,
                vocab_size=VOCAB_SIZE,
                sequence_mixer=MIXERS[sequence_mixer],
            )
            config = TrainConfig(
                model=model,
                data=data,
                learning_rate=lr,
                max_epochs=64,
                gradient_accumulation_steps=1,
                run_id=f"{sequence_mixer}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr}-kv{num_kv_pairs}",
                logger=LoggerConfig(
                    project_name="eval_mqar",
                    entity="jingzeshi-independent-researcher" # 填写你的wandb用户名  Fill in your wandb username
                )

            )
            configs.append(config)