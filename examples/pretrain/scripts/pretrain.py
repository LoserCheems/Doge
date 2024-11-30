import os
import logging
from argparse import ArgumentParser

import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from wonderful_matrices.doge.configuration_doge import DogeConfig
from wonderful_matrices.doge.modeling_doge import DogeForCausalLM


if __name__ == '__main__':
    
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config_path', type=str, default='./examples/pretrain/configs/doge_197M.yaml', help='path to yaml config file')
    arg_parser.add_argument('--logging_dir', type=str, default='logs')
    arg_parser.add_argument('--output_dir', type=str, default='results')
    arg_parser.add_argument('--tokenizer_path', type=str, default='./examples/tokenizer', help='path to tokenizer')
    arg_parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="path to checkpoint to resume training")

    args = arg_parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        hyperparameters = yaml.load(f, Loader=yaml.FullLoader)
    
    model_name = args.config_path.split('/')[-1].split('.')[0]
    logging_dir = f'{args.logging_dir}/{model_name}'
    output_dir = f'{args.output_dir}/{model_name}'

    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(filename=f'{logging_dir}/log.log', level=logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    # 加载数据集
    # Load dataset
    dataset = load_from_disk(hyperparameters['training_args']['dataset_path'])
    dataset["train"] = dataset["train"].select(range(hyperparameters['training_args']['per_epoch_max_steps'] * hyperparameters['training_args']['per_device_train_batch_size'] * hyperparameters['training_args']['gradient_accumulation_steps']))

    # 加载模型
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = DogeConfig(
        vocab_size=hyperparameters['model_config']['vocab_size'],
        hidden_size=hyperparameters['model_config']['hidden_size'],
        intermediate_size=hyperparameters['model_config']['intermediate_size'],
        num_hidden_layers=hyperparameters['model_config']['num_hidden_layers'],
        hidden_bias=hyperparameters['model_config']['hidden_bias'],
        hidden_dropout=hyperparameters['model_config']['hidden_dropout'],
        hidden_act=hyperparameters['model_config']['hidden_act'],
        max_position_embeddings=hyperparameters['model_config']['max_position_embeddings'],
        rope_theta=hyperparameters['model_config']['rope_theta'],
        use_cache=hyperparameters['model_config']['use_cache'],
        pad_token_id=hyperparameters['model_config']['pad_token_id'],
        bos_token_id=hyperparameters['model_config']['bos_token_id'],
        eos_token_id=hyperparameters['model_config']['eos_token_id'],
        num_attention_heads=hyperparameters['model_config']['num_attention_heads'],
        num_inner_values=hyperparameters['model_config']['num_inner_values'],
        num_inner_value_heads=hyperparameters['model_config']['num_inner_value_heads'],
        num_value_per_head=hyperparameters['model_config']['num_value_per_head'],
        inner_values_retrieval_size=hyperparameters['model_config']['inner_values_retrieval_size'],
        private_expert_retrieval_size=hyperparameters['model_config']['private_expert_retrieval_size'],
        num_cdmmoe_experts=hyperparameters['model_config']['num_cdmmoe_experts'],
        num_cdmmoe_heads=hyperparameters['model_config']['num_cdmmoe_heads'],
        num_cdmmoe_experts_per_head=hyperparameters['model_config']['num_cdmmoe_experts_per_head'],
    )
    config.vocab_size = tokenizer.vocab_size
    model = DogeForCausalLM(config=config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(model)
    logger.info(f"Model parameters: {num_params}")

    # 训练参数
    training_args = TrainingArguments(
        seed=233,
        logging_dir=logging_dir,
        logging_steps=hyperparameters['training_args']['logging_steps'],
        output_dir=output_dir,

        do_train=True,
        num_train_epochs=hyperparameters['training_args']['num_train_epochs'],
        per_device_train_batch_size=hyperparameters['training_args']['per_device_train_batch_size'],

        do_eval=True,
        eval_strategy="steps",
        eval_steps=hyperparameters['training_args']['eval_steps'],
        per_device_eval_batch_size=hyperparameters['training_args']['per_device_eval_batch_size'],

        learning_rate=hyperparameters['training_args']['learning_rate'],
        warmup_ratio=hyperparameters['training_args']['warmup_ratio'],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={'min_lr_rate': hyperparameters['training_args']['min_lr_rate']},
        weight_decay=hyperparameters['training_args']['weight_decay'],
        
        save_safetensors=True,
        save_strategy="steps",
        save_steps=hyperparameters['training_args']['save_steps'],

        bf16=hyperparameters['training_args']['bf16'],
        max_grad_norm=hyperparameters['training_args']['max_grad_norm'],
        gradient_accumulation_steps=hyperparameters['training_args']['gradient_accumulation_steps'],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0.0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)