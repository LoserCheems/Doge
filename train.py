import os
import argparse
import yaml
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def load_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_tokenizer_and_model(yaml):
    tokenizer = AutoTokenizer.from_pretrained(yaml['tokenizer_path'])
    if yaml['mode'] == 'pretrain':
        if yaml['model_name'] == 'doge':
            from models.configuration_doge import DogeConfig
            from models.modeling_doge import DogeForCausalLM

            config = DogeConfig(**yaml['model_config'])
            config.vocab_size = tokenizer.vocab_size
            model = DogeForCausalLM(config=config)
        elif yaml['model_name'] == 'llama':
            from transformers.models.llama.configuration_llama import LlamaConfig
            from transformers.models.llama.modeling_llama import LlamaForCausalLM

            config = LlamaConfig(**yaml['model_config'])
            config.vocab_size = tokenizer.vocab_size
            model = LlamaForCausalLM(config=config)
        elif yaml['model_name'] == 'mamba2':
            from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
            from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM

            config = Mamba2Config(**yaml['model_config'])
            config.vocab_size = tokenizer.vocab_size
            model = Mamba2ForCausalLM(config=config)
        else:
            raise ValueError(f'Invalid model name: {yaml["model_name"]}')

    elif yaml['mode'] == 'finetune':
        if yaml['model_name'] == 'doge':
            from models.modeling_doge import DogeForCausalLM
            model = DogeForCausalLM.from_pretrained(yaml['pretrained_path'])
        elif yaml['model_name'] == 'llama':
            from transformers.models.llama.modeling_llama import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(yaml['pretrained_path'])
        elif yaml['model_name'] == 'mamba2':
            from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM
            model = Mamba2ForCausalLM.from_pretrained(yaml['pretrained_path'])
        else:
            raise ValueError(f'Invalid model name: {yaml["model_name"]}')
    else:
        raise ValueError(f'Invalid mode: {yaml["mode"]}')
    return tokenizer, model


def get_dataset(yaml):
    dataset = load_from_disk(yaml['dataset_path'])
    dataset["train"] = dataset["train"].select(range(yaml['gradient_accumulation_steps'] * yaml['per_device_train_batch_size'] * yaml['num_train_steps']))
    return dataset


def get_trainer(yaml, tokenizer, model, dataset):
    model_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) // 1000000
    num_tokens = len(dataset['train'][0]['input_ids']) * yaml['gradient_accumulation_steps'] * yaml['per_device_train_batch_size'] * yaml['num_train_steps']

    logger.info(f'在 {num_tokens // 1_000_000_000}B tokens 上训练 {yaml["model_name"]}-{model_num_params}M 模型 {yaml["num_train_epochs"]} 个轮次')
    logger.info(f'Training {yaml["model_name"]}-{model_num_params}M model for {yaml["num_train_epochs"]} epochs on {num_tokens // 1_000_000_000}B tokens')

    logging_dir = f'{yaml["logging_dir"]}/{yaml["model_name"]}-{model_num_params}M'
    output_dir = f'{yaml["output_dir"]}/{yaml["model_name"]}-{model_num_params}M'

    training_args = TrainingArguments(
        # 日志和输出目录
        # Logging and output directory
        logging_dir=logging_dir,
        output_dir=output_dir,

        # 训练参数
        # Training parameters
        do_train=True,
        num_train_epochs=yaml['num_train_epochs'],
        weight_decay=yaml['weight_decay'],
        learning_rate=yaml['learning_rate'],
        warmup_ratio=yaml['warmup_ratio'],
        lr_scheduler_type=yaml['lr_scheduler_type'],
        lr_scheduler_kwargs={'min_lr_rate': yaml['min_lr_rate']},
        gradient_accumulation_steps=yaml['gradient_accumulation_steps'],

        # 评估参数
        # Evaluation parameters
        do_eval=True,
        eval_strategy='steps',
        eval_steps=yaml['eval_steps'],

        # 批大小
        # Batch size
        per_device_train_batch_size=yaml['per_device_train_batch_size'],
        per_device_eval_batch_size=yaml['per_device_eval_batch_size'],

        
        # 保存参数
        # Save parameters
        save_safetensors=yaml['save_safetensors'],
        save_strategy='steps',
        save_steps=yaml['save_steps'],

        # 混合精度
        # Mixed precision
        bf16=yaml['bf16'],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, mlm_probability=0.05
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    return trainer



def main(args):
    # 加载配置文件
    # load configuration file
    yaml = load_yaml(args.config)
    # 加载分词器并初始化模型
    # load tokenizer and initialize model
    tokenizer, model = get_tokenizer_and_model(yaml)
    # 加载数据集
    # load dataset
    dataset = get_dataset(yaml)
    # 初始化训练器
    # initialize trainer
    trainer = get_trainer(yaml, tokenizer, model, dataset)
    # 开始训练
    # start training
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(args)