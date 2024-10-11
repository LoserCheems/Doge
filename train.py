import os
import logging
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from transformers import AutoTokenizer
from models.configuration_doge import DogeConfig
from models.modeling_doge import DogeForCausalLM


model_name = 'doge_38M'
logging_dir = f'./logs_{model_name}'

# 设置日志写入文件
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(filename=f'{logging_dir}/log.log', level=logging.INFO)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

tokenizer = AutoTokenizer.from_pretrained('./models')


########################
####### 预训练 #########
########################
# 初始化模型
config = DogeConfig()
config.vocab_size = tokenizer.vocab_size
model = DogeForCausalLM(config=config)
# 宇宙百科用于文本生成训练
dataset = load_from_disk(f'./Datasets/cosmopedia_25600000_2048')
dataset["train"] = dataset["train"].shuffle(seed=233).select(range(1 * 512 * 2000))


########################
######## 微调 ##########
########################
# # 加载预训练权重
# model = DogeForCausalLM.from_pretrained("./results_doge/zh/33M/1-1/checkpoint-10000")
# # 无限指令用于指令对话训练
# dataset = load_from_disk(f'./Datasets/Infinity-Instruct_10000000_2048')
# dataset["train"] = dataset["train"].shuffle(seed=233).select(range(1 * 32 * 20000))

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(model)
logger.info(num_params)
exit()
training_args = TrainingArguments(
    output_dir=f'./results_{model_name}',
    logging_dir=logging_dir,
    logging_steps=100,

    do_train=True,
    num_train_epochs=1,
    # max_steps=5000,

    do_eval=True,
    eval_strategy="steps",
    eval_steps=500,
    
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    
    weight_decay=0.01,
    learning_rate=8e-4, # train: 2e-4 finetune: 2e-5 10M: 8e-4
    warmup_ratio=0.1, # train: 0.1 finetune: 0.0
    lr_scheduler_type='cosine_with_min_lr',
    lr_scheduler_kwargs={'min_lr_rate': 0.1},
    
    save_safetensors=False,
    save_strategy="steps",
    save_steps=500,

    bf16=True,
    max_grad_norm=1.0,
    gradient_accumulation_steps=512, # train: 256 finetune: 32
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.05
)

    
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if __name__ == '__main__':
    
    trainer.train()
