from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments  # 可以考虑直接用 TrainingArguments

MODEL = "unsloth/Qwen3-0.6B"
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    load_in_8bit=False,  # A bit more accurate, uses 2x memory
    full_finetuning=False,  # We have full finetuning now!
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,  # Best to choose alpha = rank or rank*2
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

name = "unsloth/openMathReasoning-mini"
reasoning_dataset = load_dataset(name, split="cot")
# 为什么要进行数据处理
print("一条处理前的数据样本:", reasoning_dataset[0])


# Step 1: 格式化为对话结构
def generate_conversation(examples):
    problems = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        if problem and solution:  # 过滤掉空的问题或答案
            conversations.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ])
    return {"conversations": conversations, }


# 应用格式化，得到包含 'conversations' 列的数据集
conversation_dataset = reasoning_dataset.map(
    generate_conversation,
    batched=True,
    remove_columns=reasoning_dataset.column_names  # 只保留 conversations 列
)
# 过滤掉处理后可能为空的 conversations 列表
conversation_dataset = conversation_dataset.filter(lambda x: len(x['conversations']) > 0)

# Step 2: 应用聊天模板，得到格式化的字符串列表
# 注意：这里直接获取列表，因为 apply_chat_template 需要列表输入
raw_conversations_list = conversation_dataset["conversations"]
formatted_texts = tokenizer.apply_chat_template(
    raw_conversations_list,
    tokenize=False  # 只获取格式化文本
)

# Step 3: 将格式化的字符串列表包装成 Dataset 对象
# 创建一个字典，键是 SFTTrainer 要查找的文本字段名（例如 "text"）
text_dataset_dict = {"text": formatted_texts}
# 从字典创建 Dataset 对象
final_dataset = Dataset.from_dict(text_dataset_dict)

print("一条处理后的数据样本:",
      final_dataset[0])  # 应该输出 {'text': '<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>'}

print("final_dataset 数据量:", len(final_dataset))

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=final_dataset,
    eval_dataset=None,  # Can set up evaluation!
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Use GA to mimic batch size!
        warmup_steps=5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps=30,
        learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",  # Use this for WandB etc
        output_dir="outputs",
    ),
)

# 开始训练
print("\n开始训练... (Starting training with string-returning formatting_func...)")
trainer_stats = trainer.train()

# 打印训练统计信息
print("\n训练完成！(Training finished!)")
print("训练统计信息 (Training Stats):")
print(trainer_stats)
