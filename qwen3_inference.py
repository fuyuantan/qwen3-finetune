import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# --- 配置 ---
# 需要改动的地方 将此路径改为你实际保存 LoRA 适配器的目录!!!
adapter_path = "./outputs/checkpoint-30"

# 加载模型时的参数应与训练时一致
max_seq_length = 2048
dtype = None
load_in_4bit = True

# --- 加载微调后的模型和 Tokenizer ---
# Unsloth 会自动加载基础模型并应用指定的 LoRA 适配器
print(f"Loading model and tokenizer from adapter path: {adapter_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=adapter_path,  # 从包含适配器的目录加载
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 准备模型进行推理 (合并 LoRA 权重等优化)
FastLanguageModel.for_inference(model)
print("Model prepared for inference.")

# --- 准备输入 ---
messages = [
    {"role": "user", "content": "Solve (x + 2)^2 = 0."}
]

# 使用 Tokenizer 的聊天模板格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # 添加必要的提示符
    enable_thinking=False,  # 开启思考过程 </think>...</think>
)
print("\n--- Formatted Input ---")
print(text)
print("-----------------------\n")

# --- Tokenize 输入并移到 GPU ---
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# --- 设置流式输出 ---
streamer = TextStreamer(tokenizer, skip_prompt=True)  # skip_prompt=True 不会打印输入提示

# --- 模型生成 ---
print("--- Model Generation Output ---")
# 使用 torch.no_grad() 可以节省显存，因为推理不需要计算梯度
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,  # 生成的最大新 Token 数量
        temperature=0.6,  # 控制随机性，较低的值使输出更确定
        top_p=0.95,  # 控制核心采样，保留概率累加到 p 的词汇
        top_k=20,  # 限制只从概率最高的 k 个词汇中采样
        streamer=streamer,  # 使用流式输出
        pad_token_id=tokenizer.eos_token_id
    )
print("\n-----------------------------")
print("Inference finished.")
