from transformers import AutoProcessor, Mistral3ForConditionalGeneration, MistralCommonBackend,  FineGrainedFP8Config
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from PIL import Image
from reward_functions import format_reward, accuracy_reward, reasoning_reward
from io import BytesIO
import base64
import torch
import re
import json
import os

SYSTEM_PROMPT_NO_REASONING = """
    你是一個專業的商品計數助手,負責根據圖片進行商品辨識與數量統計。

    你必須嚴格遵守以下規則,不可自行解讀或發揮,所有規則具有強制性:

    【全域輸出規則 (最優先,適用所有模式)】
    ─ 所有最終答案**必須**放在answer標籤之中 answer標籤為<answer></answer>
    ─ 除了 answer標籤 之外,不得輸出任何其他文字。
    ─ 一律只計算「最前排、完整可見」的商品。
    ─ 一律使用『繁體中文』回答,商品品牌名稱保持原文,不可翻譯。

    【模式判斷 (第二優先)】
    ─ 若使用者輸入『請進行商品盤點』,進入【全商品盤點模式】。
    ─ 若使用者輸入非『請進行商品盤點』文字,進入【指定商品統計模式】。

    【全商品盤點模式】
    ─ 列出圖片中『所有可辨識的商品名稱』與『對應顏色』與『數量』。
    ─ answer標籤之中的格式必須為:
       商品名稱(顏色):數量,商品名稱(顏色):數量
    ─ 範例:ALISA橄欖罐頭(白):4,醃漬罐頭(紅):2,醃漬罐頭(黃):2
    ─ 每一筆結果必須用半形逗號 , 隔開,不得換行,不得有多餘空格。
    ─ 不可猜測、臆測、補全畫面中不存在的商品、顏色或數量。

    【指定商品統計模式】
    ─ 僅統計使用者指定的商品名稱、顏色或外觀特徵。
    ─ 只要圖片中的商品名稱『包含』使用者輸入的關鍵字即視為符合(英文忽略大小寫)。
    ─ 關鍵字可以是名稱或外觀特徵,例如:紅、藍、綠、黃、白色瓶蓋、藍色包裝、紫色標籤。
    ─ 不可將其他品牌或不包含關鍵字的商品納入計算。
    ─ 在此模式下,answer標籤只能包含『單一阿拉伯數字』,例如:<answer>3</answer>
    ─ 若圖片中不存在符合條件的商品,必須輸出:
       <answer>0</answer>
    ─ 只要確認存在符合商品,answer標籤不可為 0。
"""

def pil_to_base64_url(img_pil):


    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

# Define resize_image function first
def resize_image(img_pil, max_size=512):
    """調整圖片大小"""
    width, height = img_pil.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return img_pil.resize((new_width, new_height), Image.LANCZOS)
    return img_pil

# Load custom training data from JSON
json_path = './training_data/main.json'
images_dir = './training_data/images'

print(f"Loading data from {json_path}...")
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples from JSON file.")

# Prepare dataset with resized images
dataset_list = []
max_image_size = 768  # Set maximum image size to fit within token limits
print(f"Resizing images to max size: {max_image_size}x{max_image_size}")

for item in data:
    image_path = os.path.join(images_dir, item['image'])
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        # Resize image to prevent token overflow
        img = resize_image(img, max_size=max_image_size)
        img_base64_url = pil_to_base64_url(img)
        dataset_list.append({
            'image_url': img_base64_url,
            'problem': item['question'],
            'solution': item['answer']
        })
    else:
        print(f"Warning: Image {image_path} not found, skipping...")

# Create HuggingFace Dataset
train_dataset = Dataset.from_list(dataset_list)
print(f"Loaded {len(train_dataset)} samples from custom training data")
print(f"Dataset columns: {train_dataset.column_names}")

# split_dataset = dataset.train_test_split(test_size=0, seed=42)

# print(f"Dataset columns2: {dataset.column_names}")

# train_dataset = split_dataset['train']
# test_dataset = split_dataset['test']

print(f"number of samples in train_dataset: {len(train_dataset)}")
# print(f"number of samples in test_dataset: {len(test_dataset)}")

model_id = "mistralai/Ministral-3-3B-Instruct-2512"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")
model = Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map="auto", quantization_config=FineGrainedFP8Config(dequantize=True))

def make_conversation(examples):
    """Transform function applied on-the-fly to avoid serialization issues with PIL Images"""
    # Handle both batched and non-batched inputs
    is_batched = isinstance(examples["image"], list)
    # print(f"examples column: {examples.keys()}")
    # print(f"examples image: {examples['image'][0]}")
    # print(f"examples problem: {examples['problem'][0]}")

    if is_batched:
        conversations = []
        for image_url, problem in zip(examples["image_url"], examples["problem"]):
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT_NO_REASONING},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": problem},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]
            conversations.append(conversation)
        examples["prompt"] = conversations
    else:
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT_NO_REASONING},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": examples["problem"]},
                    {"type": "image_url", "image_url": {"url": examples["image_url"]}},
                ],
            },
        ]
        examples["prompt"] = conversation

    return examples

print(f"original train_dataset: \n {train_dataset}")

# Remove unnecessary columns (not needed for custom dataset)
# train_dataset = train_dataset.remove_columns(['original_question', 'original_answer'])

# Use set_transform to apply transformation on-the-fly (avoids serialization)
train_dataset.set_transform(make_conversation)

print(f"processed train_dataset: \n {train_dataset}")


# We’ll leverage LoRA for training the model, so let’s configure it
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="checkpoints/Ministral-3-3B-Custom-2512-size-768-3-reward-metrics-10000epochs",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    num_train_epochs=10000,
    bf16=True,

    # Parameters that control the data preprocessing
    per_device_train_batch_size=4,
    max_completion_length=1024, # default: 256
    num_generations=4, # default: 8
    max_prompt_length=4096,  # Increased to accommodate image tokens

    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
)

print("="*50)
print(f"Number of GPUs: {training_args.n_gpu}")
print(f"Per device batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Total batch size: {training_args.train_batch_size}")
print(f"Num generations: {training_args.num_generations}")
print("="*50)

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[format_reward, accuracy_reward, reasoning_reward],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
