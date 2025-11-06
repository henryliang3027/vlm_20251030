from huggingface_hub import login
from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from typing import Optional
from trl import GRPOConfig, GRPOTrainer
from PIL import Image

import torch
import re
import json
import os


# login()

# dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
# dataset = load_dataset(dataset_id, split='train[:5%]')

# dataset = load_dataset('lmms-lab/multimodal-open-r1-8k-verified', split='train[:5%]')
# dataset.save_to_disk('./local_dataset')



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
        dataset_list.append({
            'image': img,
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

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

SYSTEM_PROMPT = (
    "你是一個專業的商品計數助手。\n"
    "使用者會提供圖片並詢問可見商品的數量。\n"
    "推理過程必須放在 <think></think> 裡，最終的數字答案需用阿拉伯數字回答，不包含單位並放在 <answer></answer> 裡。\n"
    "如果使用者使用繁體中文提問，請以繁體中文回答。\n"
    "當問題中同時包含英文品牌名稱與中文語句時，請以繁體中文回答；品牌名稱保持原文不翻譯。\n"
)

def make_conversation(examples):
    """Transform function applied on-the-fly to avoid serialization issues with PIL Images"""
    # Handle both batched and non-batched inputs
    is_batched = isinstance(examples["image"], list)
    # print(f"examples column: {examples.keys()}")
    # print(f"examples image: {examples['image'][0]}")
    # print(f"examples problem: {examples['problem'][0]}")

    if is_batched:
        conversations = []
        for img, problem in zip(examples["image"], examples["problem"]):
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": problem},
                    ],
                },
            ]
            conversations.append(conversation)
        examples["prompt"] = conversations
    else:
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": examples["image"]},
                    {"type": "text", "text": examples["problem"]},
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



# load base model Qwen/Qwen2.5-VL-3B-Instruct
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

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

# Format Enforcement: Ensures that the generation follows a specific format using <think> </think> <answer> </answer> tags for reasoning.
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.

    Args:
        completions: List of conversations, where each conversation is a list of messages.
                     We need to extract the assistant's response text.
    """
    pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    rewards = []

    for completion in completions:
        # Extract text from conversation format
        if isinstance(completion, list) and len(completion) > 0:
            # Get the last message (assistant's response)
            content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        else:
            content = str(completion)

        # Check if the content matches the expected format
        match = re.search(pattern, content, re.DOTALL)

        # print(f"is match: {bool(match)}")
        rewards.append(1.0 if match else 0.0)

    return rewards


# Solution Accuracy: Verifies whether the solution to the problem is correct, comparing it to the solution column in the dataset.
def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion matches the ground truth.
    - If both gold and prediction are parseable → use math verification.
    - If not parseable → compare as normalized text.

    Args:
        completions: List of conversations, where each conversation is a list of messages.
        solution: List of ground truth solutions.
    """
    rewards = []

    for idx, (completion, sol) in enumerate(zip(completions, solution)):
        # Extract text from conversation format
        if isinstance(completion, list) and len(completion) > 0:
            # Get the last message (assistant's response)
            content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        else:
            content = str(completion)
        # print('='*20)
        # print(f"content{idx}: {content},")
        # print(f"sol{idx}: {sol},")
        # print('='*20)
        # Extract answer from <answer> tags

        pattern = r"<answer>\s*(.*?)\s*</answer>"
        generated_answer_match = re.search(pattern, content, re.DOTALL)
        if generated_answer_match:
            generated_answer_text = generated_answer_match.group(1).strip()
        else:
            # If no tags found, use the entire content
            generated_answer_text = content.strip()

        

        ground_truth_answer_match = re.search(pattern, sol, re.DOTALL)
        if ground_truth_answer_match:
            ground_truth_answer_text = ground_truth_answer_match.group(1).strip()
        else:
            ground_truth_answer_text = sol.strip()

        # Convert the extracted answers to integers for comparison
        try:
            generated_answer_int = int(re.findall(r'\d+', generated_answer_text)[0])
            ground_truth_answer_int = int(re.findall(r'\d+', ground_truth_answer_text)[0])

            # print('='*20)
            # print(f"generated_answer_int: {generated_answer_int}")
            # print(f"ground_truth_answer_int: {ground_truth_answer_int}")
            # print('='*20)

            # Extract answer from <answer> tags

            error = abs(generated_answer_int - ground_truth_answer_int)

            if error == 0:
                reward = 1.0  # 完全正確
            elif error == 1:
                reward = 0.5  # 差1個，給一半分數
            elif error == 2:
                reward = 0.2  # 差2個，給一點分數
            elif error == 3:
                reward = 0.1  # 差3個，給很少分數
            else:
                reward = 0.0  # 差太多
        except (IndexError, ValueError):
            # Fallback to string comparison if conversion fails
            reward = float(generated_answer_text.strip() == ground_truth_answer_text.strip())
            print("No numeric answer found, fallback to string comparison. Generated: {}, Ground Truth: {}".format(generated_answer_text, ground_truth_answer_text))

        # print(f"answer_text: {generated_answer_text}")
        # print(f"ground_truth_answer_text: {ground_truth_answer_text}")
        # print(f"reward: {reward}")


        rewards.append(reward)

    print(f"rewards length: {len(rewards)}")
    return rewards


# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2.5-VL-3B-Custom-size-768-improved-reward-500epochs",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    num_train_epochs=500,
    bf16=True,

    # Parameters that control the data preprocessing
    per_device_train_batch_size=2,
    max_completion_length=1024, # default: 256
    num_generations=2, # default: 8
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
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
