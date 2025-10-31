from huggingface_hub import login
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from typing import Optional
from trl import GRPOConfig, GRPOTrainer

import torch
import re


# login()

# dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
# dataset = load_dataset(dataset_id, split='train[:5%]')

# dataset = load_dataset('lmms-lab/multimodal-open-r1-8k-verified', split='train[:5%]')
# dataset.save_to_disk('./local_dataset')





dataset = load_from_disk('./local_dataset')

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

print(f"number of samples in train_dataset: {len(train_dataset)}")
print(f"number of samples in test_dataset: {len(test_dataset)}")

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(examples):
    """Transform function applied on-the-fly to avoid serialization issues with PIL Images"""
    # Handle both batched and non-batched inputs
    is_batched = isinstance(examples["image"], list)

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

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(['original_question', 'original_answer'])

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

    for completion, sol in zip(completions, solution):
        # Extract text from conversation format
        if isinstance(completion, list) and len(completion) > 0:
            # Get the last message (assistant's response)
            content = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        else:
            content = str(completion)

        # Extract answer from <answer> tags
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            # If no tags found, use the entire content
            answer_text = content.strip()

        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    answer_text,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_text}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(answer_text.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards


# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2.5-VL-3B-Instruct-Thinking",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    num_train_epochs=1,
    bf16=True,

    # Parameters that control the data preprocessing
    per_device_train_batch_size=2,
    max_completion_length=1024, # default: 256
    num_generations=2, # default: 8
    max_prompt_length=2048,

    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# ======================================
# show one of the samples image and text

# import matplotlib.pyplot as plt

# sample = train_dataset[0]
# image = sample['image']
# text = sample['problem']

# plt.figure(figsize=(10, 8))
# plt.imshow(image)
# plt.axis('off')
# plt.title(text)
# plt.savefig('output.png', bbox_inches='tight', dpi=150)
# plt.close()

# print("圖片已儲存為 output.png")