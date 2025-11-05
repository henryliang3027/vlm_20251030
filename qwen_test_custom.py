from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from datasets import Dataset
import torch
import json
import os
from PIL import Image

# Load dataset
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

# Prepare dataset with resized images
dataset_list = []
max_image_size = 512  # Set maximum image size to fit within token limits
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
dataset = Dataset.from_list(dataset_list)
print(f"Loaded {len(dataset)} samples from custom training data")

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

print(f"number of samples in train_dataset: {len(train_dataset)}")
print(f"number of samples in test_dataset: {len(test_dataset)}")

# Load model
base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter_path = "./Qwen2.5-VL-3B-Custom2/checkpoint-600"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True, padding_side="left")

print("Loading base model and LoRA adapter...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading LoRA adapter...")
finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
finetuned_model.eval()
print("Fine-tuned model loaded!")

# System prompt
SYSTEM_PROMPT = (
    "You are a professional object-counting assistant. 你是一個專業的商品計數助手。 "
    "The user provides an image and asks you to count all visible products. "
    "使用者提供圖片，請你數出所有可見的商品數量。使用繁體中文回答。"
    "Reasoning must be placed inside <think></think>, and the final numeric answer inside <answer></answer>."
)


image_path = os.path.join(images_dir, '17.jpg')
img = Image.open(image_path).convert("RGB")
image = resize_image(img, max_size=max_image_size)
question = "圖中有幾盒雞蛋？"
# ground_truth = "<think>下排：8個阿Q桶麵泡麵杯可見。總共8個泡麵杯</think> <answer>8</answer>"

print(f"\nQuestion: {question}")
# print(f"Ground Truth: {ground_truth}")

# Create conversation
conversation = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    },
]

# Apply chat template
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

# Generate with FINE-TUNED MODEL only
print("\nGenerating response with FINE-TUNED MODEL...")
inputs_finetuned = processor(
    text=[prompt],
    images=[image],
    return_tensors="pt",
    padding=True,
).to(finetuned_model.device)

with torch.no_grad():
    finetuned_output_ids = finetuned_model.generate(
        **inputs_finetuned,
        max_new_tokens=4096,
        do_sample=False,
        # temperature=0.7,
        # top_p=0.95,
    )

# Decode fine-tuned model response
finetuned_generated_ids = finetuned_output_ids[:, inputs_finetuned.input_ids.shape[1]:]
finetuned_response = processor.batch_decode(finetuned_generated_ids, skip_special_tokens=True)[0]

print("\n" + "="*80)
print("FINE-TUNED MODEL RESPONSE:")
print("="*80)
print(finetuned_response)
# print("="*80)
# print(f"\nGround Truth: {ground_truth}")
# print("="*80)

# # Comparison Summary
# print("\n" + "="*80)
# print("COMPARISON SUMMARY")
# print("="*80)
# print(f"Ground Truth: {ground_truth}")
# print("\nBase Model (Before Training):")
# print(f"  - Uses <think></think> format: {'Yes' if '<think>' in base_response and '</think>' in base_response else 'No'}")
# print(f"  - Uses <answer></answer> format: {'Yes' if '<answer>' in base_response and '</answer>' in base_response else 'No'}")
# print("\nFine-tuned Model (After Training):")
# print(f"  - Uses <think></think> format: {'Yes' if '<think>' in finetuned_response and '</think>' in finetuned_response else 'No'}")
# print(f"  - Uses <answer></answer> format: {'Yes' if '<answer>' in finetuned_response and '</answer>' in finetuned_response else 'No'}")
# print("="*80)

# # Save image
# image.save("test_dataset_0_image.png")
# print("\nTest image saved as: test_dataset_0_image.png")