from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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

print("Loading processor...")
processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True, padding_side="left")

print("Loading base model...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
base_model.eval()
print("Base model loaded!")

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

# Generate with BASE MODEL only
print("\nGenerating response with BASE MODEL...")
inputs = processor(
    text=[prompt],
    images=[image],
    return_tensors="pt",
    padding=True,
).to(base_model.device)

with torch.no_grad():
    output_ids = base_model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=False,
        # temperature=0.7,
        # top_p=0.95,
    )

# Decode base model response
generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n" + "="*80)
print("BASE MODEL RESPONSE:")
print("="*80)
print(response)

# # Analysis Summary
# print("\n" + "="*80)
# print("ANALYSIS SUMMARY")
# print("="*80)
# print(f"Ground Truth: {ground_truth}")
# print("\nBase Model Response:")
# print(f"  - Uses <think></think> format: {'Yes' if '<think>' in response and '</think>' in response else 'No'}")
# print(f"  - Uses <answer></answer> format: {'Yes' if '<answer>' in response and '</answer>' in response else 'No'}")
# print("="*80)

# # Save image
# image.save("test_dataset_0_image.png")
# print("\nTest image saved as: test_dataset_0_image.png")