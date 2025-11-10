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