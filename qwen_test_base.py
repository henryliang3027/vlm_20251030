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


images_dir = './training_data/images'
MAX_IMAGE_SIZE = 512  # Set maximum image size to fit within token limits



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
    "你是一個專業的商品計數助手。\n"
    "使用者會提供圖片並詢問可見商品的數量。\n"
    "推理過程必須放在 <think></think> 裡，最終的數字答案需用阿拉伯數字回答，不包含單位並放在 <answer></answer> 裡。\n"
    "如果使用者使用繁體中文提問，請以繁體中文回答。\n"
    "當問題中同時包含英文品牌名稱與中文語句時，請以繁體中文回答；品牌名稱保持原文不翻譯。\n"
)


image_path = os.path.join(images_dir, '1.jpg')
img = Image.open(image_path).convert("RGB")
image = resize_image(img, max_size=MAX_IMAGE_SIZE)
question = "請問圖中有多少個ALISA罐頭？"

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