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

# Set maximum image size to fit within token limits
MAX_IMAGE_SIZE = 768  

# System prompt
SYSTEM_PROMPT = (
    "你是一個專業的商品計數助手。\n"
    "使用者會提供圖片並詢問可見商品的數量。\n"
    "推理過程必須放在 <think></think> 裡，最終的數字答案需用阿拉伯數字回答，不包含單位並放在 <answer></answer> 裡。\n"
    "如果使用者使用繁體中文提問，請以繁體中文回答。\n"
    "當問題中同時包含英文品牌名稱與中文語句時，請以繁體中文回答；品牌名稱保持原文不翻譯。\n"
)



# Load model
base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
# adapter_path = "./checkpoints/Qwen2.5-VL-3B-Custom-size-768-3-reward-metrics-10000epochs/checkpoint-92540"
adapter_path = "./checkpoints/Qwen2.5-VL-3B-Custom-size-768-3-reward-metrics-10000epochs/checkpoint-50240"
# adapter_path = "./checkpoints/Qwen2.5-VL-3B-Custom-size-768-3-reward-metrics-10000epochs/checkpoint-36170"

# specify image path
images_dir = './training_data/images'

image_path = os.path.join(images_dir, '67.jpg')
img = Image.open(image_path).convert("RGB")
image = resize_image(img, max_size=MAX_IMAGE_SIZE)
question = "請問前排有幾瓶的瓶蓋是白色的?"


# Create conversation
CONVERSATION = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    },
]

def get_base_model_response():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True, padding_side="left")

    print("Loading base model and LoRA adapter...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    base_model.eval()
    print("Base model loaded!")



    # Apply chat template
    prompt = processor.apply_chat_template(CONVERSATION, add_generation_prompt=True, tokenize=False)

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


def get_finetuned_model_response():
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


    # Apply chat template
    prompt = processor.apply_chat_template(CONVERSATION, add_generation_prompt=True, tokenize=False)

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


if __name__ == "__main__":
    get_base_model_response()
    get_finetuned_model_response()