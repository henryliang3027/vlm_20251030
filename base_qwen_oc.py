# convert jpg image to PIL Image
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import re

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

# Ssytem prompt
SYSTEM_PROMPT = (
    "You are an expert at counting objects in images. "
    "Count ALL visible objects carefully, including those that might be partially hidden. "
    "Provide your answer in the format: 'A photo of [NUMBER] [OBJECT_TYPE]'"
)

test_dataset = [
    {"image_path": "test_data/eggs.jpeg", "question": "How many egg cartons are there in the image?", "ground_truth": "4"},
    {"image_path": "test_data/instant_noodles.jpeg", "question": "How many instant noodle cups are there in the image?", "ground_truth": "18"},
]

# load jpg image and convert to PIL Image
test_data_idx = 0
test_dataset = test_dataset[test_data_idx]
image_path = test_dataset["image_path"]
image = Image.open(image_path).convert("RGB")
question = test_dataset["question"]

ground_truth = test_dataset["ground_truth"]

print(f"\nQuestion: {question}")
print(f"Ground Truth: {ground_truth}")

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

# Process inputs
inputs = processor(
    text=[prompt],
    images=[image],
    return_tensors="pt",
    padding=True,
).to(base_model.device)

# Generate with BASE MODEL (no fine-tuning)
print("\nGenerating response with BASE MODEL...")
with torch.no_grad():
    base_output_ids = base_model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,  # Set to False for deterministic output
        # temperature=0.7,  # Remove temperature (only used with do_sample=True)
        # top_p=0.95,       # Remove top_p (only used with do_sample=True)
    )


# Decode base model response
base_generated_ids = base_output_ids[:, inputs.input_ids.shape[1]:]
base_response = processor.batch_decode(base_generated_ids, skip_special_tokens=True)[0]

print("\n" + "="*80)
print("BASE MODEL RESPONSE:")
print("="*80)
print(base_response)
print("="*80)