from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from datasets import load_from_disk
import torch

# Load dataset
dataset = load_from_disk('./local_dataset')

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

print(f"Number of test samples: {len(test_dataset)}")

# Load model
base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
adapter_path = "./Qwen2.5-VL-3B-Instruct-Thinking"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True, padding_side="left")

print("Loading base model...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("Model loaded!")

# System prompt
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# Test on test_dataset[0]
print("\n" + "="*80)
print("TESTING ON test_dataset[0]")
print("="*80)

sample = test_dataset[0]
image = sample['image']
question = sample['problem']
ground_truth = sample['solution']

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
).to(model.device)

# Generate
print("\nGenerating response...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

# Decode
generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n" + "="*80)
print("MODEL RESPONSE:")
print("="*80)
print(response)
print("="*80)

# Save image
image.save("test_dataset_0_image.png")
print("\nTest image saved as: test_dataset_0_image.png")