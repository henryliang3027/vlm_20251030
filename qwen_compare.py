from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from datasets import load_from_disk
import torch
import re
import json
from tqdm import tqdm

# Load dataset
print("Loading dataset...")
dataset = load_from_disk('./local_dataset')
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_dataset = split_dataset['test']

print(f"Number of test samples: {len(test_dataset)}")

# Load models
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
base_model.eval()

print("Loading fine-tuned model...")
finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
finetuned_model.eval()
print("Models loaded!")

# System prompt
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def extract_answer(response):
    """Extract answer from response between <answer></answer> tags"""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no tags, return the whole response
    return response.strip()

def normalize_answer(answer):
    """Normalize answer for comparison"""
    if answer is None:
        return ""
    # Remove extra whitespace
    answer = ' '.join(str(answer).split())
    # Convert to lowercase
    answer = answer.lower()
    return answer

def is_correct(predicted, ground_truth):
    """Check if predicted answer matches ground truth"""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact match
    if pred_norm == gt_norm:
        return True

    # Check if ground truth is contained in prediction
    if gt_norm in pred_norm:
        return True

    # Check if prediction is contained in ground truth
    if pred_norm in gt_norm:
        return True

    return False

def generate_response(model, processor, image, question):
    """Generate response from a model"""
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

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# extract solution between <answer> tags
def extract_answer_description(a1, a2):
    """
    從 a1 提取答案（如 C），然後在 a2 的 Choices 中找到對應的描述
    """

    # 1. 從 a1 提取答案
    answer_match = re.search(r'<answer>([A-Z])</answer>', a1)
    if not answer_match:
        return a1 # 如果沒有找到答案標籤，返回原始文本
    
    answer = answer_match.group(1)  # 例如: "C"

    
    # 2. 在 a2 中找到 Choices: 後的內容
    choices_match = re.search(r'Choices:\s*(.*)', a2, re.DOTALL)
    if not choices_match:
        return a1  # 如果沒有 Choices，只返回答案字母
    
    
    choices_text = choices_match.group(1)


    # 3. 找到對應答案的描述
    # 匹配格式: "C. Pythagorean theorem"
    pattern = answer +'\.\s*(.*?)(?=\s+[A-Z]\.)'
    description_match = re.search(pattern, choices_text)
    if description_match:
        description = description_match.group(1).strip()

        return f"{answer}. {description}"
    
    return a1  # 如果找不到描述，只返回答案字母



# Statistics
results = []
base_correct_count = 0
finetuned_correct_count = 0
base_wrong_finetuned_correct = []  # Cases where base was wrong but finetuned was correct

print("\nTesting all samples...")
for idx in tqdm(range(len(test_dataset))):
    sample = test_dataset[idx]
    image = sample['image']
    question = sample['problem']
    ground_truth =  extract_answer_description(sample['solution'], question)
    print(f"ground_truth: {ground_truth}")

    # Generate responses
    base_response = generate_response(base_model, processor, image, question)
    finetuned_response = generate_response(finetuned_model, processor, image, question)

    # Extract answers
    base_answer = extract_answer(base_response)
    finetuned_answer = extract_answer(finetuned_response)

    # Check correctness
    base_is_correct = is_correct(base_answer, ground_truth)
    finetuned_is_correct = is_correct(finetuned_answer, ground_truth)

    if base_is_correct:
        base_correct_count += 1
    if finetuned_is_correct:
        finetuned_correct_count += 1

    # Record case where base was wrong but finetuned was correct
    if not base_is_correct and finetuned_is_correct:
        base_wrong_finetuned_correct.append({
            'index': idx,
            'question': question,
            'ground_truth': ground_truth,
            'base_answer': base_answer,
            'finetuned_answer': finetuned_answer,
            'base_response': base_response,
            'finetuned_response': finetuned_response,
        })

    results.append({
        'index': idx,
        'question': question,
        'ground_truth': ground_truth,
        'base_answer': base_answer,
        'finetuned_answer': finetuned_answer,
        'base_correct': base_is_correct,
        'finetuned_correct': finetuned_is_correct,
    })

# Print statistics
print("\n" + "="*80)
print("STATISTICS")
print("="*80)
print(f"Total test samples: {len(test_dataset)}")
print(f"\nBase Model:")
print(f"  Correct: {base_correct_count}/{len(test_dataset)} ({base_correct_count/len(test_dataset)*100:.2f}%)")
print(f"  Wrong: {len(test_dataset)-base_correct_count}/{len(test_dataset)} ({(len(test_dataset)-base_correct_count)/len(test_dataset)*100:.2f}%)")
print(f"\nFine-tuned Model:")
print(f"  Correct: {finetuned_correct_count}/{len(test_dataset)} ({finetuned_correct_count/len(test_dataset)*100:.2f}%)")
print(f"  Wrong: {len(test_dataset)-finetuned_correct_count}/{len(test_dataset)} ({(len(test_dataset)-finetuned_correct_count)/len(test_dataset)*100:.2f}%)")
print(f"\n**Base Wrong � Fine-tuned Correct: {len(base_wrong_finetuned_correct)} cases**")
print("="*80)


# Print details of base wrong but finetuned correct cases
if base_wrong_finetuned_correct:
    print("\n" + "="*80)
    print("DETAILED: CASES WHERE BASE MODEL WAS WRONG BUT FINE-TUNED MODEL WAS CORRECT")
    print("="*80)

    for i, case in enumerate(base_wrong_finetuned_correct, 1):
        print(f"\n{'='*80}")
        print(f"Improvement Case {i}/{len(base_wrong_finetuned_correct)} (Test Index: {case['index']})")
        print(f"{'='*80}")
        print(f"Question: {case['question']}")
        print(f"\nGround Truth: {case['ground_truth']}")
        print(f"\nBase Model Answer: {case['base_answer']}")
        print(f"Fine-tuned Model Answer: {case['finetuned_answer']}")
        print(f"\nBase Model Full Response:")
        print(f"{case['base_response']}")
        print(f"\nFine-tuned Model Full Response:")
        print(f"{case['finetuned_response']}")
        print("="*80)


