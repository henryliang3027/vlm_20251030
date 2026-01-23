import base64
from difflib import SequenceMatcher
import os
import json
import re
from datasets import Dataset
from io import BytesIO
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from PIL import Image

from unsloth.trainer import UnslothVisionDataCollator
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bf16_supported

# Load custom training data from JSON
json_path = "./training_data_synth/main.json"
images_dir = "./training_data_synth/images"

INSTRUCTION = "統計圖中的商品"

SYSTEM_PROMPT = """你是零售貨架的商品統計助手。


輸出格式（CSV）：
商品名稱,顏色,數量

規則：
1. 每個商品一行
2. 使用最短的中文商品名稱
3. 顏色用中文單字
4. 數量只用整數

範例：
冷山茶王,藍,2
飲冰室茶集,綠,1
可口可樂,紅,3"""


def pil_to_base64_url(img_pil):

    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


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


def get_raw_datalist():

    print(f"Loading data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from JSON file.")

    # Prepare dataset with resized images
    dataset_list = []

    for item in data:
        image_path = os.path.join(images_dir, item["image"])
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            # Resize image to prevent token overflow
            # img = resize_image(img, max_size=max_image_size)
            # img_base64_url = pil_to_base64_url(img)
            dataset_list.append(
                {
                    "image": img,
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )
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
    return dataset_list


def convert_to_conversation(sample):
    # Construct the prompt in the desired multi-modal format
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": SYSTEM_PROMPT},  # The text part of the prompt
            ],
        },
    ]

    return {"prompt": prompt, "image": sample["image"], "answer": sample["answer"]}


def inference(model, pil_image):
    FastVisionModel.for_inference(model)  # Enable for inference!

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}],
        },
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        pil_image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    from transformers import TextStreamer

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=1000,
        use_cache=True,
        temperature=1.5,
        min_p=0.1,
    )


class Item:
    def __init__(self, name: str, color: str, count: int):
        self.name = name
        self.color = color
        self.count = count


# 檢查是否符合格式和內容正確性
# 商品名稱,顏色,數量
def _calculate_single_score(generated_answer_text, ground_truth_answer_text) -> float:
    """計算單個樣本的分數"""
    score = 0.0
    sentence_pattern = r"^([^,\n]+),([^,\n]),(\d+)(\n([^,\n]+),([^,\n]),(\d+))*$"

    match = re.search(sentence_pattern, generated_answer_text, re.DOTALL)

    if match:
        # 格式正確獎勵
        score += 0.2
        print(f"1 score: {score}")
        # Extract item-quantity pairs from generated answer
        # Pattern: item(color):count
        item_pattern = r"([^,\n]+),([^,\n]),(\d+)"

        generated_items = []
        for match in re.finditer(item_pattern, generated_answer_text):
            item_name = match.group(1).strip()
            color = match.group(2).strip()
            count = int(match.group(3))

            generated_items.append(Item(item_name, color, count))

        # Extract item-quantity pairs from ground truth
        ground_truth_items = []
        for match in re.finditer(item_pattern, ground_truth_answer_text):
            item_name = match.group(1).strip()
            color = match.group(2).strip()
            count = int(match.group(3))
            ground_truth_items.append(Item(item_name, color, count))

        total_ground_truth_items = len(ground_truth_items)
        total_generated_items = len(generated_items)

        if total_ground_truth_items == total_generated_items:

            # 格式分數 加上 item 總數匹配分數 佔比 0.5

            score += 0.3  # 完全項目種類數量正確獎勵

            print(f"2 score: {score}")

            total_item_score = 0.0

            for generated_item in generated_items:
                generated_item_name = generated_item.name
                max_item_score = (
                    0.0  # 紀錄 generated_item 能夠在 ground_truth_item 中得到的最高分數
                )

                for ground_truth_item in ground_truth_items:
                    item_score = 0.0
                    ground_truth_item_name = ground_truth_item.name
                    sequenceMatcher = SequenceMatcher(
                        None, generated_item_name, ground_truth_item_name
                    )
                    ratio = sequenceMatcher.ratio()
                    if ratio > 0.7:
                        item_score += 0.3
                        # 檢查 generated_item count 是否匹配 ground_truth_item count
                        # 根據數量誤差給分
                        error = abs(generated_item.count - ground_truth_item.count)
                        if error == 0:
                            item_score += 0.6
                        elif error == 1:
                            item_score += 0.2
                        elif error == 2:
                            item_score += 0.1

                        # 額外檢查 generated_item color 是否匹配 ground_truth_item color
                        if generated_item.color == ground_truth_item.color:
                            item_score += 0.1

                        # 如果這次的 item_score 比目前的 max_item_score 高，更新 max_item_score
                        # 最後會匹配到最高分數的 ground_truth_item
                        if item_score > max_item_score:
                            max_item_score = item_score

                total_item_score += max_item_score

            # 這部分是計算最終的 item 分數，並加到總分中 佔比 0.5
            final_item_score = total_item_score / total_ground_truth_items / 2.0
            score += final_item_score

    return score


def inventory_mode_accuracy_score(completions, answer, **kwargs) -> list[float]:
    """
    GRPOTrainer reward function interface.

    Args:
        completions: List of generated text completions
        answer: List of ground truth answers from the dataset
        **kwargs: Additional keyword arguments (prompts, etc.)

    Returns:
        List of reward scores (one per completion)
    """
    rewards = []
    for i, completion in enumerate(completions):
        print(f"Generated completion {i}: {completion}")
        print(f"Ground truth answer {i}: {answer[i]}")
        # Get the ground truth for this sample
        completion_content = completion[0]["content"]
        ground_truth = answer[i]
        score = _calculate_single_score(completion_content, ground_truth)
        rewards.append(score)
    return rewards


if __name__ == "__main__":

    max_seq_length = 16384  # Must be this long for VLMs
    lora_rank = 16  # Larger rank = smarter, but slower

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="unsloth/Qwen3-VL-2B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=False,  # Enable vLLM fast inference
        gpu_memory_utilization=0.8,  # Reduce if out of memory
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # False if not finetuning vision layers
        finetune_language_layers=True,  # False if not finetuning language layers
        finetune_attention_modules=True,  # False if not finetuning attention layers
        finetune_mlp_modules=True,  # False if not finetuning MLP layers
        r=16,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        use_gradient_checkpointing="unsloth",  # Reduces memory usage
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )

    raw_datalist = get_raw_datalist()
    converted_dataset = [convert_to_conversation(sample) for sample in raw_datalist]

    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        log_completions=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=2,  # Decrease if out of memory
        max_prompt_length=1024,
        max_completion_length=1024,
        num_train_epochs=0.5,  # Set to 1 for a full training run
        max_steps=600,
        save_steps=60,
        max_grad_norm=0.1,
        report_to="tensorboard",  # Can use Weights & Biases
        output_dir="outputs_qwen3_grpo_600",
        # Below enables GSPO:
        # importance_sampling_level = "sequence",
        # mask_truncated_completions = False,
        # loss_type = "dr_grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        # Pass the processor to handle multimodal inputs
        processing_class=tokenizer,
        reward_funcs=[
            inventory_mode_accuracy_score,
        ],
        train_dataset=converted_dataset,
    )

    trainer.train()

    model.save_pretrained("finetune_qwen3/seperated")  # Local saving
    tokenizer.save_pretrained("finetune_qwen3/seperated")
    model.save_pretrained_merged(
        "finetune_qwen3/merged",
        tokenizer,
    )
