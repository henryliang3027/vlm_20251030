import base64
import os
import json
import torch
from datasets import Dataset
from io import BytesIO
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from PIL import Image

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported

# Load custom training data from JSON
json_path = "./training_data/main.json"
images_dir = "./training_data/images"

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
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
    ]
    return {"messages": conversation}


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


if __name__ == "__main__":

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Ministral-3-3B-Instruct-2512",
        load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,  # False if not finetuning vision layers
        finetune_language_layers=True,  # False if not finetuning language layers
        finetune_attention_modules=True,  # False if not finetuning attention layers
        finetune_mlp_modules=True,  # False if not finetuning MLP layers
        r=32,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=32,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )

    raw_datalist = get_raw_datalist()
    converted_dataset = [convert_to_conversation(sample) for sample in raw_datalist]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            # warmup_steps=5,
            max_steps=500,
            save_steps=100,
            # num_train_epochs = 1, # Set this instead of max_steps for full training runs
            learning_rate=1e-5,
            logging_steps=1,
            optim="adamw_8bit",
            fp16=not is_bf16_supported(),  # Use fp16 if bf16 is not supported
            bf16=is_bf16_supported(),  # Use bf16 if supported
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs_300",
            report_to="tensorboard",  # For Weights and Biases
            # You MUST put the below items for vision finetuning:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )

    trainer_stats = trainer.train()

    model.save_pretrained("finetune_model")  # Local saving
    tokenizer.save_pretrained("finetune_model")
    model.save_pretrained_merged(
        "finetune_model_merged",
        tokenizer,
    )

    model.save_pretrained_gguf(
        "finetune_model_gguf",
        tokenizer,
        # quantization_method="q4_k_m",
    )
