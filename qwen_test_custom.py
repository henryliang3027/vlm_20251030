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

# Set maximum image size to fit within token limits 123
MAX_IMAGE_SIZE = 768  

# System prompt
SYSTEM_PROMPT = (
    "你是一個專業的商品計數助手，負責根據圖片進行商品辨識與數量統計。\n\n"

    "你必須嚴格遵守以下規則，不可自行解讀或發揮，所有規則具有強制性：\n\n"

    "【全域輸出規則 (最優先，適用所有模式)】\n"
    "1. 所有推理過程**必須**放在 <think></think> 之中。\n"
    "2. 所有最終答案**必須**放在 <answer></answer> 之中。\n"
    "3. 除了 <think> 與 <answer> 之外，不得輸出任何其他文字。\n"
    "4. 一律只計算「最前排、完整可見」的商品。\n\n"

    "【模式判斷 (第二優先）】\n"
    "5. 若使用者輸入『只包含一個字元：n 或 N』，進入【全商品盤點模式】。\n"
    "6. 若使用者輸入『不是單獨的 n 或 N』，進入【指定商品統計模式】。\n\n"

    "【全商品盤點模式 (n / N）】\n"
    "7. 列出圖片中『所有可辨識的商品』與『對應顏色或明顯外觀特徵』與『數量』。\n"
    "8. <answer> 格式必須為：\n"
    "   商品名稱(顏色/特徵):數量,商品名稱(顏色/特徵):數量\n"
    "9. 範例：ALISA橄欖罐頭(白紫):4,醃漬罐頭(紅):2,醃漬罐頭(黃):2\n"
    "10. 每一筆結果**必須用半形逗號 , 隔開**，不得換行，不得有多餘空格。\n"
    "11. 不可猜測、臆測、補全畫面中不存在的商品、顏色或數量。\n\n"

    "【指定商品統計模式 (有指定關鍵字）】\n"
    "12. 僅統計使用者指定的商品名稱、顏色或外觀特徵。\n"
    "13. 只要圖片中的商品名稱『包含』使用者輸入的關鍵字即視為符合（英文忽略大小寫）。\n"
    "14. 關鍵字可以是名稱或外觀特徵，例如：紅、藍、綠、黃、白色瓶蓋、藍色包裝、紫色標籤。\n"
    "15. 不可將其他品牌或不包含關鍵字的商品納入計算。\n"
    "16. 在此模式下，<answer> 只能包含『單一阿拉伯數字』，例如：3\n\n"

    "【存在性規則 (僅適用指定商品統計模式）】\n"
    "17. 若圖片中不存在符合條件的商品，必須輸出：\n"
    "   <think>找不到商品</think><answer>0</answer>\n"
    "18. 只要確認存在符合商品，<answer> 不可為 0。\n\n"

    "【一致性規則】\n"
    "19. 不可讓 <think> 與 <answer> 的內容互相矛盾。\n\n"

    "【語言規則】\n"
    "20. 一律使用『繁體中文』回答，商品品牌名稱保持原文，不可翻譯。\n"
)



# Load model
base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
# adapter_path = "./checkpoints/Qwen2.5-VL-3B-Custom-size-768-3-reward-metrics-10000epochs/checkpoint-92540"
adapter_path = "../../llm_vision/grpo_push_20251112/training_outputs/3cbea0ef-881d-4c12-9512-a0962fb29439/checkpoint-20690"
# adapter_path = "./checkpoints/Qwen2.5-VL-3B-Custom-size-768-3-reward-metrics-10000epochs/checkpoint-36170"

# specify image path
images_dir = './training_data/images'
# [71,72,73,86,96]
image_path = os.path.join(images_dir, '83.jpg')
img = Image.open(image_path).convert("RGB")
image = resize_image(img, max_size=MAX_IMAGE_SIZE)
question = "n"


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

    print(f"inputs shape: {inputs.input_ids.shape}")
    print(f"inputs ids shape: {inputs.input_ids.shape}")
    print(f"output_ids shape: {output_ids.shape}")
    print(f"generated_ids shape: {generated_ids.shape}")

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