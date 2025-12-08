import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import Dataset
import torch
import json
import os
from PIL import Image

# Load dataset
def resize_image(img_pil, max_size=768):
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

# Load model
base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"


# System prompt
SYSTEM_PROMPT = """
    "你是一個專業的商品計數助手，負責根據圖片進行商品辨識與數量統計。"

    "你必須嚴格遵守以下規則，不可自行解讀或發揮，所有規則具有強制性："

    "【全域輸出規則 (最優先，適用所有模式)】"
    "2. 所有最終答案**必須**放在answer標籤之中 answer標籤為<answer></answer>"
    "3. 除了 think標籤 與 answer標籤 之外，不得輸出任何其他文字。"
    "4. 不可讓 think標籤 與 answer標籤 的內容互相矛盾。"
    "5. 一律只計算「最前排、完整可見」的商品。"
    "6. 一律使用『繁體中文』回答，商品品牌名稱保持原文，不可翻譯。"

    "【模式判斷 (第二優先）】"
    "7. 若使用者輸入『請進行商品盤點』，進入【全商品盤點模式】。"
    "8. 若使用者輸入非『請進行商品盤點』文字，進入【指定商品統計模式】。"

    "【全商品盤點模式】"
    "9. 列出圖片中『所有可辨識的商品』與『對應顏色』與『數量』。"
    "10. answer標籤之中的格式必須為："
    "   商品名稱(顏色):數量,商品名稱(顏色):數量"
    "11. 範例：ALISA橄欖罐頭(白):4,醃漬罐頭(紅):2,醃漬罐頭(黃):2"
    "12. 每一筆結果必須用半形逗號 , 隔開，不得換行，不得有多餘空格。"
    "13. 不可猜測、臆測、補全畫面中不存在的商品、顏色或數量。"

    "【指定商品統計模式】"
    "14. 僅統計使用者指定的商品名稱、顏色或外觀特徵。"
    "15. 只要圖片中的商品名稱『包含』使用者輸入的關鍵字即視為符合（英文忽略大小寫）。"
    "16. 關鍵字可以是名稱或外觀特徵，例如：紅、藍、綠、黃、白色瓶蓋、藍色包裝、紫色標籤。"
    "17. 不可將其他品牌或不包含關鍵字的商品納入計算。"
    "18. 在此模式下，answer標籤只能包含『單一阿拉伯數字』，例如：<answer>3</answer>"
    "19. 若圖片中不存在符合條件的商品，必須輸出："
    "   <think>找不到商品</think><answer>0</answer>"
    "20. 只要確認存在符合商品，answer標籤不可為 0。"
"""


class QwenVLM():
    def __init__(self):
        # Load model
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True, padding_side="left")

        print("Loading base model...")
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.base_model.eval()
        print("Base model loaded!")

    def generate(self, image_path, question = "Count all products in the image."):
        img = Image.open(image_path).convert("RGB")
        image = resize_image(img)
        

        # Create conversation
        message = [
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
        prompt = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

        # Generate with BASE MODEL only
        print("\nGenerating response with BASE MODEL...")
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.base_model.device)

        with torch.no_grad():
            output_ids = self.base_model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                # temperature=0.7,
                # top_p=0.95,
            )

        # Decode base model response
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        return response

if __name__ == "__main__":
    qwenVLM = QwenVLM()

    image_list = ["2.jpg", "3.jpg", "4.jpg", "11.jpg", "15.jpg", "101.jpg", "104.jpg"]

    image_list_d = ["2.jpg", "3.jpg", "4.jpg", "11.jpg", "15.jpg"]
    questions = [
        "圖中有幾包沙拉", 
        "圖中有幾盒紅色的餅乾盒",
        "圖中有幾個麵包的標籤是紅色的",
        "圖中有幾罐紅色的醬",
        "圖中有幾個罐頭的蓋子是米色的",
    ]

    image_list_2 =["101.jpg"]

    sum_time_elapsed_1 = 0
    sum_time_elapsed_2 = 0
    for img_name, question in zip(image_list_d, questions):

        image_path = os.path.join(images_dir, img_name)

        # calaulate time elapsed
        print('=' * 20)
        start_time_1 = time.time()
        qwenVLM.generate(image_path, question=question)
        end_time_1 = time.time()
        time_elapsed_1 = end_time_1 - start_time_1
        print(f"Time elapsed: {time_elapsed_1} seconds")
        sum_time_elapsed_1 += time_elapsed_1
        print('=' * 20)

        # print('+' * 20)
        # start_time_2 = time.time()
        # ministralVLM.generate(image_path,system_prompt_mode=2)
        # end_time_2 = time.time()
        # time_elapsed_2 = end_time_2 - start_time_2
        # print(f"Time elapsed: {time_elapsed_2} seconds")
        # sum_time_elapsed_2 += time_elapsed_2
        # print('+' * 20)

    print(f"Average Time elapsed (No Reasoning): {sum_time_elapsed_1/len(image_list)} seconds")
    print(f"Average Time elapsed (With Reasoning): {sum_time_elapsed_2/len(image_list)} seconds")

