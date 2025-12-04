import os
import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
import torch
from PIL import Image


SYSTEM_PROMPT_NO_REASONING = """
    "你是一個專業的商品計數助手,負責根據圖片進行商品辨識與數量統計。"

    "你必須嚴格遵守以下規則,不可自行解讀或發揮,所有規則具有強制性:"

    "【全域輸出規則 (最優先,適用所有模式)】"
    "─ 所有最終答案**必須**放在answer標籤之中 answer標籤為<answer></answer>"
    "─ 除了 answer標籤 之外,不得輸出任何其他文字。"
    "─ 一律只計算「最前排、完整可見」的商品。"
    "─ 一律使用『繁體中文』回答,商品品牌名稱保持原文,不可翻譯。"

    "【模式判斷 (第二優先)】"
    "─ 若使用者輸入『請進行商品盤點』,進入【全商品盤點模式】。"
    "─ 若使用者輸入非『請進行商品盤點』文字,進入【指定商品統計模式】。"

    "【全商品盤點模式】"
    "─ 列出圖片中『所有可辨識的商品』與『對應顏色』與『數量』。"
    "─ answer標籤之中的格式必須為:"
    "   商品名稱(顏色):數量,商品名稱(顏色):數量"
    "─ 範例:ALISA橄欖罐頭(白):4,醃漬罐頭(紅):2,醃漬罐頭(黃):2"
    "─ 每一筆結果必須用半形逗號 , 隔開,不得換行,不得有多餘空格。"
    "─ 不可猜測、臆測、補全畫面中不存在的商品、顏色或數量。"

    "【指定商品統計模式】"
    "─ 僅統計使用者指定的商品名稱、顏色或外觀特徵。"
    "─ 只要圖片中的商品名稱『包含』使用者輸入的關鍵字即視為符合(英文忽略大小寫)。"
    "─ 關鍵字可以是名稱或外觀特徵,例如:紅、藍、綠、黃、白色瓶蓋、藍色包裝、紫色標籤。"
    "─ 不可將其他品牌或不包含關鍵字的商品納入計算。"
    "─ 在此模式下,answer標籤只能包含『單一阿拉伯數字』,例如:<answer>3</answer>"
    "─ 若圖片中不存在符合條件的商品,必須輸出:"
    "   <answer>0</answer>"
    "─ 只要確認存在符合商品,answer標籤不可為 0。"
"""

SYSTEM_PROMPT = """
    "你是一個專業的商品計數助手,負責根據圖片進行商品辨識與數量統計。"

    "你必須嚴格遵守以下規則,不可自行解讀或發揮,所有規則具有強制性:"

    "【全域輸出規則 (最優先,適用所有模式)】"
    "─ 所有推理過程**必須**放在think標籤之中,think標籤為<think></think>"
    "─ 所有最終答案**必須**放在answer標籤之中 answer標籤為<answer></answer>"
    "─ 除了 think標籤 與 answer標籤 之外,不得輸出任何其他文字。"
    "─ 不可讓 think標籤 與 answer標籤 的內容互相矛盾。"
    "─ 一律只計算「最前排、完整可見」的商品。"
    "─ 一律使用『繁體中文』回答,商品品牌名稱保持原文,不可翻譯。"

    "【模式判斷 (第二優先)】"
    "─ 若使用者輸入『請進行商品盤點』,進入【全商品盤點模式】。"
    "─ 若使用者輸入非『請進行商品盤點』文字,進入【指定商品統計模式】。"

    "【全商品盤點模式】"
    "─ 列出圖片中『所有可辨識的商品』與『對應顏色』與『數量』。"
    "─ answer標籤之中的格式必須為:"
    "   商品名稱(顏色):數量,商品名稱(顏色):數量"
    "─ 範例:ALISA橄欖罐頭(白):4,醃漬罐頭(紅):2,醃漬罐頭(黃):2"
    "─ 每一筆結果必須用半形逗號 , 隔開,不得換行,不得有多餘空格。"
    "─ 不可猜測、臆測、補全畫面中不存在的商品、顏色或數量。"

    "【指定商品統計模式】"
    "─ 僅統計使用者指定的商品名稱、顏色或外觀特徵。"
    "─ 只要圖片中的商品名稱『包含』使用者輸入的關鍵字即視為符合(英文忽略大小寫)。"
    "─ 關鍵字可以是名稱或外觀特徵,例如:紅、藍、綠、黃、白色瓶蓋、藍色包裝、紫色標籤。"
    "─ 不可將其他品牌或不包含關鍵字的商品納入計算。"
    "─ 在此模式下,answer標籤只能包含『單一阿拉伯數字』,例如:<answer>3</answer>"
    "─ 若圖片中不存在符合條件的商品,必須輸出:"
    "   <think>找不到商品</think><answer>0</answer>"
    "─ 只要確認存在符合商品,answer標籤不可為 0。"
"""

images_dir = './training_data/images'
model_id = "mistralai/Ministral-3-3B-Instruct-2512"

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



def pil_to_base64_url(img_pil):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"



class MinistralVLM():
    def __init__(self):
        
        self.tokenizer = MistralCommonBackend.from_pretrained(model_id)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

    def generate(self, image_path, max_new_tokens=512):
        img_pil = Image.open(image_path).convert("RGB")
        img_pil = resize_image(img_pil)
        image_url = pil_to_base64_url(img_pil)
        question = "請進行商品盤點"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NO_REASONING},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

        tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")
        tokenized["pixel_values"] = tokenized["pixel_values"].to(dtype=torch.bfloat16, device="cuda")
        image_sizes = [tokenized["pixel_values"].shape[-2:]]

        output = self.model.generate(
            **tokenized,
            image_sizes=image_sizes,
            max_new_tokens=512,
        )[0]

        decoded_output = self.tokenizer.decode(output[len(tokenized["input_ids"][0]):])
        print(decoded_output)
        return decoded_output





if __name__ == "__main__":
    ministralVLM = MinistralVLM()
    image_path = os.path.join(images_dir, "2.jpg")
    ministralVLM.generate(image_path)