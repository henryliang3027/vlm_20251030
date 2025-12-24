import os
from PIL import Image
from unsloth import FastVisionModel  # FastLanguageModel for LLMs
import time

SYSTEM_PROMPT_FULL_INVENTORY = """
You are a product counting assistant.

Rules:
1. Count EACH individual product that is fully visible, ignore price tags
2. Products with the same appearance = count together as one item with quantity
3. Answer in English only
4. Output format: BrandOrType(Color)(PackageType):Quantity

BrandOrType naming:
- If you see English text on the product, use it
- If no clear English name, describe what you see
- Keep it simple - use 1-3 words maximum

Package types: Bottle, Can, Box, Pouch

Colors: Use the main packaging color (not liquid color)
- Red, Yellow, Green, Blue, White, Clear, Orange, Pink, Brown, Purple

Example:
Cold Mountain(Blue)(Bottle):2,GREEN TEA(Green)(Box):1

Important:
- Count identical products together (same brand + same color = one item)
- Separate items with commas only
"""

SYSTEM_PROMPT_SINGLE_COUNT = """
You are a product counting assistant.

Task: Single Product Count - Count ONLY the specified product

Rules:
1. Only count the product mentioned in the user's question, ignore price tags
2. Ignore all other products in the image
3. Count products with the same appearance together
4. Answer in English only
5. Output format: BrandOrType(Color)(PackageType):Quantity

BrandOrType naming:
- If you see English text on the product, use it
- If no clear English name, describe what you see
- Keep it simple - use 1-3 words maximum

Package types: Bottle, Can, Box, Pouch

Colors: Use the main packaging color (not liquid color)
- Red, Yellow, Green, Blue, White, Clear, Orange, Pink, Brown, Purple

Example:
User asks: "Count Coca-Cola"
Answer: Coca-Cola(Red)(Bottle):3

User asks: "Count green tea boxes"
Answer: GREEN TEA(Green)(Box):2

Important:
- Only count the specified product
- If the product is not in the image, answer: NotFound:0
"""


images_dir = "./training_data/images"
model_id = "unsloth/Ministral-3-3B-Instruct-2512"


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
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{img_base64}"


class MinistralVLM:
    def __init__(self):

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_id,
            load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
        )

        self.model = FastVisionModel.get_peft_model(
            self.model,
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

        FastVisionModel.for_inference(self.model)  # Enable for inference!

    def generate(
        self,
        image_path,
        question="Count all products in the image.",
        system_prompt_mode=1,
    ):
        img_pil = Image.open(image_path).convert("RGB")
        img_pil = resize_image(img_pil)
        image_url = pil_to_base64_url(img_pil)

        if system_prompt_mode == 1:
            system_prompt = SYSTEM_PROMPT_FULL_INVENTORY
        else:
            system_prompt = SYSTEM_PROMPT_SINGLE_COUNT

        messages = [
            # {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            img_pil,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        # print(123123123)
        # text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        # print(123123123)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )

        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        print(response)

        return response


if __name__ == "__main__":
    ministralVLM = MinistralVLM()

    image_list = ["2.jpg", "3.jpg", "4.jpg", "11.jpg", "15.jpg", "101.jpg", "104.jpg"]
    image_list_2 = ["c93.jpg"]
    image_list_d = ["2.jpg", "3.jpg", "4.jpg", "11.jpg", "15.jpg"]
    questions = [
        "How many salad pouches are there in the image?",
        "How many red cookie boxes are there in the image?",
        "How many bread with red label are there in the image?",
        "How many red salsa jar are there in the image?",
        "How many jar with beige lids are there in the image?",
    ]

    questions_2 = [
        """分類並統計圖中的商品
        輸出格式為
        -商品名稱
        -顏色
        -數量

        例如：
        -商品名稱：麥香綠茶
        -顏色：綠色
        -數量：2瓶

        -商品名稱：蘋果汁
        -顏色：紅色
        -數量：2瓶""",
    ]

    sum_time_elapsed_1 = 0
    sum_time_elapsed_2 = 0
    for img_name, question in zip(image_list_2, questions_2):

        image_path = os.path.join(images_dir, img_name)

        # calaulate time elapsed
        print("=" * 20)
        start_time_1 = time.time()
        ministralVLM.generate(image_path, question=question, system_prompt_mode=2)
        end_time_1 = time.time()
        time_elapsed_1 = end_time_1 - start_time_1
        print(f"Time elapsed: {time_elapsed_1} seconds")
        sum_time_elapsed_1 += time_elapsed_1
        print("=" * 20)

        # print('+' * 20)
        # start_time_2 = time.time()
        # ministralVLM.generate(image_path,system_prompt_mode=2)
        # end_time_2 = time.time()
        # time_elapsed_2 = end_time_2 - start_time_2
        # print(f"Time elapsed: {time_elapsed_2} seconds")
        # sum_time_elapsed_2 += time_elapsed_2
        # print('+' * 20)

    print(
        f"Average Time elapsed (No Reasoning): {sum_time_elapsed_1/len(image_list)} seconds"
    )
    print(
        f"Average Time elapsed (With Reasoning): {sum_time_elapsed_2/len(image_list)} seconds"
    )
