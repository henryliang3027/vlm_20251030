from openai import OpenAI
import base64
import json
import os
import shutil
from difflib import SequenceMatcher

# from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from PIL import Image

# lora_model_unsloth = (
#     "training_outputs/f6cb5810-7bd6-4594-a46f-a041710c7486/checkpoint-10000"
# )

# model, tokenizer = FastVisionModel.from_pretrained(
#     model_name=lora_model_unsloth,
#     load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
# )

# FastVisionModel.for_inference(model)

SYSTEM_PROMPT = """你是零售貨架的商品統計助手。

輸出格式（CSV）：
商品名稱,顏色,數量

規則：

1. 每個商品一行
2. 使用最短的中文商品名稱
3. 顏色用中文單字,顏色必須是該商品的主要顏色
4. 數量只用整數

範例：
冷山茶王,藍,2
飲冰室茶集,綠,1
可口可樂,紅,3"""

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="no-key-needed",  # 本地通常不驗證，填任意字串即可
)


def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"


def inference(base64_image, question):
    response = client.chat.completions.create(
        model="ministral_custom",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": base64_image}},
                ],
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# def inference_with_unsloth(pil_image, question):

#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image"},
#                 {"type": "text", "text": question},
#             ],
#         },
#     ]
#     input_text = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = tokenizer(
#         pil_image,
#         input_text,
#         add_special_tokens=False,
#         return_tensors="pt",
#     ).to("cuda")

#     finetuned_output_ids = model.generate(
#         **inputs,
#         max_new_tokens=1000,
#         use_cache=True,
#         temperature=0,
#         min_p=0.1,
#     )

#     # Decode fine-tuned model response
#     finetuned_generated_ids = finetuned_output_ids[:, inputs.input_ids.shape[1] :]
#     response = tokenizer.batch_decode(
#         finetuned_generated_ids, skip_special_tokens=True
#     )[0]

#     return response


def compare_answers(predicted, expected, name_threshold=0.2):
    """
    比較預測答案和標準答案
    以換行符號分割，然後以逗號分割每一行成品名、顏色、數量
    針對品名使用 SequenceMatcher 計算相似度
    顏色和數量需要完全一致
    只有品名、顏色、數量都對才算該品項正確
    """
    # 去除首尾空白並分割
    predicted_lines = [
        line.strip() for line in predicted.strip().split("\n") if line.strip()
    ]
    expected_lines = [
        line.strip() for line in expected.strip().split("\n") if line.strip()
    ]

    print("Predicted Lines:", predicted_lines)
    print("Expected Lines:", expected_lines)

    if len(predicted_lines) != len(expected_lines):
        return False

    # 計算有多少預測行能在期望答案中找到匹配
    match_count = 0
    matched_expected_indices = set()  # 追蹤已匹配的期望行，避免重複匹配

    for predicted_line in predicted_lines:
        # 分割成品名、顏色、數量
        predicted_parts = predicted_line.split(",")
        if len(predicted_parts) != 3:
            print(f"警告: 預測行格式不正確: {predicted_line}")
            continue

        pred_name, pred_color, pred_quantity = [p.strip() for p in predicted_parts]

        # 在期望答案中尋找匹配
        found_match = False
        for idx, expected_line in enumerate(expected_lines):
            # 避免重複匹配同一個期望行
            if idx in matched_expected_indices:
                continue

            # 分割期望行
            expected_parts = expected_line.split(",")
            if len(expected_parts) != 3:
                continue

            exp_name, exp_color, exp_quantity = [p.strip() for p in expected_parts]

            # 計算品名相似度
            name_similarity = SequenceMatcher(None, pred_name, exp_name).ratio()

            # 檢查三個條件：品名相似度 >= threshold，顏色一樣，數量一樣
            if (
                name_similarity >= name_threshold
                and pred_color == exp_color
                and pred_quantity == exp_quantity
            ):
                print(
                    f"✓ 匹配: '{predicted_line}' <-> '{expected_line}' (相似度: {name_similarity:.2f})"
                )
                match_count += 1
                matched_expected_indices.add(idx)
                found_match = True
                break

        if not found_match:
            print(f"✗ 未匹配: '{predicted_line}'")

    print(f"Match Count: {match_count}/{len(expected_lines)}")

    # 只有當所有預測行都能找到匹配時，才算正確
    return match_count == len(expected_lines)


def evaluate():
    """
    評估模型在測試集上的表現
    """
    # 讀取測試數據
    test_json_path = "test_real/test.json"
    images_dir = "test_real/images"

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    total = len(test_data)
    correct = 0
    results = []

    print(f"開始評估 {total} 張圖片...\n")

    for idx, item in enumerate(test_data, 1):
        image_name = item["image"]
        question = item["question"]
        expected_answer = item["answer"]

        # 構建圖片路徑
        image_path = os.path.join(images_dir, image_name)

        # 檢查圖片是否存在
        if not os.path.exists(image_path):
            print(f"[{idx}/{total}] 圖片不存在: {image_name}")
            results.append(
                {"image": image_name, "status": "圖片不存在", "correct": False}
            )
            continue

        # 進行推理
        try:

            base64_image = convert_image_to_base64(image_path)
            predicted_answer = inference(base64_image, question)
            # pil_image = Image.open(image_path).convert("RGB")
            # predicted_answer = inference_with_unsloth(pil_image, question)
            print(image_path)
            print(predicted_answer)

            # 比較答案
            is_correct = compare_answers(predicted_answer, expected_answer)

            if is_correct:
                correct += 1
                status = "✓ 正確"
            else:
                status = "✗ 錯誤"

            print(f"[{idx}/{total}] {image_name}: {status}")
            print("=" * 20)

            results.append(
                {
                    "image": image_name,
                    "predicted": predicted_answer,
                    "expected": expected_answer,
                    "correct": is_correct,
                }
            )

        except Exception as e:
            print(f"[{idx}/{total}] {image_name}: 推理錯誤 - {str(e)}")
            results.append(
                {"image": image_name, "status": f"錯誤: {str(e)}", "correct": False}
            )

    # 計算準確率
    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"評估完成！")
    print(f"總數: {total}")
    print(f"正確: {correct}")
    print(f"錯誤: {total - correct}")
    print(f"準確率: {accuracy:.2f}%")
    print(f"{'='*50}")

    # 保存詳細結果
    output_path = "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total": total,
                    "correct": correct,
                    "incorrect": total - correct,
                    "accuracy": accuracy,
                },
                "details": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n詳細結果已保存至: {output_path}")

    # 顯示一些錯誤案例
    incorrect_cases = [r for r in results if not r.get("correct", False)]
    if incorrect_cases:
        print(f"\n錯誤案例範例（前5個）：")
        for case in incorrect_cases[:5]:
            print(f"\n圖片: {case['image']}")
            if "predicted" in case and "expected" in case:
                print(f"預測:\n{case['predicted']}")
                print(f"標準答案:\n{case['expected']}")
            elif "status" in case:
                print(f"狀態: {case['status']}")

    # # 挑出正確的樣本並整理到 easy/ 資料夾
    # print(f"\n{'='*50}")
    # print("開始整理正確樣本到 easy_q4/ 資料夾...")

    # # 創建 easy/ 和 easy/images/ 資料夾
    # easy_dir = "easy_q4"
    # easy_images_dir = os.path.join(easy_dir, "images")
    # os.makedirs(easy_images_dir, exist_ok=True)

    # # 收集正確的樣本
    # correct_samples = []
    # for idx, item in enumerate(test_data):
    #     result = results[idx]
    #     if result.get("correct", False):
    #         # 複製圖片到 easy/images/
    #         src_image_path = os.path.join(images_dir, item["image"])
    #         dst_image_path = os.path.join(easy_images_dir, item["image"])

    #         if os.path.exists(src_image_path):
    #             shutil.copy2(src_image_path, dst_image_path)
    #             print(f"✓ 複製圖片: {item['image']}")

    #         # 添加到正確樣本列表
    #         correct_samples.append(
    #             {
    #                 "image": item["image"],
    #                 "question": item["question"],
    #                 "answer": item["answer"],
    #             }
    #         )

    # # 保存 easy/main.json
    # easy_json_path = os.path.join(easy_dir, "main.json")
    # with open(easy_json_path, "w", encoding="utf-8") as f:
    #     json.dump(correct_samples, f, ensure_ascii=False, indent=4)

    # print(f"\n正確樣本整理完成！")
    # print(f"總共 {len(correct_samples)} 個正確樣本")
    # print(f"圖片保存至: {easy_images_dir}")
    # print(f"標記文件: {easy_json_path}")
    # print(f"{'='*50}")


if __name__ == "__main__":
    evaluate()
