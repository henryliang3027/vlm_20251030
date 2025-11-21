import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # 使用单一 GPU，避免多 GPU 设备冲突
    trust_remote_code=True  # 使用仓库中的自定义代码，避免版本兼容问题
)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

task_prompt = "<OD>"
inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(model.device, torch.bfloat16)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

image_size = image.size
parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=image_size)

print(parsed_answer)

# import requests
# import os
# import torch
# from PIL import Image
# from transformers import AutoProcessor, AutoModelForCausalLM 


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Florence-2-large",
#     torch_dtype=torch_dtype,
#     trust_remote_code=True,
#     attn_implementation="eager"  # 使用传统 attention 实现，避免 SDPA 错误
# ).to(device)  # 强制使用单个 GPU，避免多 GPU 设备不匹配
# processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# prompt = "<OD>"

# def resize_image(img_pil, max_size=512):
#     """調整圖片大小"""
#     width, height = img_pil.size
#     if width > max_size or height > max_size:
#         if width > height:
#             new_width = max_size
#             new_height = int(height * (max_size / width))
#         else:
#             new_height = max_size
#             new_width = int(width * (max_size / height))
#         return img_pil.resize((new_width, new_height), Image.LANCZOS)
#     return img_pil

# # Set maximum image size to fit within token limits 123
# MAX_IMAGE_SIZE = 768  

# # specify image path
# images_dir = '../training_data/images'

# image_path = os.path.join(images_dir, '67.jpg')
# img = Image.open(image_path).convert("RGB")
# image = resize_image(img, max_size=MAX_IMAGE_SIZE)
# question = "請問前排有幾瓶的瓶蓋是白色的?"

# inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

# generated_ids = model.generate(
#     input_ids=inputs["input_ids"],
#     pixel_values=inputs["pixel_values"],
#     max_new_tokens=1024,
#     num_beams=1,  # 改为 greedy search，避免 beam search 与 attention 实现冲突
#     do_sample=False
# )
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

# parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

# print(parsed_answer)

