"""
Qwen2.5-VL 3B 物體計數專用 GRPO 訓練 FastAPI 服務 (簡化版)
專注於提升物體計數的準確性，特別針對遮擋情況的處理
支持上傳包含 main.csv 和 images 資料夾的 ZIP 文件
"""

import csv
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import os
# 關閉 wandb 和 swanlab 追蹤
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["SWANLAB_MODE"] = "disabled"
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
import shutil
import pandas as pd
from enum import Enum
import zipfile
import tempfile
import threading

# 導入訓練相關模組
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
    TrainerCallback
)
import torch
from PIL import Image
import re
import random
import numpy as np
from Qwen2_5_GRPO import Qwen2VLGRPOTrainer
from reward_functions import format_reward, reasoning_reward, accuracy_reward

# 設置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 應用
app = FastAPI(
    title="Qwen2.5-VL GRPO訓練API (簡化版)",
    description="專門用於物體計數任務的視覺語言模型GRPO訓練服務",
    version="1.0.0"
)

# 系統提示詞
SYSTEM_PROMPT = (
    "你是一個專業的商品計數助手。\n"
    "使用者會提供圖片並詢問可見商品的數量。\n"
    "推理過程必須放在 <think></think> 裡，最終的數字答案需用阿拉伯數字回答，不包含單位並放在 <answer></answer> 裡。\n"
    "如果使用者使用繁體中文提問，請以繁體中文回答。\n"
    "當問題中同時包含英文品牌名稱與中文語句時，請以繁體中文回答；品牌名稱保持原文不翻譯。\n"
)

# 目錄設置
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "training_outputs" 
DEFAULT_MODEL_PATH = "/app/models/qwen3b"

# 創建必要目錄
for dir_path in [UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 訓練狀態
class TrainingStatus(str, Enum):
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"



# 配置模型
class TrainingConfig(BaseModel):
    num_train_epochs: int = Field(default=100, ge=10, le=1000)
    learning_rate: float = Field(default=1e-5, gt=0, le=1)
    per_device_train_batch_size: int = Field(default=4, ge=1, le=32)
    num_generations: int = Field(default=4, ge=1, le=20)
    temperature: float = Field(default=0.2, gt=0, le=2.0)
    max_image_size: int = Field(default=768, ge=128, le=2048)
    max_completion_length: int = Field(default=1024, ge=128, le=2048)
    max_prompt_length: int = Field(default=4096, ge=512, le=8192)
    logging_steps: int = Field(default=10, ge=1, le=1000)
    save_steps: int = Field(default=10, ge=1, le=1000)
    seed: int = Field(default=43, ge=0)
    bf16: bool = Field(default=True)
    use_4bit: bool = Field(default=False)

class LoRAConfig(BaseModel):
    r: int = Field(default=8, ge=1, le=512)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.1, ge=0, le=0.5)


class TrainingRequest(BaseModel):
    job_name: str
    base_model_path: str = Field(default=DEFAULT_MODEL_PATH)
    data_file_id: str
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)

class JobStatus(BaseModel):
    job_id: str
    job_name: str
    status: TrainingStatus
    progress: float = 0.0
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    created_at: datetime
    error_message: Optional[str] = None
    output_dir: Optional[str] = None

# 全局變量
training_jobs: Dict[str, JobStatus] = {}
uploaded_files: Dict[str, Dict[str, Any]] = {}
training_processes: Dict[str, bool] = {}

# 核心函數
def set_random_seed(seed=43):
    """設置隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def extract_number_from_answer(text):
    """從答案中提取數字"""
    numbers = re.findall(r'\b(\d+)\b', text.strip())
    return int(numbers[0]) if numbers else None


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


def make_conversation(examples):
    """Transform function applied on-the-fly to avoid serialization issues with PIL Images"""
    # Handle both batched and non-batched inputs
    is_batched = isinstance(examples["image"], list)
    # print(f"examples column: {examples.keys()}")
    # print(f"examples image: {examples['image'][0]}")
    # print(f"examples problem: {examples['problem'][0]}")

    if is_batched:
        conversations = []
        for image, problem in zip(examples["image"], examples["problem"]):
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": problem},
                    ],
                },
            ]
            conversations.append(conversation)
        examples["prompt"] = conversations
    else:
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": examples["image"]},
                    {"type": "text", "text": examples["problem"]},
                ],
            },
        ]
        examples["prompt"] = conversation

    return examples

def process_data(example, max_image_size=512):
    """處理訓練樣本"""
    print(f"max_image_size: {max_image_size}")
    dataset_list = []
    try:
        image_path = example['image']
        question = example['question']
        answer = example['answer']
        
        if not os.path.exists(image_path):
            return dataset_list
        
        img = Image.open(image_path).convert('RGB')

        # Resize image to prevent token overflow
        img = resize_image(img, max_size=max_image_size)
        dataset_list.append({
            'image': image_path,
            'problem': question,
            'solution': answer
        })

    except Exception as e:
        logger.warning(f"處理樣本失敗: {e}")
    
    return dataset_list


def detect_zip_creator_detailed(zip_content: bytes) -> dict:
    """详细检测ZIP创建信息"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            temp_zip.write(zip_content)
            temp_zip_path = temp_zip.name
        
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # macOS 特征检查
                has_macosx_dir = any(f.startswith('__MACOSX/') for f in file_list)
                has_ds_store = any('.DS_Store' in f for f in file_list)
                has_resource_fork = any(f.startswith('._') for f in file_list)
                has_hidden_files = any('/._' in f for f in file_list)
                

                platform = "Unknown"
                
                macos_score = sum([has_macosx_dir, has_ds_store, has_resource_fork, has_hidden_files])
                
                if macos_score > 0:
                    platform = "macOS"
                else :
                    platform = "Windows"
  
                return platform
 
        finally:
            os.unlink(temp_zip_path)
            
    except Exception as e:
        return {"platform": "Unknown", "error": str(e)}


def extract_and_validate_macos_zip(zip_content: bytes, extract_to: str) -> dict:
    """解壓並驗證ZIP文件"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            temp_zip.write(zip_content)
            temp_zip_path = temp_zip.name
        
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"ZIP文件包含的文件和目錄: {file_list}")
                
                # 過濾掉 macOS 的隱藏文件
                filtered_files = []
                for f in file_list:
                    # 跳過 macOS 元數據文件
                    if f.startswith('__MACOSX/'):
                        continue
                    if f.startswith('._'):
                        continue
                    if f.endswith('.DS_Store'):
                        continue
                    if '/._' in f:  # 子目錄中的資源分叉文件
                        continue
                    filtered_files.append(f)
                
                # 檢查必要文件（使用過濾後的列表）
                has_csv = any('main.csv' in f for f in filtered_files)
                has_images = any('images/' in f for f in filtered_files)
                
                if not has_csv or not has_images:
                    return {"success": False, "error": "ZIP必須包含main.csv和images目錄"}
                
                # 只解壓有效文件
                for file_name in filtered_files:
                    if file_name.endswith('/'):  # 跳過目錄
                        continue
                    try:
                        # 檢查文件是否存在於zip中且不是隱藏文件
                        zip_ref.extract(file_name, extract_to)
                        logger.info(f"解壓文件: {file_name}")
                    except Exception as e:
                        logger.warning(f"解壓文件失敗 {file_name}: {e}")
                        continue
                
                # 重新組織文件結構
                csv_path = None
                images_dir = None
                
                for root, dirs, files in os.walk(extract_to):
                    # 清理 macOS 隱藏文件
                    files = [f for f in files if not f.startswith('._') and f != '.DS_Store']
                    
                    if 'main.csv' in files:
                        csv_path = os.path.join(root, 'main.csv')
                    if 'images' in dirs:
                        images_dir = os.path.join(root, 'images')
                
                # 移動到根目錄
                target_csv = os.path.join(extract_to, 'main.csv')
                target_images = os.path.join(extract_to, 'images')
                
                if csv_path and csv_path != target_csv:
                    if os.path.exists(target_csv):
                        os.remove(target_csv)
                    shutil.move(csv_path, target_csv)
                if images_dir and images_dir != target_images:
                    if os.path.exists(target_images):
                        shutil.rmtree(target_images)
                    shutil.move(images_dir, target_images)
                
                
                # 驗證CSV
                df = pd.read_csv(target_csv)
                required_cols = ['image', 'question', 'answer']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    return {"success": False, "error": f"CSV缺少列: {missing_cols}"}
                
                # 更新圖片路徑
                for idx, row in df.iterrows():
                    img_filename = os.path.basename(row['image'])
                    new_path = os.path.join(target_images, img_filename)
                    df.at[idx, 'image'] = new_path
  
                
                df.to_csv(target_csv, index=False)

                
                # 統計有效圖片
                valid_images = sum(1 for _, row in df.iterrows() 
                                 if os.path.exists(row['image']))
                
                return {
                    "success": True,
                    "csv_path": target_csv,
                    "images_dir": target_images,
                    "rows": len(df),
                    "valid_images": valid_images
                }
                
        finally:
            os.unlink(temp_zip_path)
            
    except Exception as e:
        return {"success": False, "error": f"處理失敗: {str(e)}"}


def extract_and_validate_zip(zip_content: bytes, extract_to: str, ) -> dict:
    """解壓並驗證ZIP文件"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            temp_zip.write(zip_content)
            temp_zip_path = temp_zip.name
        
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # 檢查必要文件
                has_csv = any('main.csv' in f for f in file_list)
                has_images = any('images/' in f for f in file_list)
                
                if not has_csv or not has_images:
                    return {"success": False, "error": "ZIP必須包含main.csv和images目錄"}
                
                # 解壓文件
                zip_ref.extractall(extract_to)
                
                # 重新組織文件結構
                csv_path = None
                images_dir = None
                
                for root, dirs, files in os.walk(extract_to):
                    if 'main.csv' in files:
                        csv_path = os.path.join(root, 'main.csv')
                    if 'images' in dirs:
                        images_dir = os.path.join(root, 'images')
                
                # 移動到根目錄
                target_csv = os.path.join(extract_to, 'main.csv')
                target_images = os.path.join(extract_to, 'images')
                
                if csv_path != target_csv:
                    shutil.move(csv_path, target_csv)
                if images_dir != target_images:
                    if os.path.exists(target_images):
                        shutil.rmtree(target_images)
                    shutil.move(images_dir, target_images)
                
                # 驗證CSV
                df = pd.read_csv(target_csv)
                required_cols = ['image', 'question', 'answer']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    return {"success": False, "error": f"CSV缺少列: {missing_cols}"}
                
                # 更新圖片路徑
                for idx, row in df.iterrows():
                    img_filename = os.path.basename(row['image'])
                    new_path = os.path.join(target_images, img_filename)
                    df.at[idx, 'image'] = new_path
                
                df.to_csv(target_csv, index=False)
                
                # 統計有效圖片
                valid_images = sum(1 for _, row in df.iterrows() 
                                 if os.path.exists(row['image']))
                
                return {
                    "success": True,
                    "csv_path": target_csv,
                    "images_dir": target_images,
                    "rows": len(df),
                    "valid_images": valid_images
                }
                
        finally:
            os.unlink(temp_zip_path)
            
    except Exception as e:
        return {"success": False, "error": f"處理失敗: {str(e)}"}

class TrainingProgressCallback(TrainerCallback):
    """訓練進度回調"""
    def __init__(self, job_id):
        self.job_id = job_id
        
    def on_train_begin(self, args, state, control, **kwargs):
        # 檢查是否在訓練開始時就被取消
        if self.job_id in training_processes and not training_processes[self.job_id]:
            control.should_training_stop = True
            if self.job_id in training_jobs:
                training_jobs[self.job_id].status = TrainingStatus.CANCELLED
            return control
            
        if self.job_id in training_jobs:
            training_jobs[self.job_id].status = TrainingStatus.TRAINING
            if state.max_steps:
                training_jobs[self.job_id].total_steps = state.max_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        # 每步檢查是否被取消
        if self.job_id in training_processes and not training_processes[self.job_id]:
            control.should_training_stop = True
            if self.job_id in training_jobs:
                training_jobs[self.job_id].status = TrainingStatus.CANCELLED
            return control
            
        if self.job_id in training_jobs:
            job = training_jobs[self.job_id]
            if state.global_step:
                job.current_step = state.global_step
            if state.max_steps:
                job.total_steps = state.max_steps
                job.progress = (state.global_step / state.max_steps) * 100
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        # 日誌時也檢查取消狀態
        print("on_log called with logs:", logs)
        if self.job_id in training_processes and not training_processes[self.job_id]:
            control.should_training_stop = True
            if self.job_id in training_jobs:
                training_jobs[self.job_id].status = TrainingStatus.CANCELLED
            return control
            
        if self.job_id in training_jobs and logs:
            job = training_jobs[self.job_id]
            if 'loss' in logs:
                job.loss = logs['loss']

def train_model_background(job_id: str, training_request: TrainingRequest, csv_path: str):
    """後台訓練函數"""
    def run_training():
        try:
            # 檢查是否已被取消
            if job_id in training_processes and not training_processes[job_id]:
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            # 更新狀態
            if job_id in training_jobs:
                training_jobs[job_id].status = TrainingStatus.TRAINING
            
            # 創建輸出目錄
            output_dir = os.path.join(OUTPUT_DIR, job_id)
            os.makedirs(output_dir, exist_ok=True)
            training_jobs[job_id].output_dir = output_dir

            # 徹底清理GPU狀態（修復第3次訓練失敗問題）
            if torch.cuda.is_available():
                # 強制垃圾回收
                import gc
                gc.collect()

                # 清理所有GPU的緩存
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        # 重置記憶體統計（影響device_map="auto"的決策）
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()

                torch.cuda.synchronize()
                logger.info(f"GPU cache cleared for {torch.cuda.device_count()} GPUs")
            
            # 設置隨機種子
            set_random_seed(training_request.training_config.seed)
            
            # 處理數據
            ds = load_dataset("csv", data_files=csv_path, split="train")
            processed_data = []
            
            for i, example in enumerate(ds):
                # 每處理一些樣本就檢查是否被取消
                if i % 10 == 0 and job_id in training_processes and not training_processes[job_id]:
                    if job_id in training_jobs:
                        training_jobs[job_id].status = TrainingStatus.CANCELLED
                    return
                    
                if os.path.exists(example['image']):
                    sample_data = process_data(example, training_request.training_config.max_image_size)
                    processed_data.extend(sample_data)
            
            if not processed_data:
                raise ValueError("沒有有效的訓練樣本")
            
            print(f"已處理完成 {len(processed_data)} 個樣本")
            
            # 再次檢查是否被取消
            if job_id in training_processes and not training_processes[job_id]:
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            train_dataset = Dataset.from_list(processed_data)
            train_dataset.set_transform(make_conversation)
            
            # 載入模型前檢查取消狀態
            if job_id in training_processes and not training_processes[job_id]:
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            # 載入模型
            model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")
            
            

            
            model_kwargs = {
                'pretrained_model_name_or_path': model_id,
                'dtype': torch.bfloat16,
                'device_map': "auto",
            }
            
            if training_request.training_config.use_4bit:
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                **model_kwargs
            )

            
            model.config.use_cache = False
            # model.gradient_checkpointing_enable()
            
            if training_request.training_config.use_4bit:
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            
            # 應用LoRA
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=training_request.lora_config.r,
                lora_alpha=training_request.lora_config.lora_alpha,
                lora_dropout=training_request.lora_config.lora_dropout,
                target_modules=["q_proj", "v_proj",],
            )
            
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            
            # 訓練前最後檢查
            if job_id in training_processes and not training_processes[job_id]:
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            # 訓練配置
            training_args = GRPOConfig(
                output_dir=output_dir,
                num_train_epochs=training_request.training_config.num_train_epochs,
                learning_rate=training_request.training_config.learning_rate,
                per_device_train_batch_size=training_request.training_config.per_device_train_batch_size,
                num_generations=training_request.training_config.num_generations,
                temperature=training_request.training_config.temperature,
                max_completion_length=training_request.training_config.max_completion_length,
                max_prompt_length=training_request.training_config.max_prompt_length,
                logging_steps=training_request.training_config.logging_steps,
                save_steps=training_request.training_config.save_steps,
                bf16=training_request.training_config.bf16,
                remove_unused_columns=False,
                report_to=["tensorboard"],
                save_strategy="steps",
            )




            
            # 創建訓練器
            trainer = GRPOTrainer(
                model=model,
                processing_class=processor,
                reward_funcs=[format_reward, accuracy_reward, reasoning_reward],
                args=training_args,
                train_dataset=train_dataset,
            )

            # 另一個自定義訓練器選項
            # trainer = Qwen2VLGRPOTrainer(
            #     model=model,
            #     processing_class=processor,
            #     reward_funcs=[format_reward, reasoning_reward, accuracy_reward],
            #     args=training_args,
            #     train_dataset=train_dataset,
            # )
            
            # 添加回調
            trainer.add_callback(TrainingProgressCallback(job_id))
            
            # 開始訓練
            try:
                trainer.train()
            except KeyboardInterrupt:
                # 處理中斷
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            # 訓練後檢查是否被取消（以防在訓練最後階段被取消）
            if job_id in training_processes and not training_processes[job_id]:
                if job_id in training_jobs:
                    training_jobs[job_id].status = TrainingStatus.CANCELLED
                return
            
            # 保存模型
            trainer.save_model(output_dir)
            processor.save_pretrained(output_dir)
            
            # 完成
            if job_id in training_jobs:
                training_jobs[job_id].status = TrainingStatus.COMPLETED
                training_jobs[job_id].progress = 100.0
            
        except Exception as e:
            if job_id in training_jobs:
                training_jobs[job_id].status = TrainingStatus.FAILED
                training_jobs[job_id].error_message = str(e)
            logger.error(f"訓練失敗: {e}")
    
    # 在新線程中運行
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

# API 端點
@app.get("/data-format")
async def get_data_format_guide():
    """獲取訓練數據格式說明"""
    return {
        "zip_structure": {
            "required_files": {
                "main.csv": "主要數據文件，包含 image, question, answer 三個欄位",
                "images/": "圖片目錄，包含所有訓練圖片"
            },
            "example_structure": {
                "training_data.zip": {
                    "main.csv": "CSV數據文件",
                    "images/": {
                        "apple1.jpg": "圖片文件",
                        "apple2.png": "圖片文件",
                        "orange1.jpg": "圖片文件"
                    }
                }
            }
        },
        "csv_format": {
            "required_columns": [
                {
                    "name": "image",
                    "description": "圖片檔案名稱",
                    "example": "apple1.jpg"
                },
                {
                    "name": "question", 
                    "description": "對圖片的計數問題",
                    "example": "How many apples are in this image?"
                },
                {
                    "name": "answer",
                    "description": "標準答案",
                    "example": "A photo of 3 apples"
                }
            ],
            "csv_example": {
                "headers": ["image", "question", "answer"],
                "sample_rows": [
                    ["apple1.jpg", "How many apples are in this image?", "A photo of 3 apples"],
                    ["apple2.png", "Count the apples in the picture", "A photo of 5 apples"],
                    ["orange1.jpg", "How many oranges do you see?", "A photo of 2 oranges"]
                ]
            }
        },
        "image_requirements": {
            "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".gif"],
            "recommended_size": "224x224 到 1024x1024 像素",
            "max_file_size": "建議每張圖片小於 5MB",
            "naming": "使用英文字母和數字，避免特殊字符"
        },
        "answer_format": {
            "recommended_pattern": "A photo of [數量] [物體名稱]",
            "correct_examples": [
                "A photo of 3 apples",
                "A photo of 5 cars", 
                "A photo of 1 cat"
            ],
            "avoid": [
                "There are 3 apples",
                "3",
                "Three apples"
            ]
        },
        "tips": [
            "確保 CSV 中的 image 與 images 目錄中的檔案名對應",
            "建議包含不同數量的樣本以平衡數據",
            "可以包含部分遮擋的物體，系統會學習處理",
            "使用 /upload-data/validate 端點可以在上傳前驗證格式"
        ]
    }

@app.get("/")
async def root():
    return {
        "message": "Qwen2.5-VL GRPO訓練API (簡化版)",
        "version": "1.0.0", 
        "gpu_available": torch.cuda.is_available(),
        "endpoints": {
            "data_format": "/data-format - 獲取數據格式說明",
            "upload": "/upload-data - 上傳ZIP訓練數據包",
            "files": "/files - 查看已上傳文件",
            "train": "/train - 開始訓練, 使用Qwen2.5VL-3B 下載網址 https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main",
            "jobs": "/jobs - 查看訓練任務"
        }
    }

@app.post("/upload-data")
async def upload_training_data(file: UploadFile = File(...)):
    """上傳ZIP訓練數據包"""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="只支持ZIP格式")
    
    file_id = str(uuid.uuid4())
    extract_dir = os.path.join(UPLOAD_DIR, file_id)
    
    try:
        zip_content = await file.read()
        os.makedirs(extract_dir, exist_ok=True)

        platform = detect_zip_creator_detailed(zip_content)
        logger.info(f"檢測到ZIP創建平台: {platform}")
        

        # 根據平台選擇處理方式
        if platform == "macOS":
            validation_result = extract_and_validate_macos_zip(zip_content, extract_dir)
        else:
            # 視為 Windows
            validation_result = extract_and_validate_zip(zip_content, extract_dir)
            
        if not validation_result["success"]:
            shutil.rmtree(extract_dir)
            raise HTTPException(status_code=400, detail=validation_result["error"])
        
        uploaded_files[file_id] = {
            "filename": file.filename,
            "csv_path": validation_result["csv_path"],
            "images_dir": validation_result["images_dir"],
            "upload_time": datetime.now(),
            "rows": validation_result["rows"],
            "valid_images": validation_result["valid_images"]
        }
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "upload_time": datetime.now(),
            "csv_rows": validation_result["rows"],
            "valid_images": validation_result["valid_images"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        raise HTTPException(status_code=500, detail=f"處理失敗: {str(e)}")

@app.get("/files")
async def list_uploaded_files():
    """獲取已上傳文件列表"""
    return {"files": [
        {
            "file_id": file_id,
            "filename": info["filename"],
            "upload_time": info["upload_time"],
            "csv_rows": info["rows"],
            "valid_images": info["valid_images"]
        }
        for file_id, info in uploaded_files.items()
    ]}

@app.post("/train")
async def start_training(training_request: TrainingRequest, background_tasks: BackgroundTasks):
    """開始訓練\n
    base_model_path:使用Qwen2.5VL-3B\n 
    下載網址 https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main
    """
    if training_request.data_file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="數據文件不存在")
    
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="GPU不可用")
    
    file_info = uploaded_files[training_request.data_file_id]
    csv_path = file_info["csv_path"]
    
    job_id = str(uuid.uuid4())
    training_jobs[job_id] = JobStatus(
        job_id=job_id,
        job_name=training_request.job_name,
        status=TrainingStatus.PENDING,
        created_at=datetime.now()
    )
    
    training_processes[job_id] = True
    background_tasks.add_task(train_model_background, job_id, training_request, csv_path)
    
    return {
        "job_id": job_id,
        "message": "訓練已啟動",
        "status": "started"
    }

@app.get("/jobs")
async def list_jobs():
    """獲取所有訓練任務"""
    return {"jobs": list(training_jobs.values())}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """獲取訓練狀態"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="任務不存在")
    return training_jobs[job_id]


@app.get("/jobs/{job_id}/download")
async def download_model(job_id: str):
    """下載訓練完成的模型"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    job = training_jobs[job_id]
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="訓練未完成")
    
    output_dir = job.output_dir
    if not output_dir or not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail="模型文件不存在")
    
    # 創建zip文件
    zip_path = os.path.join(OUTPUT_DIR, f"{job_id}_model.zip")
    shutil.make_archive(zip_path[:-4], 'zip', output_dir)
    
    return FileResponse(
        zip_path,
        media_type='application/zip',
        filename=f"{job.job_name}_model.zip"
    )

@app.delete("/jobs/{job_id}", deprecated=True)
async def cancel_job(job_id: str):
    """取消訓練任務\n可能會導致GPU錯誤，暫停使用"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="任務不存在")
    
    job = training_jobs[job_id]
    
    # 檢查任務狀態
    if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail="任務已結束，無法取消")
    
    # 設置取消標記
    training_processes[job_id] = False
    
    # 立即更新狀態
    training_jobs[job_id].status = TrainingStatus.CANCELLED
    
    return {"message": "任務取消指令已發送，訓練將在下一個檢查點停止"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8881)