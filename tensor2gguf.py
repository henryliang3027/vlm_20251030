from unsloth import FastVisionModel  # FastLanguageModel for LLMs


model, tokenizer = FastVisionModel.from_pretrained(
    model_name="finetune_model_merged",
    load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
)

model.save_pretrained_gguf(
    "finetune_model_gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
