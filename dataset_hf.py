from datasets import load_dataset
dataset = load_dataset("unsloth/LaTeX_OCR", split="train[:1%]")


print(type(dataset[2]["image"]))
