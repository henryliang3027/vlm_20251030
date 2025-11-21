from datasets import load_dataset, load_from_disk, Dataset

# Load custom dataset from CSV
csv_path = './training_data/main.csv'
print(f"Loading data from {csv_path}...")
ds = load_dataset("csv", data_files=csv_path, split="train")
print(f"Loaded {len(ds)} samples from CSV file.")
print(f"Dataset columns: {ds.column_names}")
for i, sample in enumerate(ds):
    if i < 42:  # Print first 3 samples for verification
        print(f"Sample {i} image: {sample['image']}")
        print(f"Sample {i} question: {sample['question']}")
        print(f"Sample {i} answer: {sample['answer']}")
        