# convert json to csv
import json
import csv
import os
import pandas as pd

# Paths
json_path = './training_data/main.json'
csv_path = './training_data/main.csv'
target_csv = './training_data/main_updated.csv'

# Load JSON data
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Loaded {len(data)} samples from JSON file.")
# Write to CSV
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image', 'question', 'answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    for item in data:
        writer.writerow({
            'image': item['image'],
            'question': item['question'],
            'answer': item['answer']  # 只为answer添加引号
        })
print(f"Data successfully written to {csv_path}.")



df = pd.read_csv(csv_path, quotechar='"', quoting=csv.QUOTE_MINIMAL,skipinitialspace=True)
required_cols = ['image', 'question', 'answer']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print({"success": False, "error": f"CSV缺少列: {missing_cols}"})


df.to_csv(target_csv, index=False, quoting=csv.QUOTE_MINIMAL, quotechar='"')
