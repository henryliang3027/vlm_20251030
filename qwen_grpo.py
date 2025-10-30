from huggingface_hub import login
from datasets import load_dataset


login()

dataset_id = 'lmms-lab/multimodal-open-r1-8k-verified'
dataset = load_dataset(dataset_id, split='train[:5%]')

split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

print(train_dataset)
print(test_dataset)

# show one of the samples image and text

import matplotlib.pyplot as plt

sample = train_dataset[0]
image = sample['image']
text = sample['problem']

plt.figure(figsize=(10, 8))
plt.imshow(image)
plt.axis('off')
plt.title(text)
plt.savefig('output.png', bbox_inches='tight', dpi=150)
plt.close()

print("圖片已儲存為 output.png")