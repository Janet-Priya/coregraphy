import pandas as pd
import json
import os
from huggingface_hub import hf_hub_download

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading...")
file_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    filename="raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl",
    repo_type="dataset"
)

print(f"File at: {file_path}")
print("Loading 100k reviews...")

data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if len(data) >= 100000:
            break
        try:
            obj = json.loads(line)
            text = obj.get('text', '')
            if text and len(text) > 30:
                data.append({
                    'text': text,
                    'rating': obj.get('rating'),
                    'timestamp': obj.get('timestamp')
                })
        except:
            continue
        if i % 50000 == 0:
            print(f"  scanned {i:,}, kept {len(data):,}")

df = pd.DataFrame(data)
df['year'] = pd.to_datetime(df['timestamp'], unit='ms').dt.year
df = df[df['year'].between(2020, 2023)]
print(f"Reviews 2020-2023: {len(df)}")

df.to_csv(os.path.join(OUT_DIR, "fashion_reviews.csv"), index=False)
print("Saved to data/raw/fashion_reviews.csv")