import pandas as pd
import re
import numpy as np
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH  = os.path.join(BASE_DIR, "data", "raw", "aesthetics_kb.csv")
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── step 1: load fashion dataset ──────────────────────────────────────────────
print("Loading fashion dataset from Hugging Face...")
ds = load_dataset("ashraq/fashion-product-images-small", split="train")
meta = ds.to_pandas()
print(f"Loaded {len(meta)} products")
print(f"Columns: {meta.columns.tolist()}")

# ── step 2: extract and combine text fields ───────────────────────────────────
print("\nExtracting text fields...")

def safe_str(val):
    if isinstance(val, list):
        return ' '.join([str(v) for v in val])
    if val is None:
        return ''
    return str(val)

# figure out which columns have useful text
text_cols = []
for col in meta.columns:
    sample = meta[col].dropna().head(3).tolist()
    if any(isinstance(s, str) and len(s) > 10 for s in sample):
        text_cols.append(col)

print(f"Text columns found: {text_cols}")

# combine all text columns into one
meta['combined_text'] = meta[text_cols].apply(
    lambda row: ' '.join([safe_str(v) for v in row.values]), axis=1
)

# ── step 3: clean text ────────────────────────────────────────────────────────
print("Cleaning text...")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

meta['clean_text'] = meta['combined_text'].apply(clean_text)
meta = meta[meta['clean_text'].apply(lambda x: len(x.split()) >= 5)]
meta = meta.drop_duplicates(subset='clean_text').reset_index(drop=True)
print(f"Products after cleaning: {len(meta)}")

# save cleaned meta for reference
meta[['clean_text']].to_csv(os.path.join(OUT_DIR, "meta_clean.csv"), index=False)

# ── step 4: load KB ───────────────────────────────────────────────────────────
print("\nLoading knowledge base...")
kb = pd.read_csv(KB_PATH)

junk = ['__NOTOC__', 'Welcome to', 'The purpose of this page', '#REDIRECT']
kb = kb[~kb.description.apply(lambda x: any(str(x).startswith(j) for j in junk))]
kb = kb.drop_duplicates(subset='aesthetic').reset_index(drop=True)
print(f"KB aesthetics loaded: {len(kb)}")

# ── step 5: embed both datasets ───────────────────────────────────────────────
print("\nLoading sentence transformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# KB embeddings
kb_emb_path = os.path.join(OUT_DIR, "kb_embeddings.npy")
if os.path.exists(kb_emb_path):
    print("Loading cached KB embeddings...")
    kb_embeddings = np.load(kb_emb_path)
else:
    print("Embedding KB descriptions...")
    kb_embeddings = model.encode(
        kb['description'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    np.save(kb_emb_path, kb_embeddings)

# product embeddings
meta_emb_path = os.path.join(OUT_DIR, "meta_embeddings.npy")
if os.path.exists(meta_emb_path):
    print("Loading cached product embeddings...")
    meta_embeddings = np.load(meta_emb_path)
else:
    print("Embedding product descriptions...")
    meta_embeddings = model.encode(
        meta['clean_text'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    np.save(meta_emb_path, meta_embeddings)

print(f"KB embeddings shape:   {kb_embeddings.shape}")
print(f"Meta embeddings shape: {meta_embeddings.shape}")

# ── step 6: run bertopic ──────────────────────────────────────────────────────
print("\nRunning BERTopic...")

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=30,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    nr_topics="auto",
    verbose=True
)

topics, probs = topic_model.fit_transform(
    meta['clean_text'].tolist(),
    meta_embeddings
)

meta['topic'] = topics
n_topics = len(set(topics)) - 1
print(f"\nTopics discovered: {n_topics}")
print(topic_model.get_topic_info().head(20).to_string())

# save topic info
topic_info = topic_model.get_topic_info()
topic_info.to_csv(os.path.join(OUT_DIR, "topic_info.csv"), index=False)
meta[['clean_text', 'topic']].to_csv(
    os.path.join(OUT_DIR, "meta_with_topics.csv"), index=False
)

# ── step 7: map clusters to KB aesthetics ─────────────────────────────────────
print("\nMapping clusters to KB aesthetics...")

results = []
for topic_id in sorted(set(topics)):
    if topic_id == -1:
        continue

    mask = meta['topic'] == topic_id
    centroid = meta_embeddings[mask].mean(axis=0)

    sims = cosine_similarity([centroid], kb_embeddings)[0]
    best_idx = sims.argmax()
    best_score = float(sims[best_idx])

    top_words = [w for w, _ in topic_model.get_topic(topic_id)[:5]]

    results.append({
        'topic_id':        topic_id,
        'size':            int(mask.sum()),
        'top_words':       ', '.join(top_words),
        'best_kb_match':   kb.iloc[best_idx]['aesthetic'],
        'similarity_score': round(best_score, 3),
        'matched':         best_score > 0.35
    })

mapping_df = pd.DataFrame(results).sort_values('similarity_score', ascending=False)
mapping_df.to_csv(os.path.join(OUT_DIR, "cluster_to_aesthetic_mapping.csv"), index=False)

# ── step 8: print results ─────────────────────────────────────────────────────
print("\n── RESULTS ──────────────────────────────────────────────────────────")
print(f"Total clusters discovered:        {len(mapping_df)}")
print(f"Matched to a KB aesthetic:        {mapping_df.matched.sum()}")
print(f"Unmatched (emerging unnamed vibes): {(~mapping_df.matched).sum()}")

print("\nTop matched aesthetics:")
matched = mapping_df[mapping_df.matched].head(10)
print(matched[['topic_id','size','top_words','best_kb_match','similarity_score']].to_string(index=False))

print("\nEmerging unnamed clusters:")
unmatched = mapping_df[~mapping_df.matched].head(10)
print(unmatched[['topic_id','size','top_words','best_kb_match','similarity_score']].to_string(index=False))

print(f"\nOutputs saved to {OUT_DIR}")