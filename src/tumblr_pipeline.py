import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading Tumblr posts...")
posts = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "tumblr_aesthetic_posts.csv"))
print(f"Total posts: {len(posts)}")
print(f"Years: {posts['year'].value_counts().sort_index()}")

print("\nLoading KB...")
kb = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "aesthetics_kb.csv"))
junk = ['__NOTOC__', 'Welcome to', 'The purpose of this page', '#REDIRECT']
kb = kb[~kb.description.apply(lambda x: any(str(x).startswith(j) for j in junk))]
kb = kb.drop_duplicates(subset='aesthetic').reset_index(drop=True)

def clean_kb(text):
    text = re.sub(r'File[A-Za-z0-9_]+', '', text)
    text = re.sub(r'\bfileoriginal\b|\bfilelow\b|\bfilehigh\b|\bnjpg\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

kb['description'] = kb['description'].apply(clean_kb)
print(f"KB aesthetics: {len(kb)}")

# ── clean posts ───────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

posts['clean_text'] = posts['text'].apply(clean_text)
posts = posts[posts['clean_text'].apply(lambda x: len(x.split()) >= 5)]
posts = posts.reset_index(drop=True)
print(f"Posts after cleaning: {len(posts)}")

# ── embed ─────────────────────────────────────────────────────────────────────
print("\nLoading model...")
model = SentenceTransformer('all-mpnet-base-v2')

kb_emb_path = os.path.join(OUT_DIR, "kb_embeddings.npy")
if os.path.exists(kb_emb_path):
    print("Loading cached KB embeddings...")
    kb_embeddings = np.load(kb_emb_path)
else:
    print("Embedding KB...")
    kb_embeddings = model.encode(
        kb['description'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    np.save(kb_emb_path, kb_embeddings)

print("Embedding Tumblr posts...")
post_emb_path = os.path.join(OUT_DIR, "tumblr_embeddings.npy")
if os.path.exists(post_emb_path):
    print("Loading cached post embeddings...")
    post_embeddings = np.load(post_emb_path)
else:
    post_embeddings = model.encode(
        posts['clean_text'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    np.save(post_emb_path, post_embeddings)

print(f"KB embeddings:   {kb_embeddings.shape}")
print(f"Post embeddings: {post_embeddings.shape}")

# ── map each post to closest KB aesthetic ─────────────────────────────────────
print("\nMapping each post to KB aesthetic...")

# batch cosine similarity — all posts vs all KB aesthetics
sims = cosine_similarity(post_embeddings, kb_embeddings)
best_idx = sims.argmax(axis=1)
best_scores = sims.max(axis=1)

posts['kb_match'] = kb.iloc[best_idx]['aesthetic'].values
posts['similarity'] = best_scores
posts['matched'] = best_scores > 0.35

print(f"Posts matched to a KB aesthetic: {posts.matched.sum()}")
print(f"Posts unmatched:                 {(~posts.matched).sum()}")

# ── per aesthetic tag analysis ────────────────────────────────────────────────
print("\n── PER AESTHETIC TAG ANALYSIS ───────────────────────────────────────")
for tag in posts['aesthetic_tag'].unique():
    subset = posts[posts['aesthetic_tag'] == tag]
    top_matches = subset['kb_match'].value_counts().head(3)
    avg_sim = subset['similarity'].mean()
    print(f"\n#{tag} ({len(subset)} posts, avg similarity: {avg_sim:.3f})")
    print(f"  Top KB matches: {', '.join(top_matches.index.tolist())}")

# ── temporal analysis ─────────────────────────────────────────────────────────
print("\n── TEMPORAL ANALYSIS ────────────────────────────────────────────────")
temporal = posts[posts['matched']].groupby(
    ['aesthetic_tag', 'year']
).size().reset_index(name='count')

pivot = temporal.pivot_table(
    index='aesthetic_tag',
    columns='year',
    values='count',
    fill_value=0
)
print(pivot.to_string())

# save everything
posts.to_csv(os.path.join(OUT_DIR, "tumblr_mapped.csv"), index=False)
temporal.to_csv(os.path.join(OUT_DIR, "tumblr_temporal.csv"), index=False)
print(f"\nSaved to {OUT_DIR}")