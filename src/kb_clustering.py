import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH  = os.path.join(BASE_DIR, "data", "raw", "aesthetics_kb.csv")
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)


# ── step 1: load KB ───────────────────────────────────────────────────────────
print("Loading knowledge base...")
kb = pd.read_csv(KB_PATH)

junk = ['__NOTOC__', 'Welcome to', 'The purpose of this page', '#REDIRECT']
kb = kb[~kb.description.apply(lambda x: any(str(x).startswith(j) for j in junk))]
kb = kb.drop_duplicates(subset='aesthetic').reset_index(drop=True)

def clean_kb_description(text):
    # remove wiki file references
    text = re.sub(r'File[A-Za-z0-9_]+', '', text)
    text = re.sub(r'file[A-Za-z0-9_]+', '', text)
    text = re.sub(r'category[A-Za-z0-9_]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bnjpg\b|\bfilec\b|\bfileoriginal\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

kb['description'] = kb['description'].apply(clean_kb_description)
print(f"Aesthetics loaded: {len(kb)}")

# ── step 2: embed ─────────────────────────────────────────────────────────────
print("\nEmbedding KB descriptions...")
model = SentenceTransformer('all-MiniLM-L6-v2')

emb_path = os.path.join(OUT_DIR, "kb_embeddings.npy")
if os.path.exists(emb_path):
    print("Loading cached embeddings...")
    embeddings = np.load(emb_path)
else:
    embeddings = model.encode(
        kb['description'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    np.save(emb_path, embeddings)

print(f"Embeddings shape: {embeddings.shape}")

# ── step 3: run BERTopic ──────────────────────────────────────────────────────
print("\nRunning BERTopic on KB...")

umap_model = UMAP(
    n_neighbors=5,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=3,   # small since we only have 654 docs
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)


vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2))
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    nr_topics="auto",
    verbose=True
)

topics, probs = topic_model.fit_transform(
    kb['description'].tolist(),
    embeddings
)

kb['topic'] = topics
n_topics = len(set(topics)) - 1
print(f"\nAesthetic families discovered: {n_topics}")

# ── step 4: show results ──────────────────────────────────────────────────────
print("\nTopic info:")
print(topic_model.get_topic_info().to_string())

# show which aesthetics landed in each cluster
print("\n── AESTHETIC FAMILIES ───────────────────────────────────────────────")
for topic_id in sorted(set(topics)):
    if topic_id == -1:
        label = "NOISE (outlier aesthetics)"
    else:
        keywords = [w for w, _ in topic_model.get_topic(topic_id)[:5]]
        label = f"Family {topic_id}: {', '.join(keywords)}"
    
    members = kb[kb['topic'] == topic_id]['aesthetic'].tolist()
    print(f"\n{label}")
    print(f"  Members ({len(members)}): {', '.join(members[:15])}")
    if len(members) > 15:
        print(f"  ... and {len(members) - 15} more")

# ── step 5: save ──────────────────────────────────────────────────────────────
kb[['aesthetic', 'topic', 'description', 'url']].to_csv(
    os.path.join(OUT_DIR, "kb_clusters.csv"), index=False
)
topic_model.get_topic_info().to_csv(
    os.path.join(OUT_DIR, "kb_topic_info.csv"), index=False
)
print(f"\nSaved to {OUT_DIR}")