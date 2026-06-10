import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
import re

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH  = os.path.join(BASE_DIR, "data", "raw", "aesthetics_kb.csv")
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load + clean KB ───────────────────────────────────────────────────────────
print("Loading KB...")
kb = pd.read_csv(KB_PATH)

junk = ['__NOTOC__', 'Welcome to', 'The purpose of this page', '#REDIRECT']
kb = kb[~kb.description.apply(lambda x: any(str(x).startswith(j) for j in junk))]
kb = kb.drop_duplicates(subset='aesthetic').reset_index(drop=True)

def clean_kb_description(text):
    # remove wiki file references
    text = re.sub(r'File[A-Za-z0-9_]+', '', text)
    text = re.sub(r'file[A-Za-z0-9_]+', '', text)
    text = re.sub(r'category[A-Za-z0-9_]+', '', text, flags=re.IGNORECASE)
    # remove specific junk tokens
    junk_tokens = [
        'fileoriginal', 'filelow', 'filehigh', 'filec', 'filedg',
        'filegvc', 'filearabian', 'filegraffiti', 'fileethereal',
        'filescreenshot', 'fileoriginal', 'njpg', 'cjpeg', 'filekissaten',
        'filedorfic', 'filenscg', 'categorycari', 'filediner'
    ]
    for token in junk_tokens:
        text = re.sub(rf'\b{token}\b', '', text, flags=re.IGNORECASE)
    # clean up leftover whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

kb['description'] = kb['description'].apply(clean_kb_description)
print(f"Aesthetics: {len(kb)}")

# ── embed ─────────────────────────────────────────────────────────────────────
print("Loading embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')

emb_path = os.path.join(OUT_DIR, "kb_embeddings.npy")
if os.path.exists(emb_path):
    embeddings = np.load(emb_path)
else:
    embeddings = model.encode(
        kb['description'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )
    np.save(emb_path, embeddings)

# ── cluster ───────────────────────────────────────────────────────────────────
print("Clustering...")
umap_model = UMAP(
    n_neighbors=5,
    n_components=5,
    min_dist=0.0,
    metric='cosine',
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=3,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    nr_topics="auto",
    verbose=False
)

topics, probs = topic_model.fit_transform(
    kb['description'].tolist(),
    embeddings
)

kb['topic'] = topics

# ── 2D umap for visualisation ─────────────────────────────────────────────────
print("Reducing to 2D for visualisation...")
umap_2d = UMAP(
    n_neighbors=5,
    n_components=2,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
coords = umap_2d.fit_transform(embeddings)
kb['x'] = coords[:, 0]
kb['y'] = coords[:, 1]

# ── build family name from topic keywords ─────────────────────────────────────
def get_family_label(topic_id):
    if topic_id == -1:
        return "Outlier"
    words = [w for w, _ in topic_model.get_topic(topic_id)[:3]]
    return ' / '.join(words).title()

kb['family_label'] = kb['topic'].apply(get_family_label)

# ── plotly scatter ────────────────────────────────────────────────────────────
print("Building visualisation...")

# color palette — one per topic
unique_topics = sorted(kb['topic'].unique())
n_topics = len(unique_topics)

# generate distinct colors
import colorsys
colors = {}
for i, t in enumerate(unique_topics):
    if t == -1:
        colors[t] = '#888888'
    else:
        hue = i / max(n_topics - 1, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors[t] = f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'

fig = go.Figure()

for topic_id in unique_topics:
    mask = kb['topic'] == topic_id
    subset = kb[mask]
    label = get_family_label(topic_id)

    fig.add_trace(go.Scatter(
        x=subset['x'],
        y=subset['y'],
        mode='markers',
        name=label,
        text=subset['aesthetic'],
        textposition='top center',
        textfont=dict(size=8),
        marker=dict(
            size=12,
            color=colors[topic_id],
            opacity=0.85,
            line=dict(width=0.5, color='white')
        ),
        hovertemplate=(
            '<b>%{text}</b><br>'
            f'Family: {label}<br>'
            '<extra></extra>'
        )
    ))

for topic_id in unique_topics:
    if topic_id == -1:
        continue
    mask = kb['topic'] == topic_id
    cx = kb[mask]['x'].mean()
    cy = kb[mask]['y'].mean()
    label = get_family_label(topic_id)
    fig.add_annotation(
        x=cx, y=cy,
        text=f"<b>{label}</b>",
        showarrow=False,
        font=dict(size=10, color='white'),
        bgcolor='rgba(0,0,0,0.5)',
        borderpad=3
    )


fig.update_layout(
    title=dict(
        text='Vibe Clustering — Fashion Aesthetic Landscape',
        font=dict(size=20),
        x=0.5
    ),
    width=1400,
    height=900,
    paper_bgcolor='#0f0f0f',
    plot_bgcolor='#0f0f0f',
    font=dict(color='white'),
    legend=dict(
        font=dict(size=9),
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='rgba(255,255,255,0.1)',
        borderwidth=1,
        itemsizing='constant'
    ),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    hoverlabel=dict(bgcolor='#1a1a1a', font_size=12)
)

# save as interactive HTML
out_path = os.path.join(OUT_DIR, "aesthetic_map.html")
fig.write_html(out_path)
print(f"Saved interactive map to {out_path}")

# also save the coordinates
kb[['aesthetic', 'topic', 'family_label', 'x', 'y']].to_csv(
    os.path.join(OUT_DIR, "kb_with_coords.csv"), index=False
)
print("Done — open aesthetic_map.html in your browser")