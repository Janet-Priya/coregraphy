import pandas as pd
import numpy as np
import os
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")

# ── load data ─────────────────────────────────────────────────────────────────
posts = pd.read_csv(os.path.join(OUT_DIR, "tumblr_mapped.csv"))
kb    = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "aesthetics_kb.csv"))
kb_coords = pd.read_csv(os.path.join(OUT_DIR, "kb_with_coords.csv"))

junk = ['__NOTOC__', 'Welcome to', 'The purpose of this page', '#REDIRECT']
kb = kb[~kb.description.apply(lambda x: any(str(x).startswith(j) for j in junk))]
kb = kb.drop_duplicates(subset='aesthetic').reset_index(drop=True)

# ── chart 1: tag → KB match heatmap ──────────────────────────────────────────
print("Building heatmap...")

# for each tag get top KB matches and their avg similarity
heatmap_data = []
for tag in posts['aesthetic_tag'].unique():
    subset = posts[posts['aesthetic_tag'] == tag]
    top = subset.groupby('kb_match')['similarity'].mean().nlargest(5)
    for aesthetic, score in top.items():
        heatmap_data.append({
            'tag': f"#{tag}",
            'kb_aesthetic': aesthetic,
            'similarity': round(score, 3)
        })

heatmap_df = pd.DataFrame(heatmap_data)
pivot = heatmap_df.pivot_table(
    index='tag',
    columns='kb_aesthetic',
    values='similarity',
    fill_value=0
)

fig1 = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=pivot.columns.tolist(),
    y=pivot.index.tolist(),
    colorscale='Viridis',
    text=np.round(pivot.values, 2),
    texttemplate="%{text}",
    textfont=dict(size=9),
    hoverongaps=False,
    hovertemplate="Tag: %{y}<br>KB Aesthetic: %{x}<br>Similarity: %{z:.3f}<extra></extra>"
))

fig1.update_layout(
    title=dict(text="Tumblr Aesthetic Tags → KB Aesthetic Matches", x=0.5, font=dict(size=18)),
    paper_bgcolor='#0f0f0f',
    plot_bgcolor='#0f0f0f',
    font=dict(color='white'),
    width=1400,
    height=700,
    xaxis=dict(tickangle=45, tickfont=dict(size=9)),
    yaxis=dict(tickfont=dict(size=11))
)

fig1.write_html(os.path.join(OUT_DIR, "heatmap_tag_to_kb.html"))
print("Saved heatmap")

# ── chart 2: similarity score bar chart per tag ───────────────────────────────
print("Building similarity bar chart...")

tag_sim = posts.groupby('aesthetic_tag')['similarity'].mean().sort_values(ascending=True)

fig2 = go.Figure(go.Bar(
    x=tag_sim.values,
    y=[f"#{t}" for t in tag_sim.index],
    orientation='h',
    marker=dict(
        color=tag_sim.values,
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(title="Avg Similarity")
    ),
    hovertemplate="%{y}<br>Avg similarity: %{x:.3f}<extra></extra>"
))

fig2.update_layout(
    title=dict(text="How Well Each Aesthetic Tag Matches the KB", x=0.5, font=dict(size=18)),
    paper_bgcolor='#0f0f0f',
    plot_bgcolor='#0f0f0f',
    font=dict(color='white'),
    width=900,
    height=700,
    xaxis=dict(title="Average Cosine Similarity", range=[0, 0.7]),
    yaxis=dict(title="")
)

fig2.write_html(os.path.join(OUT_DIR, "bar_tag_similarity.html"))
print("Saved bar chart")

# ── chart 3: aesthetic map with tumblr tags overlaid ─────────────────────────
print("Building aesthetic map with Tumblr overlay...")

# load KB embeddings and compute 2D coords
kb_emb_path = os.path.join(OUT_DIR, "kb_embeddings.npy")
post_emb_path = os.path.join(OUT_DIR, "tumblr_embeddings.npy")

kb_embeddings   = np.load(kb_emb_path)
post_embeddings = np.load(post_emb_path)

from umap import UMAP
print("Fitting UMAP on KB...")
umap_2d = UMAP(n_neighbors=5, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
kb_2d = umap_2d.fit_transform(kb_embeddings)

print("Projecting Tumblr posts into same space...")
post_2d = umap_2d.transform(post_embeddings)

kb['x'] = kb_2d[:, 0]
kb['y'] = kb_2d[:, 1]
posts['x'] = post_2d[:, 0]
posts['y'] = post_2d[:, 1]

fig3 = go.Figure()

# KB dots — grey background
fig3.add_trace(go.Scatter(
    x=kb['x'], y=kb['y'],
    mode='markers',
    name='KB Aesthetics',
    text=kb['aesthetic'],
    marker=dict(size=6, color='rgba(150,150,150,0.4)'),
    hovertemplate="<b>%{text}</b><extra></extra>"
))

# Tumblr posts coloured by tag
colors = px.colors.qualitative.Plotly
tags = posts['aesthetic_tag'].unique()

for i, tag in enumerate(tags):
    subset = posts[posts['aesthetic_tag'] == tag]
    fig3.add_trace(go.Scatter(
        x=subset['x'], y=subset['y'],
        mode='markers',
        name=f"#{tag}",
        text=subset['kb_match'],
        marker=dict(
            size=8,
            color=colors[i % len(colors)],
            opacity=0.8,
            symbol='diamond'
        ),
        hovertemplate=f"<b>#{tag}</b><br>KB match: %{{text}}<extra></extra>"
    ))

fig3.update_layout(
    title=dict(text="Aesthetic Landscape — KB + Tumblr Posts Overlaid", x=0.5, font=dict(size=18)),
    paper_bgcolor='#0f0f0f',
    plot_bgcolor='#0f0f0f',
    font=dict(color='white'),
    width=1400,
    height=900,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    legend=dict(font=dict(size=9), bgcolor='rgba(0,0,0,0.5)')
)

fig3.write_html(os.path.join(OUT_DIR, "map_kb_tumblr_overlay.html"))
print("Saved overlay map")

print(f"\nAll 3 visualisations saved to {OUT_DIR}")
print("Open these in your browser:")
print("  1. heatmap_tag_to_kb.html    — tag to KB match heatmap")
print("  2. bar_tag_similarity.html   — similarity scores per tag")
print("  3. map_kb_tumblr_overlay.html — full aesthetic map with Tumblr overlay")