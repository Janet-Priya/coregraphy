import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading KB and embeddings...")
kb = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "aesthetics_kb.csv"))
kb_embeddings = np.load(os.path.join(OUT_DIR, "kb_embeddings.npy"))

junk = ['__NOTOC__', 'Welcome to', 'The purpose of this page', '#REDIRECT']
kb = kb[~kb.description.apply(lambda x: any(str(x).startswith(j) for j in junk))]
kb = kb.drop_duplicates(subset='aesthetic').reset_index(drop=True)
print(f"KB aesthetics: {len(kb)}")

# ── focus aesthetics — your 20 tumblr tags + key neighbours ──────────────────
focus = [
    "Dark Academia", "Light Academia", "Cottagecore", "Goblincore",
    "Fairycore", "Coquette", "Baddie", "Coastal Grandmother",
    "Gorpcore", "Indie Sleaze", "Soft Girl", "Grunge",
    "Kawaii", "Hypebeast", "Craftcore", "Cyberpunk",
    "Balletcore", "Streetwear", "Y2K", "Quiet Luxury",
    "Cottagecore", "Bloomcore", "Gardencore", "Cozycore",
    "Cleancore", "Dreamcore", "Cybergoth", "Cyberprep",
    "Health Goth", "Corporate Goth", "Goth", "Witchcore",
    "Grandmacore", "Maximalism", "Lifestyle Minimalism"
]

# match to actual KB names (case insensitive)
kb_lower = kb['aesthetic'].str.lower()
matched = []
for name in focus:
    match = kb[kb_lower == name.lower()]
    if len(match) > 0:
        matched.append(match.iloc[0])

focus_kb = pd.DataFrame(matched).drop_duplicates(subset='aesthetic').reset_index(drop=True)
print(f"Focus aesthetics matched: {len(focus_kb)}")

# get embeddings for focus aesthetics
focus_indices = [kb[kb['aesthetic'] == a].index[0] for a in focus_kb['aesthetic']]
focus_embeddings = kb_embeddings[focus_indices]

# ── compute pairwise cosine similarity ────────────────────────────────────────
print("Computing pairwise similarities...")
sim_matrix = cosine_similarity(focus_embeddings)
sim_df = pd.DataFrame(
    sim_matrix,
    index=focus_kb['aesthetic'].tolist(),
    columns=focus_kb['aesthetic'].tolist()
)

sim_df.to_csv(os.path.join(OUT_DIR, "aesthetic_distance_matrix.csv"))
print("Saved distance matrix")

# ── chart 1: heatmap of pairwise similarities ─────────────────────────────────
print("Building similarity heatmap...")

labels = focus_kb['aesthetic'].tolist()

fig1 = go.Figure(data=go.Heatmap(
    z=sim_matrix,
    x=labels,
    y=labels,
    colorscale='RdYlGn',
    zmin=0.2,
    zmax=1.0,
    hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>"
))

fig1.update_layout(
    title=dict(text="Aesthetic Distance Matrix — How Similar Are Aesthetics to Each Other?", x=0.5, font=dict(size=16)),
    paper_bgcolor='#0f0f0f',
    plot_bgcolor='#0f0f0f',
    font=dict(color='white'),
    width=1000,
    height=800,
    xaxis=dict(tickangle=45, tickfont=dict(size=9)),
    yaxis=dict(tickfont=dict(size=9))
)

fig1.write_html(os.path.join(OUT_DIR, "aesthetic_distance_heatmap.html"))
print("Saved heatmap")

# ── chart 2: top 5 nearest neighbours for each focus aesthetic ────────────────
print("Finding nearest neighbours...")

neighbours = []
for aesthetic in labels:
    row = sim_df[aesthetic].drop(aesthetic).nlargest(5)
    for neighbour, score in row.items():
        neighbours.append({
            'aesthetic': aesthetic,
            'neighbour': neighbour,
            'similarity': round(score, 3)
        })

neighbours_df = pd.DataFrame(neighbours)
neighbours_df.to_csv(os.path.join(OUT_DIR, "aesthetic_neighbours.csv"), index=False)

print("\n── NEAREST NEIGHBOURS ───────────────────────────────────────────────")
for aesthetic in labels:
    top = neighbours_df[neighbours_df['aesthetic'] == aesthetic].head(3)
    neighbours_list = ', '.join([f"{r['neighbour']} ({r['similarity']})" for _, r in top.iterrows()])
    print(f"{aesthetic:25} → {neighbours_list}")

# ── chart 3: network graph of aesthetic relationships ────────────────────────
print("\nBuilding network graph...")

# only show edges with similarity > 0.45 to keep it clean
threshold = 0.45
edge_x, edge_y = [], []
node_x, node_y = [], []
node_text = []

# use UMAP coords if available, else compute
coords_path = os.path.join(OUT_DIR, "kb_with_coords.csv")
if os.path.exists(coords_path):
    coords = pd.read_csv(coords_path)
    pos = {}
    for aesthetic in labels:
        row = coords[coords['aesthetic'] == aesthetic]
        if len(row) > 0:
            pos[aesthetic] = (row.iloc[0]['x'], row.iloc[0]['y'])
else:
    from umap import UMAP
    umap_2d = UMAP(n_neighbors=8, n_components=2, min_dist=0.05, metric='cosine', random_state=42)
    coords_2d = umap_2d.fit_transform(focus_embeddings)
    pos = {labels[i]: (coords_2d[i, 0], coords_2d[i, 1]) for i in range(len(labels))}

# build edges
for i, a1 in enumerate(labels):
    for j, a2 in enumerate(labels):
        if i >= j:
            continue
        if sim_matrix[i][j] > threshold and a1 in pos and a2 in pos:
            edge_x += [pos[a1][0], pos[a2][0], None]
            edge_y += [pos[a1][1], pos[a2][1], None]

# build nodes
for aesthetic in labels:
    if aesthetic in pos:
        node_x.append(pos[aesthetic][0])
        node_y.append(pos[aesthetic][1])
        node_text.append(aesthetic)

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    line=dict(width=0.8, color='rgba(100,200,150,0.4)'),
    hoverinfo='none',
    showlegend=False
))

# colour nodes by family
family_colors = {
    'Dark Academia': '#8B4513', 'Light Academia': '#D2691E',
    'Cottagecore': '#228B22', 'Goblincore': '#556B2F',
    'Fairycore': '#FF69B4', 'Bloomcore': '#FF1493',
    'Gardencore': '#32CD32', 'Cozycore': '#DAA520',
    'Craftcore': '#CD853F', 'Grandmacore': '#DEB887',
    'Cyberpunk': '#00FFFF', 'Cybergoth': '#7B68EE',
    'Cyberprep': '#4169E1', 'Dreamcore': '#9370DB',
    'Cleancore': '#87CEEB', 'Goth': '#800080',
    'Corporate Goth': '#4B0082', 'Health Goth': '#483D8B',
    'Grunge': '#808000', 'Indie Sleaze': '#B8860B',
    'Hypebeast': '#FF4500', 'Streetwear': '#FF6347',
    'Baddie': '#FF1493', 'Kawaii': '#FFB6C1',
    'Coquette': '#FFC0CB', 'Balletcore': '#FFE4E1',
    'Gorpcore': '#6B8E23', 'Coastal Grandmother': '#87CEEB',
    'Maximalism': '#FFD700', 'Lifestyle Minimalism': '#F5F5F5',
}

node_colors = [family_colors.get(n, '#00d4aa') for n in node_text]

fig3.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition='top center',
    textfont=dict(size=9, color='white'),
    marker=dict(
        size=14,
        color=node_colors,
        opacity=0.95,
        line=dict(width=1.5, color='white')
    ),
    hovertemplate="<b>%{text}</b><extra></extra>"
))

fig3.update_layout(
    title=dict(text="Aesthetic Relationship Network — Connected = Semantically Similar", x=0.5, font=dict(size=16)),
    paper_bgcolor='#0f0f0f',
    plot_bgcolor='#0f0f0f',
    font=dict(color='white'),
    width=1200,
    height=900,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    showlegend=False
)

fig3.write_html(os.path.join(OUT_DIR, "aesthetic_network.html"))
print("Saved network graph")

print(f"\nAll outputs saved to {OUT_DIR}")
print("Files:")
print("  aesthetic_distance_heatmap.html — pairwise similarity matrix")
print("  aesthetic_network.html          — relationship network graph")
print("  aesthetic_neighbours.csv        — top 5 neighbours per aesthetic")