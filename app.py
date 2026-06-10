import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(sys_path, "src"))
import pixel_art as pa

# ── config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Aesthetic Atlas",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")

# ── palette (from build_dashboard.py) ────────────────────────────────────────
BG     = "#161310"
CARD   = "#201d17"
INK    = "#ece4d2"
MUTE   = "#9a917b"
LINE   = "#3b3a2e"
GREEN  = "#83a85f"
ROSE   = "#d98b78"
GOLD   = "#d8b25e"
BLUE   = "#7c9fb0"
PURPLE = "#9a86b8"
PINK   = "#e0a6a0"

# Family classification + colours live in build_dashboard.py so the app and the
# static dashboard always agree. This used to be a hand-synced copy and drifted
# out of date — now there is a single source of truth.
from build_dashboard import FAMILY_COLORS, super_family

# ── pixel art svg helpers ─────────────────────────────────────────────────────
monitor_svg  = pa.to_svg(pa.MONITOR, 5, with_screen=True)
plant_svg    = pa.to_svg(pa.PLANT, 4)
mug_svg      = pa.to_svg(pa.MUG, 4)
flower_svg   = pa.to_svg(pa.FLOWER, 3)
leaf_svg     = pa.to_svg(pa.LEAF, 3)
star_svg     = pa.to_svg(pa.STAR, 3)
mushroom_svg = pa.to_svg(pa.MUSHROOM, 3)

# ── css ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

.stApp {{ background-color: {BG}; }}
section[data-testid="stSidebar"] {{ background-color: {CARD}; }}
.stApp, .stMarkdown, p, li, label {{ 
    font-family: 'VT323', monospace !important; 
    color: {INK}; 
}}

.pixel-hero {{
    text-align: center;
    padding: 20px 0 30px;
    border-bottom: 2px solid {LINE};
    margin-bottom: 24px;
}}
.pixel-hero h1 {{
    font-family: 'Press Start 2P', monospace;
    font-size: 28px;
    color: {INK};
    text-shadow: 3px 3px 0 #0c0a08;
    margin: 12px 0 8px;
}}
.pixel-hero .sub {{
    font-family: 'VT323', monospace;
    font-size: 22px;
    color: {GOLD};
    letter-spacing: 2px;
}}

.pix-card {{
    background: {CARD};
    border: 2px solid {LINE};
    padding: 16px 18px;
    margin-bottom: 16px;
    box-shadow: 4px 4px 0 #0c0a08;
    position: relative;
}}
.pix-card::before {{
    content: '';
    position: absolute;
    top: -2px; left: -2px;
    width: 6px; height: 6px;
    background: {GOLD};
}}
.pix-card::after {{
    content: '';
    position: absolute;
    bottom: -2px; right: -2px;
    width: 6px; height: 6px;
    background: {GOLD};
}}
.pix-card h3 {{
    font-family: 'Press Start 2P', monospace;
    font-size: 11px;
    color: {GREEN};
    margin: 0 0 12px;
}}

.metric-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}}
.metric-box {{
    background: {CARD};
    border: 2px solid {LINE};
    padding: 14px;
    text-align: center;
    box-shadow: 3px 3px 0 #0c0a08;
}}
.metric-num {{
    font-family: 'Press Start 2P', monospace;
    font-size: 20px;
    color: {GOLD};
}}
.metric-lab {{
    font-family: 'VT323', monospace;
    font-size: 18px;
    color: {MUTE};
    margin-top: 6px;
}}

.finding {{
    padding: 10px 0;
    border-bottom: 1px dashed {LINE};
}}
.finding:last-child {{ border: none; }}
.f-title {{
    font-family: 'Press Start 2P', monospace;
    font-size: 10px;
    color: {INK};
}}
.f-body {{
    font-family: 'VT323', monospace;
    font-size: 19px;
    color: {MUTE};
    margin-top: 5px;
    line-height: 1.3;
}}

.match-card {{
    background: {CARD};
    border: 2px solid {GREEN};
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 3px 3px 0 #0c0a08;
}}
.match-name {{
    font-family: 'Press Start 2P', monospace;
    font-size: 12px;
    color: {GREEN};
}}
.match-score {{
    font-family: 'VT323', monospace;
    font-size: 20px;
    color: {GOLD};
}}
.match-desc {{
    font-family: 'VT323', monospace;
    font-size: 18px;
    color: {MUTE};
    margin-top: 6px;
    line-height: 1.3;
}}

.chip {{
    display: inline-block;
    padding: 4px 10px;
    margin: 3px;
    font-family: 'VT323', monospace;
    font-size: 18px;
    border: 1px solid {LINE};
}}
.chip-a {{ background: {ROSE}; color: #0c0a08; }}
.chip-r {{ background: {CARD}; color: {INK}; }}

.cursor {{
    display: inline-block;
    width: 10px; height: 18px;
    background: {GREEN};
    margin-left: 3px;
    animation: blink 1s steps(1) infinite;
    vertical-align: -3px;
}}
@keyframes blink {{ 50% {{ opacity: 0; }} }}

/* ── de-Streamlit: hide the default top chrome ──────────────────── */
header[data-testid="stHeader"] {{ display: none; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stStatusWidget"] {{ display: none; }}
#MainMenu {{ visibility: hidden; }}
footer {{ display: none; }}

/* tighten the big default top/bottom whitespace now the header is gone */
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 2.5rem;
    max-width: 1150px;
}}

/* force remaining default widgets onto the warm palette + pixel font */
.stTextArea textarea, .stTextInput input {{
    background-color: {CARD} !important;
    color: {INK} !important;
    border: 1px solid {LINE} !important;
    font-family: 'VT323', monospace !important;
    font-size: 18px !important;
}}
.stButton button {{
    font-family: 'VT323', monospace !important;
    font-size: 19px !important;
    border: 1px solid {LINE} !important;
}}
</style>
""", unsafe_allow_html=True)

# ── load data ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

@st.cache_data
def load_kb():
    kb = pd.read_csv(os.path.join(RAW_DIR, "aesthetics_kb.csv"))
    junk = ['__NOTOC__', 'Welcome to', 'The purpose of this page', '#REDIRECT']
    kb = kb[~kb.description.apply(lambda x: any(str(x).startswith(j) for j in junk))]
    kb = kb.drop_duplicates(subset='aesthetic').reset_index(drop=True)
    def clean(text):
        text = re.sub(r'File[A-Za-z0-9_]+', '', text)
        text = re.sub(r'\bfileoriginal\b|\bfilelow\b|\bnjpg\b', '', text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', text).strip()
    kb['description'] = kb['description'].apply(clean)
    return kb

@st.cache_data
def load_embeddings():
    path = os.path.join(OUT_DIR, "kb_embeddings.npy")
    if os.path.exists(path):
        return np.load(path)
    else:
        st.info("generating embeddings for the first time — this takes a few minutes...")
        kb_data = load_kb()
        embs = load_model().encode(
            kb_data['description'].tolist(),
            show_progress_bar=False,
            batch_size=32
        )
        np.save(path, embs)
        return embs

@st.cache_data
def load_coords():
    coords = pd.read_csv(os.path.join(OUT_DIR, "kb_with_coords.csv"))
    coords['super_family'] = coords['family_label'].fillna('Unclustered').apply(super_family)
    return coords

@st.cache_data
def load_trends():
    path = os.path.join(OUT_DIR, "google_trends.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None

@st.cache_data
def load_distance():
    path = os.path.join(OUT_DIR, "aesthetic_distance_matrix.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_neighbours():
    path = os.path.join(OUT_DIR, "aesthetic_neighbours.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

model         = load_model()
kb            = load_kb()
kb_embeddings = load_embeddings()
coords        = load_coords()
trends_df     = load_trends()
dist_df       = load_distance()
neighbours_df = load_neighbours()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 20px">
        {monitor_svg}
        <div style="font-family:'Press Start 2P',monospace;font-size:11px;
                    color:{GREEN};margin-top:10px">AESTHETIC<br>ATLAS</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("navigate", [
        "🏠 home",
        "🔍 find my aesthetic",
        "🗺️ landscape map",
        "📈 trend analysis",
        "🔗 aesthetic distances",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <div style="margin-top:30px;padding-top:16px;border-top:1px dashed {LINE}">
        <div style="display:flex;gap:8px;justify-content:center">
            {plant_svg}{mug_svg}
        </div>
        <div style="font-family:'VT323',monospace;font-size:16px;
                    color:{MUTE};text-align:center;margin-top:8px">
            657 aesthetics mapped<br>2018 → 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 home":
    st.markdown(f"""
    <div class="pixel-hero">
        <div>{monitor_svg}</div>
        <h1>THE AESTHETIC ATLAS</h1>
        <div class="sub">mapping fashion vibes · 2018–2026<span class="cursor"></span></div>
    </div>
    """, unsafe_allow_html=True)

    # metrics
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-box">
            <div class="metric-num">657</div>
            <div class="metric-lab">aesthetics mapped</div>
        </div>
        <div class="metric-box">
            <div class="metric-num">51</div>
            <div class="metric-lab">vibe families</div>
        </div>
        <div class="metric-box">
            <div class="metric-num">8yr</div>
            <div class="metric-lab">trend history</div>
        </div>
        <div class="metric-box">
            <div class="metric-num">53k</div>
            <div class="metric-lab">reviews analysed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="pix-card">
            <h3>the findings</h3>
            <div class="finding">
                <div class="f-title">covid drove escapism</div>
                <div class="f-body">cottagecore & dark academia rose from near-zero in 2020,
                peaked 2021 — straight off the lockdown curve.</div>
            </div>
            <div class="finding">
                <div class="f-title">aesthetics self-organise</div>
                <div class="f-body">the model grouped cozycore · bloomcore · gardencore · craftcore
                with no labels. same for goth and cyber families.</div>
            </div>
            <div class="finding">
                <div class="f-title">the post-2022 turn</div>
                <div class="f-body">interest moved from nature & academia toward confidence &
                nostalgia — coquette, y2k, gorpcore.</div>
            </div>
            <div class="finding">
                <div class="f-title">streetwear is evergreen</div>
                <div class="f-body">the only tag holding high interest across all 8 years
                while everything else rises and falls.</div>
            </div>
            <div class="finding">
                <div class="f-title">identity ≠ basket</div>
                <div class="f-body">amazon reviews talk fit, size, comfort — not aesthetics.
                how we dress as identity barely touches how we shop.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="pix-card">
            <h3>two vocabularies</h3>
            <div style="font-family:'VT323',monospace;font-size:18px;
                        color:{GOLD};margin-bottom:8px">how we IDENTIFY</div>
            <div>
                {''.join(f'<span class="chip chip-a">{w}</span>'
                for w in ["cottagecore","coquette","y2k","gorpcore",
                          "balletcore","dark academia","fairycore"])}
            </div>
            <div style="font-family:'VT323',monospace;font-size:18px;
                        color:{GOLD};margin:14px 0 8px">how we SHOP</div>
            <div>
                {''.join(f'<span class="chip chip-r">{w}</span>'
                for w in ["fit","size","comfortable","quality",
                          "wear","color","leather","nice"])}
            </div>
            <div style="font-family:'VT323',monospace;font-size:18px;
                        color:{MUTE};margin-top:16px;line-height:1.4">
                aesthetic language lives on tumblr and pinterest.<br>
                the checkout is purely functional.
            </div>
        </div>
        <div class="pix-card" style="margin-top:0">
            <h3>decorations</h3>
            <div style="display:flex;gap:12px;align-items:flex-end;justify-content:center;padding:8px 0">
                {flower_svg}{plant_svg}{mushroom_svg}{leaf_svg}{star_svg}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FIND MY AESTHETIC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 find my aesthetic":
    st.markdown(f"""
    <div class="pixel-hero">
        <h1>FIND YOUR VIBE</h1>
        <div class="sub">describe your style → nearest aesthetic<span class="cursor"></span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pix-card">
        <h3>how it works</h3>
        <div style="font-family:'VT323',monospace;font-size:19px;color:{MUTE}">
            type how you dress in plain english. the model converts your description
            into a semantic embedding and finds the closest aesthetics from 657 documented ones.
            no keywords needed — just describe the vibe.
        </div>
    </div>
    """, unsafe_allow_html=True)

    user_input = st.text_area(
        "describe your style:",
        placeholder="e.g. I love dark colours, vintage books, oversized coats, old libraries, tweed, candlelight and muted tones...",
        height=100,
        label_visibility="visible"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        top_n = st.slider("matches to show", 3, 10, 5)
    with col2:
        search_btn = st.button("▶ find my aesthetic", type="primary", use_container_width=True)

    if search_btn and user_input.strip():
        with st.spinner("embedding your vibe..."):
            user_emb = model.encode([user_input])
            sims = cosine_similarity(user_emb, kb_embeddings)[0]
            # Mask out non-aesthetic wiki pages (guides, glossaries, disambiguation,
            # category/list pages) so they never surface as matches. Done here at
            # ranking time — not in load_kb() — to keep kb aligned with the
            # cached embeddings (.npy) row-for-row.
            junk_pat = re.compile(
                r'\(disambiguation\)|how to|differences between|list of|glossary|'
                r'sandbox|\bwiki\b|categor|template|^aesthetics$', re.I)
            is_junk = kb['aesthetic'].astype(str).str.contains(junk_pat).to_numpy()
            sims = np.where(is_junk, -1.0, sims)
            top_indices = sims.argsort()[::-1][:top_n]

        st.markdown("---")

        # best match hero
        best_idx = top_indices[0]
        best_aesthetic = kb.iloc[best_idx]['aesthetic']
        best_score = sims[best_idx]
        best_desc = kb.iloc[best_idx]['description'][:400]
        best_url = kb.iloc[best_idx]['url']

        st.markdown(f"""
        <div class="match-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div class="match-name">🎯 {best_aesthetic}</div>
                <div class="match-score">similarity: {best_score:.3f}</div>
            </div>
            <div class="match-desc">{best_desc}...</div>
            <div style="margin-top:10px">
                <a href="{best_url}" target="_blank"
                   style="font-family:'VT323',monospace;font-size:18px;color:{GOLD}">
                   → read more on aesthetics wiki
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # neighbours from distance matrix
        if neighbours_df is not None:
            nb = neighbours_df[neighbours_df['aesthetic'] == best_aesthetic].head(3)
            if len(nb) > 0:
                st.markdown(f"""
                <div style="font-family:'Press Start 2P',monospace;font-size:10px;
                            color:{MUTE};margin:12px 0 8px">semantically close:</div>
                """, unsafe_allow_html=True)
                nb_cols = st.columns(3)
                for i, (_, row) in enumerate(nb.iterrows()):
                    with nb_cols[i]:
                        st.markdown(f"""
                        <div style="background:{CARD};border:1px solid {LINE};
                                    padding:10px;text-align:center">
                            <div style="font-family:'Press Start 2P',monospace;
                                        font-size:9px;color:{GREEN}">{row['neighbour']}</div>
                            <div style="font-family:'VT323',monospace;
                                        font-size:17px;color:{MUTE}">{row['similarity']:.3f}</div>
                        </div>
                        """, unsafe_allow_html=True)

        # other matches
        st.markdown(f"""
        <div style="font-family:'Press Start 2P',monospace;font-size:10px;
                    color:{MUTE};margin:16px 0 8px">other matches:</div>
        """, unsafe_allow_html=True)

        for rank, idx in enumerate(top_indices[1:], 2):
            aesthetic = kb.iloc[idx]['aesthetic']
            score = sims[idx]
            desc = kb.iloc[idx]['description'][:200]
            url = kb.iloc[idx]['url']
            with st.expander(f"#{rank} — {aesthetic}  ({score:.3f})"):
                st.write(desc + "...")
                st.markdown(f"[→ aesthetics wiki]({url})")

        # map showing where user lands
        st.markdown("---")
        st.markdown(f"""
        <div style="font-family:'Press Start 2P',monospace;font-size:11px;
                    color:{GREEN};margin-bottom:12px">where you land on the map</div>
        """, unsafe_allow_html=True)

        # Place "you" at the similarity-weighted centre of your top matches'
        # precomputed coordinates. The previous version refit UMAP on all
        # embeddings on every query (single-threaded due to random_state) —
        # that was the multi-minute hang. The map already uses these same
        # precomputed coords, so this is instant and lands you among your matches.
        coord_lookup = coords.drop_duplicates('aesthetic').set_index('aesthetic')
        pts, wts = [], []
        for idx in top_indices:
            a = kb.iloc[idx]['aesthetic']
            if a in coord_lookup.index:
                pts.append([coord_lookup.loc[a, 'x'], coord_lookup.loc[a, 'y']])
                wts.append(max(float(sims[idx]), 0.0))
        if pts:
            pts = np.asarray(pts, dtype=float)
            wts = np.asarray(wts, dtype=float)
            if wts.sum() == 0:
                wts = np.ones(len(wts))
            user_coord = np.average(pts, axis=0, weights=wts).reshape(1, 2)
        else:
            user_coord = np.array([[float(coords['x'].mean()),
                                    float(coords['y'].mean())]])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=coords['x'], y=coords['y'],
            mode='markers',
            text=coords['aesthetic'],
            marker=dict(size=5, color='rgba(150,150,150,0.35)'),
            hovertemplate="<b>%{text}</b><extra></extra>",
            name='all aesthetics'
        ))
        for idx in top_indices:
            row = coords[coords['aesthetic'] == kb.iloc[idx]['aesthetic']]
            if len(row) > 0:
                fig.add_trace(go.Scatter(
                    x=row['x'], y=row['y'],
                    mode='markers+text',
                    text=[kb.iloc[idx]['aesthetic']],
                    textposition='top center',
                    textfont=dict(size=9, color=INK),
                    marker=dict(size=14, color=GREEN, symbol='star',
                                line=dict(width=1, color='white')),
                    name=kb.iloc[idx]['aesthetic']
                ))
        fig.add_trace(go.Scatter(
            x=[user_coord[0, 0]], y=[user_coord[0, 1]],
            mode='markers+text',
            text=['YOU'],
            textposition='top center',
            textfont=dict(size=11, color=ROSE),
            marker=dict(size=20, color=ROSE, symbol='diamond',
                        line=dict(width=2, color='white')),
            name='your vibe'
        ))
        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(color=INK), height=480,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False, margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LANDSCAPE MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ landscape map":
    st.markdown(f"""
    <div class="pixel-hero">
        <h1>THE LANDSCAPE</h1>
        <div class="sub">657 aesthetics · placed by meaning<span class="cursor"></span></div>
    </div>
    """, unsafe_allow_html=True)

    search = st.text_input("search an aesthetic:", placeholder="e.g. cottagecore")

    families = sorted(coords['super_family'].unique())
    selected_families = st.multiselect(
        "filter by family:", families, default=families
    )

    filtered = coords[coords['super_family'].isin(selected_families)]

    fig = go.Figure()
    # draw Unclustered first/underneath and dimmed so the named families pop
    fam_order = ([f for f in selected_families if f == 'Unclustered'] +
                 [f for f in selected_families if f != 'Unclustered'])
    for fam in fam_order:
        sub = filtered[filtered['super_family'] == fam]
        is_unc = fam == 'Unclustered'
        color = FAMILY_COLORS.get(fam, '#6b665a')
        fig.add_trace(go.Scatter(
            x=sub['x'], y=sub['y'],
            mode='markers', name=fam,
            text=sub['aesthetic'],
            marker=dict(size=6 if is_unc else 9,
                        color=color, opacity=0.3 if is_unc else 0.9,
                        symbol='square',
                        line=dict(width=0.5, color='#14110d')),
            hovertemplate="<b>%{text}</b><br>" + fam + "<extra></extra>"
        ))

    if search:
        match = coords[coords['aesthetic'].str.lower().str.contains(search.lower(), na=False)]
        if len(match) > 0:
            fig.add_trace(go.Scatter(
                x=match['x'], y=match['y'],
                mode='markers+text',
                text=match['aesthetic'],
                textposition='top center',
                textfont=dict(size=10, color=INK),
                marker=dict(size=16, color=ROSE, symbol='star',
                            line=dict(width=2, color='white')),
                name='search result'
            ))
            st.success(f"found {len(match)} match(es)")
        else:
            st.warning(f"no aesthetic matching '{search}'")

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=INK), height=650,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(font=dict(size=11), bgcolor='rgba(0,0,0,0.5)',
                    bordercolor=LINE, borderwidth=1),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 trend analysis":
    st.markdown(f"""
    <div class="pixel-hero">
        <h1>RISE & FALL</h1>
        <div class="sub">google search interest · 2018–2026<span class="cursor"></span></div>
    </div>
    """, unsafe_allow_html=True)

    if trends_df is not None:
        selected = st.multiselect(
            "select aesthetics:",
            options=trends_df.columns.tolist(),
            default=['cottagecore', 'dark academia', 'quiet luxury',
                     'y2k fashion', 'streetwear', 'coquette aesthetic']
        )

        if selected:
            fig = go.Figure()
            palette = [GREEN, ROSE, GOLD, BLUE, PURPLE, PINK,
                       "#8fae7a", "#c0925a", "#cf6f7e", "#c97f5a"]

            for i, aesthetic in enumerate(selected):
                fig.add_trace(go.Scatter(
                    x=trends_df.index,
                    y=trends_df[aesthetic],
                    mode='lines', name=aesthetic,
                    line=dict(width=2.5, color=palette[i % len(palette)],
                              shape='spline'),
                    hovertemplate=f"<b>{aesthetic}</b><br>%{{x|%b %Y}}: %{{y}}<extra></extra>"
                ))

            # covid shading
            fig.add_vrect(
                x0="2020-03-01", x1="2021-06-01",
                fillcolor=ROSE, opacity=0.08, line_width=0
            )
            fig.add_annotation(
                x="2020-10-01", y=106,
                text="covid lockdowns",
                showarrow=False,
                font=dict(color=ROSE, size=13, family="VT323, monospace")
            )

            fig.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(color=INK, family="VT323, monospace"),
                height=480,
                xaxis=dict(title="year", showgrid=True,
                           gridcolor=LINE, tickformat='%Y'),
                yaxis=dict(title="search interest (0-100)",
                           showgrid=True, gridcolor=LINE, range=[0, 112]),
                legend=dict(bgcolor='rgba(0,0,0,0.5)',
                            bordercolor=LINE, borderwidth=1),
                hovermode='x unified',
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

            # peak table
            st.markdown(f"""
            <div style="font-family:'Press Start 2P',monospace;font-size:10px;
                        color:{GREEN};margin:16px 0 10px">peak interest</div>
            """, unsafe_allow_html=True)

            peak_data = []
            for aesthetic in selected:
                peak_date = trends_df[aesthetic].idxmax()
                peak_val = int(trends_df[aesthetic].max())
                peak_data.append({
                    'aesthetic': aesthetic,
                    'peak month': peak_date.strftime('%B %Y'),
                    'score': peak_val
                })
            st.dataframe(
                pd.DataFrame(peak_data),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("trends data not found. run src/temporal_trends.py first.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AESTHETIC DISTANCES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗 aesthetic distances":
    st.markdown(f"""
    <div class="pixel-hero">
        <h1>AESTHETIC DISTANCES</h1>
        <div class="sub">who sits next to whom<span class="cursor"></span></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div style="font-family:'Press Start 2P',monospace;font-size:10px;
                    color:{GREEN};margin-bottom:10px">find neighbours</div>
        """, unsafe_allow_html=True)

        if dist_df is not None:
            selected_aesthetic = st.selectbox(
                "choose an aesthetic:",
                options=dist_df.index.tolist()
            )
            if selected_aesthetic:
                similar = dist_df[selected_aesthetic].drop(selected_aesthetic).nlargest(8)
                fig = go.Figure(go.Bar(
                    x=similar.values,
                    y=similar.index,
                    orientation='h',
                    marker=dict(
                        color=similar.values,
                        colorscale=[[0, BLUE], [0.5, GREEN], [1.0, GOLD]],
                        showscale=False
                    ),
                    hovertemplate="%{y}<br>similarity: %{x:.3f}<extra></extra>"
                ))
                fig.update_layout(
                    paper_bgcolor=BG, plot_bgcolor=BG,
                    font=dict(color=INK, family="VT323, monospace"),
                    height=380,
                    xaxis=dict(title="cosine similarity",
                               showgrid=True, gridcolor=LINE, range=[0, 1]),
                    yaxis=dict(showgrid=False),
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div style="font-family:'Press Start 2P',monospace;font-size:10px;
                    color:{GREEN};margin-bottom:10px">similarity matrix</div>
        """, unsafe_allow_html=True)

        if dist_df is not None:
            scale = [[0.0, "#1c1a14"], [0.35, "#3f5d4a"],
                     [0.65, GREEN], [0.85, ROSE], [1.0, GOLD]]
            fig = go.Figure(go.Heatmap(
                z=dist_df.values,
                x=dist_df.columns,
                y=dist_df.index,
                colorscale=scale,
                zmin=0, zmax=1,
                xgap=2, ygap=2,
                hovertemplate="%{y} ↔ %{x}<br>similarity: %{z:.3f}<extra></extra>"
            ))
            fig.update_layout(
                paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(color=INK, family="VT323, monospace"),
                height=500,
                xaxis=dict(tickfont=dict(size=9), tickangle=-45, showgrid=False),
                yaxis=dict(tickfont=dict(size=9), showgrid=False,
                           autorange="reversed"),
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)