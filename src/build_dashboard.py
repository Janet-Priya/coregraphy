"""Build a self-contained cozy pixel-art dashboard from the pipeline outputs.

Reads the processed CSVs and emits outputs/dashboard.html — a single static
file (Plotly from CDN) you can open locally or host on GitHub Pages.

    python src/build_dashboard.py
"""
from __future__ import annotations

import os
import pandas as pd
import plotly.graph_objects as go

import pixel_art as pa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
OUT = os.path.join(ROOT, "outputs", "dashboard.html")

# ---- palette --------------------------------------------------------------
BG = "#161310"
CARD = "#201d17"
CARD2 = "#26221a"
INK = "#ece4d2"
MUTE = "#9a917b"
LINE = "#3b3a2e"
GREEN = "#83a85f"
ROSE = "#d98b78"
GOLD = "#d8b25e"
BLUE = "#7c9fb0"
PURPLE = "#9a86b8"
PINK = "#e0a6a0"

FAMILY_COLORS = {
    "Academia": "#c0925a",
    "Goth & Dark": PURPLE,
    "Cyber & Tech": BLUE,
    "Punk & Music": "#b34a4a",
    "Cottage & Nature": GREEN,
    "Cute & Coquette": PINK,
    "Doll & Performance": "#c58fb0",
    "Street & Sport": GOLD,
    "Corporate & Minimal": "#a9b2a0",
    "East Asian": "#cf6f7e",
    "Regional & Cultural": "#5fa39a",
    "Art & Design": "#c97f5a",
    "Retro & Y2K": "#8fae7a",
    "Unclustered": "#6b665a",
}

# Order matters: first matching rule wins. Cyber & Tech sits *before* Punk & Music
# so cyber/steam/dieselpunk stay tech and only bare "punk" falls to music. Goth
# before Clean so "corporate goth" reads as goth, not corporate.
SUPER_RULES = [
    ("Academia", ["academia"]),
    ("Goth & Dark", ["goth", "dark", "horror", "catholic", "supernatural", "emo", "witch", "occult"]),
    ("Cyber & Tech", ["cyber", "tech", "aero", "frutiger", "grid", "laser", "diesel",
                      "vapor", "steampunk", "steam"]),
    ("Punk & Music", ["punk", "industrial", "ebm", "gabber", "phonk", "metal", "glam",
                      "bands", "kandi", "plur", "rave", "britpop", "synth", "dungeon",
                      "scene", "hardcore"]),
    ("Cottage & Nature", ["cottage", "cozy", "hygge", "cabin", "fairy", "garden", "bloom",
                          "mushroom", "forest", "nature", "goblin", "farm", "plant"]),
    ("Cute & Coquette", ["candy", "kawaii", "cute", "girly", "coquette", "sweet", "pink",
                         "lolita", "bubblegum", "balletcore", "soft", "delicate", "hime"]),
    ("Doll & Performance", ["doll", "kitsch", "burlesque", "vegas", "ballet", "performer",
                            "dance", "maid"]),
    ("Street & Sport", ["street", "hype", "blok", "sport", "gorp", "skate", "athleisure"]),
    ("Corporate & Minimal", ["corporate", "business", "minimal", "office", "formal"]),
    ("East Asian", ["japan", "chinese", "korea", "dynasty", "rednote", "bodikon", "hanbok"]),
    ("Regional & Cultural", ["indigenous", "arabian", "desert", "greek", "greece", "divine",
                             "chonga", "abg", "chicha", "cumbia", "gaucho", "bosozoku",
                             "lowrider", "biker", "spanish", "cani", "mdlr", "latin"]),
    ("Art & Design", ["art", "design", "movement", "architecture", "abstract", "acid", "maximal"]),
    ("Retro & Y2K", ["y2k", "retro", "vintage", "indie", "grunge", "2000", "nostalg"]),
]


def super_family(label: str) -> str:
    low = str(label).lower()
    for name, keys in SUPER_RULES:
        if any(k in low for k in keys):
            return name
    return "Unclustered"


def cozy(fig: go.Figure, height: int = 460, legend=True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="VT323, monospace", size=18, color=INK),
        margin=dict(l=20, r=20, t=20, b=20),
        hoverlabel=dict(font=dict(family="VT323, monospace", size=18),
                        bgcolor=CARD2, bordercolor=GOLD),
        showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=17, color=INK),
                    orientation="h", yanchor="bottom", y=1.0, x=0),
        colorway=[GREEN, ROSE, GOLD, BLUE, PURPLE, "#c97f5a", PINK, "#8fae7a"],
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                     tickfont=dict(size=15, color=MUTE), title_font=dict(size=17, color=MUTE))
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                     tickfont=dict(size=15, color=MUTE), title_font=dict(size=17, color=MUTE))
    return fig


def div(fig: go.Figure, name: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       div_id=name, config={"displayModeBar": False, "responsive": True})


# ---- figures --------------------------------------------------------------

def fig_map() -> str:
    df = pd.read_csv(os.path.join(PROC, "kb_with_coords.csv"))
    df["fam"] = df["family_label"].map(super_family)
    order = sorted(df["fam"].unique(), key=lambda f: -(df["fam"] == f).sum())
    fig = go.Figure()
    # draw "Unclustered" first/underneath and dimmed so named families pop
    order = (["Unclustered"] if "Unclustered" in order else []) + \
            [f for f in order if f != "Unclustered"]
    for fam in order:
        sub = df[df["fam"] == fam]
        is_other = fam == "Unclustered"
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers", name=fam,
            marker=dict(size=6 if is_other else 9,
                        color=FAMILY_COLORS[fam], opacity=0.28 if is_other else 0.92,
                        line=dict(width=0.6, color="#14110d"), symbol="square"),
            text=sub["aesthetic"],
            hovertemplate="<b>%{text}</b><br>" + fam + "<extra></extra>",
        ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return div(cozy(fig, height=520), "map")


def fig_trends() -> str:
    df = pd.read_csv(os.path.join(PROC, "google_trends.csv"), parse_dates=["date"])
    show = {
        "cottagecore": GREEN, "dark academia": "#c0925a", "y2k fashion": PINK,
        "coquette aesthetic": ROSE, "gorpcore": BLUE, "streetwear": GOLD,
    }
    fig = go.Figure()
    for col, c in show.items():
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col], mode="lines", name=col,
            line=dict(color=c, width=2.5, shape="spline"),
            hovertemplate="%{x|%b %Y}<br>" + col + ": %{y}<extra></extra>",
        ))
    fig.add_vrect(x0="2020-03-01", x1="2021-06-01", fillcolor=ROSE, opacity=0.08,
                  line_width=0)
    fig.add_annotation(x="2020-10-01", y=104, text="lockdown escapism",
                       showarrow=False, font=dict(color=ROSE, size=16))
    fig.update_yaxes(title="search interest", range=[0, 112])
    return div(cozy(fig, height=440), "trends")


def fig_heatmap() -> str:
    df = pd.read_csv(os.path.join(PROC, "aesthetic_distance_matrix.csv"), index_col=0)
    scale = [[0.0, "#1c1a14"], [0.35, "#3f5d4a"], [0.65, GREEN], [0.85, ROSE], [1.0, GOLD]]
    fig = go.Figure(go.Heatmap(
        z=df.values, x=df.columns, y=df.index, colorscale=scale, zmin=0, zmax=1,
        xgap=2, ygap=2,
        colorbar=dict(title="sim", thickness=12, tickfont=dict(size=13, color=MUTE),
                      outlinewidth=0),
        hovertemplate="%{y} ↔ %{x}<br>similarity %{z:.2f}<extra></extra>",
    ))
    fig.update_xaxes(tickfont=dict(size=11, color=MUTE), tickangle=-55, showgrid=False)
    fig.update_yaxes(tickfont=dict(size=11, color=MUTE), showgrid=False, autorange="reversed")
    return div(cozy(fig, height=620, legend=False), "heatmap")


# ---- pixel helpers --------------------------------------------------------

def progress(value: int, color: str, cells: int = 14) -> str:
    filled = round(value / 100 * cells)
    out = []
    for i in range(cells):
        c = color if i < filled else LINE
        out.append(f'<span class="pcell" style="background:{c}"></span>')
    return f'<span class="pbar">{"".join(out)}</span>'


def scatter_decor() -> str:
    """A few pixel decorations sprinkled around the page."""
    bits = [
        ("flower", pa.FLOWER, "top:78px;left:6%", 3),
        ("mush", pa.MUSHROOM, "top:120px;right:5%", 3),
        ("star1", pa.STAR, "top:60px;left:40%", 2),
        ("star2", pa.STAR, "top:160px;right:30%", 2),
        ("leaf2", pa.LEAF, "top:230px;left:3%", 2),
    ]
    out = []
    for name, art, pos, sc in bits:
        out.append(f'<div class="decor" style="{pos}">{pa.to_svg(art, sc)}</div>')
    return "".join(out)


def garland() -> str:
    leaf = pa.to_svg(pa.LEAF, 3)
    n = 22
    cells = "".join(f'<span class="gleaf" style="--i:{i}">{leaf}</span>' for i in range(n))
    return f'<div class="garland">{cells}</div>'


# ---- findings -------------------------------------------------------------

FINDINGS = [
    ("covid drove escapism",
     "cottagecore & dark academia rose from near-zero in 2020 and peaked 2021 — straight off the lockdown curve.", GREEN),
    ("aesthetics self-organise",
     "the model grouped cozycore · bloomcore · gardencore · craftcore with no labels. same for the goth and cyber families.", ROSE),
    ("the post-2022 turn",
     "interest slid from nature & academia toward confidence & nostalgia — coquette, y2k, gorpcore.", GOLD),
    ("streetwear is evergreen",
     "the only tag holding high interest across all 8 years while everything else rises and falls.", BLUE),
    ("identity ≠ basket",
     "amazon reviews talk fit, size, comfort — not aesthetics. how we dress as identity barely touches how we shop.", PURPLE),
]

REVIEW_WORDS = ["fit", "size", "comfortable", "quality", "wear", "color", "leather", "nice"]
TAG_WORDS = ["cottagecore", "coquette", "y2k", "gorpcore", "balletcore", "goth", "fairycore"]


def build():
    fig_map_html = fig_map()
    fig_trends_html = fig_trends()
    fig_heat_html = fig_heatmap()

    monitor = pa.to_svg(pa.MONITOR, 7, with_screen=True)
    plant = pa.to_svg(pa.PLANT, 6)
    mug = pa.to_svg(pa.MUG, 6)

    find_list = "".join(
        f'<li><span class="dot" style="background:{c}"></span>'
        f'<span class="fl-t">{t}</span></li>' for t, _, c in FINDINGS
    )
    find_cards = "".join(
        f'<div class="finding"><div class="f-head"><span class="dot" style="background:{c}">'
        f'</span><span class="f-title">{t}</span></div>'
        f'<p class="f-body">{b}</p></div>' for t, b, c in FINDINGS
    )
    bars = (
        f'<div class="bar-row"><span class="bar-lab">lockdown escapism</span>{progress(100, GREEN)}<span class="bar-pct">100</span></div>'
        f'<div class="bar-row"><span class="bar-lab">post-2022 nostalgia</span>{progress(72, GOLD)}<span class="bar-pct">72</span></div>'
        f'<div class="bar-row"><span class="bar-lab">streetwear evergreen</span>{progress(91, BLUE)}<span class="bar-pct">91</span></div>'
    )
    tag_chips = "".join(f'<span class="chip chip-a">{w}</span>' for w in TAG_WORDS)
    rev_chips = "".join(f'<span class="chip chip-r">{w}</span>' for w in REVIEW_WORDS)

    html = PAGE.format(
        css=CSS, garland=garland(), decor=scatter_decor(),
        monitor=monitor, plant=plant, mug=mug,
        find_list=find_list, find_cards=find_cards, bars=bars,
        map=fig_map_html, trends=fig_trends_html, heatmap=fig_heat_html,
        tag_chips=tag_chips, rev_chips=rev_chips,
        leaf_s=pa.to_svg(pa.LEAF, 2),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print("wrote", OUT, f"({len(html)//1024} kb)")


# ---- template -------------------------------------------------------------

CSS = """
:root{--bg:#161310;--card:#201d17;--card2:#26221a;--ink:#ece4d2;--mute:#9a917b;
--line:#3b3a2e;--green:#83a85f;--rose:#d98b78;--gold:#d8b25e;--blue:#7c9fb0;--purple:#9a86b8;}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:'VT323',monospace;font-size:21px;line-height:1.25;
background-image:radial-gradient(circle at 18% 12%,rgba(131,168,95,.05),transparent 38%),
radial-gradient(circle at 82% 78%,rgba(217,139,120,.05),transparent 40%);}
svg{image-rendering:pixelated}
.pix{font-family:'Press Start 2P',monospace;letter-spacing:.5px}
.wrap{max-width:1140px;margin:0 auto;padding:0 22px 80px;position:relative}

.garland{display:flex;justify-content:center;gap:0;overflow:hidden;height:46px;
margin-bottom:6px}
.gleaf{display:inline-block;transform:translateY(calc(sin(var(--i))*0px));opacity:.92}
.gleaf:nth-child(odd){transform:translateY(6px) scaleX(-1)}
.decor{position:absolute;z-index:0;opacity:.85;pointer-events:none}

.hero{text-align:center;padding:6px 0 26px;position:relative;z-index:1}
.hero .orb{display:flex;justify-content:center;margin-bottom:10px}
.hero h1{font-family:'Press Start 2P',monospace;font-size:30px;color:var(--ink);
margin:6px 0 12px;text-shadow:3px 3px 0 #0c0a08}
.hero .sub{color:var(--gold);font-size:24px;letter-spacing:1px}
.hero .quote{color:var(--mute);font-size:21px;margin-top:8px}
.cursor{display:inline-block;width:11px;height:20px;background:var(--green);
margin-left:3px;animation:blink 1s steps(1) infinite;vertical-align:-2px}
@keyframes blink{50%{opacity:0}}

.card{background:var(--card);border:2px solid var(--line);padding:18px 20px;
box-shadow:0 0 0 2px #0c0a08, 5px 5px 0 0 rgba(0,0,0,.35);position:relative}
.card::before,.card::after{content:'';position:absolute;width:7px;height:7px;
background:var(--gold)}
.card::before{top:-2px;left:-2px}.card::after{bottom:-2px;right:-2px}
.card h2{font-family:'Press Start 2P',monospace;font-size:14px;color:var(--green);
margin:0 0 14px;display:flex;align-items:center;gap:8px}
.card h2 .tag{color:var(--mute);font-family:'VT323',monospace;font-size:19px;
margin-left:auto;letter-spacing:0}

.grid{display:grid;gap:20px}
.g-quest{grid-template-columns:1fr 1.25fr;align-items:stretch}
.g-find{grid-template-columns:.85fr 1.3fr;margin-top:20px}
.g-bottom{grid-template-columns:1fr;margin-top:20px}
.g-disc{grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:820px){.g-quest,.g-find,.g-disc{grid-template-columns:1fr}}

.stats{display:grid;grid-template-columns:1fr 1fr;gap:12px;height:100%}
.stat{background:var(--card2);border:1px solid var(--line);padding:12px 14px}
.stat .n{font-family:'Press Start 2P',monospace;font-size:22px;color:var(--gold)}
.stat .l{color:var(--mute);font-size:19px;margin-top:6px}

.scene{display:flex;align-items:flex-end;justify-content:center;gap:14px;
padding:10px 0 4px;min-height:150px}
.scene .lift{transform:translateY(-6px)}
.scene-cap{text-align:center;color:var(--mute);font-size:19px;margin-top:4px}

.flist{list-style:none;margin:0;padding:0}
.flist li{display:flex;align-items:center;gap:10px;padding:9px 6px;
border-bottom:1px dashed var(--line);font-size:21px}
.flist li:last-child{border:0}
.dot{width:10px;height:10px;display:inline-block;flex:0 0 auto;
box-shadow:0 0 0 2px #0c0a08}
.finding{padding:11px 0;border-bottom:1px dashed var(--line)}
.finding:last-child{border:0}
.f-head{display:flex;align-items:center;gap:9px}
.f-title{font-family:'Press Start 2P',monospace;font-size:12px;color:var(--ink)}
.f-body{margin:7px 0 0 19px;color:var(--mute);font-size:20px;line-height:1.3}

.bars{margin-top:14px;border-top:1px dashed var(--line);padding-top:12px}
.bar-row{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:19px}
.bar-lab{color:var(--mute);width:185px}
.bar-pct{color:var(--gold);width:34px;text-align:right}
.pbar{display:inline-flex;gap:3px;flex:1}
.pcell{width:13px;height:14px;display:inline-block}

.section-h{font-family:'Press Start 2P',monospace;font-size:16px;color:var(--ink);
margin:34px 4px 14px;display:flex;align-items:center;gap:12px}
.section-h .ln{flex:1;height:2px;background:var(--line)}
.section-h .sm{color:var(--mute);font-family:'VT323';font-size:20px;letter-spacing:0}

.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.chip{padding:5px 10px;font-size:19px;border:1px solid var(--line)}
.chip-a{color:#0c0a08;background:var(--rose)}
.chip-r{color:var(--ink);background:var(--card2)}
.disc-note{color:var(--mute);font-size:20px;line-height:1.35;margin-top:6px}
.vs{font-family:'Press Start 2P';font-size:11px;color:var(--gold);margin:14px 0 4px}

.foot{text-align:center;color:var(--mute);font-size:19px;margin-top:48px;
border-top:1px dashed var(--line);padding-top:18px}
.foot b{color:var(--green)}
a{color:var(--gold);text-decoration:none}
"""

PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>the aesthetic atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{css}</style></head>
<body>
{garland}
<div class="wrap">
{decor}
<header class="hero">
  <div class="orb">{monitor}</div>
  <h1>THE AESTHETIC ATLAS</h1>
  <div class="sub">mapping fashion vibes &middot; 2018&ndash;2026</div>
  <div class="quote">&ldquo;we'll be in full bloom by the end of this&rdquo;<span class="cursor"></span></div>
</header>

<section class="grid g-quest">
  <div class="card">
    <h2>today's quest <span class="tag">the numbers</span></h2>
    <div class="stats">
      <div class="stat"><div class="n">655</div><div class="l">aesthetics mapped</div></div>
      <div class="stat"><div class="n">29</div><div class="l">core vibe families</div></div>
      <div class="stat"><div class="n">8 yr</div><div class="l">of trend history</div></div>
      <div class="stat"><div class="n">97k</div><div class="l">reviews cross-read</div></div>
    </div>
  </div>
  <div class="card">
    <h2>what's the story? <span class="tag">est. read 2 min</span></h2>
    <div class="scene">
      {plant}
      <div class="lift">{monitor}</div>
      {mug}
    </div>
    <div class="scene-cap">embeddings &rarr; clusters &rarr; a living map of how we dress</div>
  </div>
</section>

<section class="grid g-find">
  <div class="card">
    <h2>the findings</h2>
    <ul class="flist">{find_list}</ul>
  </div>
  <div class="card">
    <h2>currently mapping&hellip; <span class="tag">5 things i'm sure of</span></h2>
    {find_cards}
    <div class="bars">{bars}</div>
  </div>
</section>

<div class="section-h">{leaf_s} the map <span class="sm">655 aesthetics, placed by meaning</span><span class="ln"></span></div>
<div class="card">
  <h2>the landscape <span class="tag">colour = discovered family &middot; hover a square</span></h2>
  {map}
</div>

<div class="section-h">{leaf_s} the trends <span class="sm">search interest, 2018&ndash;2026</span><span class="ln"></span></div>
<div class="card">
  <h2>rise &amp; fall <span class="tag">shaded = covid window</span></h2>
  {trends}
</div>

<div class="section-h">{leaf_s} the families <span class="sm">who sits next to whom</span><span class="ln"></span></div>
<div class="card">
  <h2>similarity matrix <span class="tag">warmer = more alike</span></h2>
  {heatmap}
</div>

<div class="section-h">{leaf_s} the disconnect <span class="sm">identity vs basket</span><span class="ln"></span></div>
<section class="grid g-disc">
  <div class="card">
    <h2>two vocabularies</h2>
    <div class="vs">how we IDENTIFY</div>
    <div class="chips">{tag_chips}</div>
    <div class="vs">how we SHOP</div>
    <div class="chips">{rev_chips}</div>
  </div>
  <div class="card">
    <h2>the gap</h2>
    <p class="disc-note">Aesthetic language &mdash; cottagecore, coquette, gorpcore &mdash; is how
    people <i>describe themselves</i>. But the same people, in their Amazon reviews,
    reach for <i>fit, size, comfortable, quality</i>.</p>
    <p class="disc-note">The story we tell about our style and the basket we
    actually fill barely overlap. Aesthetic identity lives on Tumblr and
    Pinterest; the checkout is purely functional.</p>
  </div>
</section>

<div class="foot">
  built from embeddings &middot; BERTopic clusters &middot; UMAP coordinates &middot; google trends &middot; amazon reviews<br>
  <b>the aesthetic atlas</b> &mdash; a map of how fashion feels
</div>
</div>
</body></html>"""


if __name__ == "__main__":
    build()
