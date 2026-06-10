import pandas as pd
import time
import os
from pytrends.request import TrendReq
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# aesthetics to track — your 20 tumblr tags
AESTHETICS = [
    "dark academia",
    "cottagecore",
    "y2k fashion",
    "quiet luxury",
    "goblincore",
    "coquette aesthetic",
    "fairycore",
    "baddie aesthetic",
    "coastal grandmother",
    "gorpcore",
    "indie sleaze",
    "soft girl aesthetic",
    "grunge aesthetic",
    "kawaii fashion",
    "hypebeast",
    "light academia",
    "craftcore",
    "cyberpunk fashion",
    "balletcore",
    "streetwear"
]

pytrends = TrendReq(hl='en-US', tz=360)

# Google Trends only allows 5 keywords at a time
# batch into groups of 5
def batch(lst, n=5):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

all_data = {}

for group in batch(AESTHETICS, 5):
    print(f"Fetching: {group}")
    try:
        pytrends.build_payload(
            group,
            timeframe='2018-01-01 2026-01-01',
            geo=''
        )
        df = pytrends.interest_over_time()
        if not df.empty:
            df = df.drop(columns=['isPartial'], errors='ignore')
            for col in df.columns:
                all_data[col] = df[col]
        time.sleep(60)  # avoid rate limiting
    except Exception as e:
        print(f"  Error: {e}")
        time.sleep(10)
        continue

trends_df = pd.DataFrame(all_data)
trends_df.index = pd.to_datetime(trends_df.index)
trends_df.to_csv(os.path.join(OUT_DIR, "google_trends.csv"))
print(f"Saved trends data — shape: {trends_df.shape}")
print(trends_df.head())

# ── visualise ─────────────────────────────────────────────────────────────────
print("\nBuilding trend chart...")

# group 1 — nature/cozy aesthetics
group1 = ['cottagecore', 'goblincore', 'fairycore', 'dark academia', 'light academia']
# group 2 — fashion aesthetics  
group2 = ['y2k fashion', 'quiet luxury', 'streetwear', 'hypebeast', 'gorpcore']
# group 3 — feminine aesthetics
group3 = ['coquette aesthetic', 'baddie aesthetic', 'soft girl aesthetic', 'balletcore', 'coastal grandmother']

def make_trend_chart(df, aesthetics, title):
    fig = go.Figure()
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
        '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E9'
    ]
    for i, aesthetic in enumerate(aesthetics):
        if aesthetic in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[aesthetic],
                mode='lines',
                name=aesthetic,
                line=dict(width=2.5, color=colors[i % len(colors)]),
                hovertemplate=f"<b>{aesthetic}</b><br>Date: %{{x}}<br>Interest: %{{y}}<extra></extra>"
            ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        paper_bgcolor='#0f0f0f',
        plot_bgcolor='#0f0f0f',
        font=dict(color='white'),
        width=1200,
        height=500,
        xaxis=dict(
            title="Year",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickformat='%Y'
        ),
        yaxis=dict(
            title="Search Interest (0-100)",
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=1
        ),
        hovermode='x unified'
    )
    return fig

# build and save all three charts
cols = trends_df.columns.tolist()

fig1 = make_trend_chart(trends_df, [a for a in group1 if a in cols], "Nature & Academia Aesthetics — Google Search Trends 2018–2026")
fig1.write_html(os.path.join(OUT_DIR, "trends_nature_academia.html"))

fig2 = make_trend_chart(trends_df, [a for a in group2 if a in cols], "Fashion Aesthetics — Google Search Trends 2018–2026")
fig2.write_html(os.path.join(OUT_DIR, "trends_fashion.html"))

fig3 = make_trend_chart(trends_df, [a for a in group3 if a in cols], "Feminine Aesthetics — Google Search Trends 2018–2026")
fig3.write_html(os.path.join(OUT_DIR, "trends_feminine.html"))

# ── peak year per aesthetic ───────────────────────────────────────────────────
print("\n── PEAK INTEREST BY AESTHETIC ───────────────────────────────────────")
for aesthetic in AESTHETICS:
    if aesthetic in trends_df.columns:
        peak_date = trends_df[aesthetic].idxmax()
        peak_val = trends_df[aesthetic].max()
        print(f"{aesthetic:25} peak: {peak_date.strftime('%Y-%m')} (score: {peak_val})")

print(f"\nSaved 3 trend charts to {OUT_DIR}")