# 🧵 The Aesthetic Atlas

**Mapping the fashion-aesthetic landscape with NLP — how online style subcultures cluster, relate, and rise and fall over time (2018–2026).**

> Cottagecore, dark academia, coquette, gorpcore, Y2K… the 2020s turned fashion into a vocabulary of *aesthetics*. This project treats that vocabulary as data: it scrapes the aesthetics people actually name, embeds them with a language model, lets the model discover how they cluster, and cross-references those clusters against 8 years of search trends and real purchase reviews.

It is **not** a single model on a clean dataset — it's an end-to-end pipeline: **scrape → embed → cluster → reduce → cross-reference → interactive app**.

---

## 📸 Demo

> _Add a screenshot or 20-second GIF of the Aesthetic Finder + Landscape Map here._
>
> `docs/finder.gif` · `docs/map.png`

**Two ways to explore it:**
- 🎛️ **Interactive app** (Streamlit) — a semantic "find your aesthetic" search, the landscape map, trend charts, and a similarity matrix.
- 🖼️ **Static dashboard** (`outputs/dashboard.html`) — a single self-contained, cozy pixel-art page summarising the findings. Opens in any browser, no server needed.

---

## 🔑 Key findings

1. **COVID drove escapist aesthetics.** *Cottagecore* and *dark academia* rose from near-zero in early 2020 and peaked in 2020–21 — tracking the lockdown timeline almost exactly (cottagecore search interest peaked Nov 2020).
2. **Aesthetics self-organise into semantic families — unsupervised.** Given only text descriptions, the model grouped *Cozycore · Bloomcore · Gardencore · Craftcore* into one tight cluster without being told they were related. The same happened for the Goth family and the Cyber family.
3. **A post-2022 turn from "escape" to "confidence & nostalgia."** Interest slid away from nature/academia aesthetics toward *coquette, Y2K,* and *gorpcore*.
4. **Streetwear is the only evergreen.** It held high, steady interest across the entire 8-year window while every other aesthetic rose and fell.
5. **Aesthetic identity ≠ purchase language.** Amazon fashion reviews talk *fit, size, comfort, quality* — almost never aesthetic terms. How we perform style as identity barely overlaps with how we actually shop.

---

## 🏗️ How it works

```
                 ┌──────────────────────────────────────────────────────────┐
  DATA SOURCES   │  Aesthetics Wiki     Tumblr posts     Amazon reviews      │
                 │  (655 aesthetics,    (by tag, dated)   (fashion meta +     │
                 │   descriptions)                         review text)       │
                 └───────────┬───────────────┬─────────────────┬─────────────┘
                             │               │                 │
   EMBED                     ▼               ▼                 ▼
   (sentence-transformers)   text  ──►  dense vector embeddings (MPNet / MiniLM)
                             │
   CLUSTER + REDUCE          ▼
   (BERTopic, UMAP,          topics ──► 2-D coordinates ──► "the landscape map"
    HDBSCAN)                 │
   RELATE                    ▼
   (cosine similarity)       aesthetic-to-aesthetic distance matrix + neighbours
                             │
   CROSS-REFERENCE           ▼
   (Google Trends,           cultural popularity over time  +  purchase-language gap
    review topics)
                             │
   SERVE                     ▼
                    Streamlit app  +  static pixel-art dashboard
```

**The semantic finder** embeds your free-text description with the same model used on the
knowledge base, then ranks all aesthetics by cosine similarity — so "dark moody academic,
tweed, candlelight" returns *Academia / Dark Academia*, not a keyword match.

**The landscape map** positions each aesthetic by meaning (UMAP over the embeddings) and
colours it by its discovered family. **Position is the model's real output**; colour is a
human-readable annotation on top.

---

## 🧰 Tech stack

| Layer | Tools |
|---|---|
| Embeddings | `sentence-transformers` (`all-mpnet-base-v2`, `all-MiniLM-L6-v2`) |
| Topic modelling / clustering | `BERTopic`, `HDBSCAN` |
| Dimensionality reduction | `UMAP` |
| Similarity / analysis | `scikit-learn`, `numpy`, `pandas` |
| Trends | Google Trends |
| App & viz | `Streamlit`, `Plotly` |
| Static dashboard | hand-built HTML/CSS + hand-authored pixel art (SVG) |

---

## 📁 Repository structure

```
vibe_clustering/
├── app.py                     # Streamlit app (finder, map, trends, distances)
├── README.md
├── requirements.txt
├── src/
│   ├── scraper.py             # scrape the Aesthetics Wiki knowledge base
│   ├── kb_clustering.py       # embed KB → BERTopic → UMAP coords
│   ├── aesthetic_distance.py  # similarity matrix + nearest neighbours
│   ├── tumblr_pipeline.py     # embed Tumblr posts by aesthetic tag
│   ├── review_pipeline.py     # cluster Amazon fashion reviews
│   ├── temporal_trends.py     # popularity-over-time analysis
│   ├── build_dashboard.py     # generate the static pixel-art dashboard
│   ├── pixel_art.py           # hand-authored pixel art → SVG
│   └── ...
├── data/
│   ├── raw/                   # scraped knowledge base, Tumblr, reviews
│   └── processed/             # embeddings, coords, matrices (regenerable)
└── outputs/
    └── dashboard.html         # the static cozy dashboard
```

> **Family classification lives in one place** (`src/build_dashboard.py`) and is imported by
> the app, so the map colours never drift between the two views.

---

## ▶️ Running it

```bash
# 1. install
pip install -r requirements.txt

# 2. (optional) regenerate the processed artifacts from raw data
python src/kb_clustering.py        # embeddings + map coordinates
python src/aesthetic_distance.py   # similarity matrix + neighbours
python src/temporal_trends.py      # trends

# 3a. launch the interactive app
streamlit run app.py               # → http://localhost:8501

# 3b. or rebuild the static dashboard
python src/build_dashboard.py      # → outputs/dashboard.html
```

The app loads precomputed embeddings (`data/processed/kb_embeddings.npy`); on first run
without them, it generates them once and caches the result.

---

## ⚠️ Limitations (read this — it's the honest part)

- **Trend signal is Tumblr + Google Trends.** Pinterest and Instagram have locked down their
  APIs, so the social-trend side leans on Tumblr (where aesthetic language is richest) plus
  search interest. It's a real but partial window into culture, not the whole internet.
- **Clustering has a floor.** ~45% of aesthetics land in an "Unclustered" bucket — mostly
  BERTopic's generic catch-all topic and flagged outliers. Rather than invent labels for
  them, the map leaves them grey. The coloured families are the trustworthy part.
- **Colour follows the cluster, not the name.** Because aesthetics are coloured by their
  *topic*, a famous aesthetic can occasionally sit outside its "obvious" family if its topic
  was the generic one. The 2-D *position* (from the embeddings) is the reliable signal.
- **Embeddings see the intro, not the whole article.** Wiki descriptions are long; the
  embedding model caps input length, so matches are driven by each aesthetic's opening
  summary. Fine in practice, but deep-body details don't influence results.
- **Reviews are a proxy for purchase behaviour**, not a controlled study of identity vs. spend.
- **The map's families and the trend lines are deliberately separate.** Families are an
  unsupervised grouping of the 654 knowledge-base aesthetics (used to colour the map). The
  trends track ~20 specific aesthetic search terms over time. They come from different data
  and aren't grouped by family — so the map shows families while the trends show individual
  aesthetics by design, not by oversight.

---

## 🔭 Future work

- Add a stronger purchase-aesthetic bridge (e.g. **Depop** listings, where sellers use exact
  aesthetic language on real items for sale).
- Name-first family colouring as an option, so recognisable aesthetics always colour correctly.
- Deploy the app (Streamlit Community Cloud) and the dashboard (GitHub Pages).

---

*Built as a data/ML portfolio project — full pipeline from raw scrape to interactive product.*
# coregraphy
