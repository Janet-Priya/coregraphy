import requests
import pandas as pd
import time
import os
from datetime import datetime

API_KEY = "ktOopMBaOTMQvji8gj8MrOALVurscMIGyaAC0MuIr2Du6peh67"  

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)

# aesthetic tags to scrape — these have rich descriptive language
TAGS = [
    "dark academia",
    "cottagecore",
    "y2k fashion",
    "quiet luxury",
    "streetwear",
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
    "balletcore"
]

def scrape_tag(tag, api_key, limit=200):
    posts = []
    before = None
    url = "https://api.tumblr.com/v2/tagged"

    while len(posts) < limit:
        params = {
            "tag": tag,
            "api_key": api_key,
            "limit": 20,
            "filter": "text"  # plain text only, no html
        }
        if before:
            params["before"] = before

        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json()

            if data.get("meta", {}).get("status") != 200:
                print(f"  API error for tag '{tag}': {data.get('meta')}")
                break

            batch = data.get("response", [])
            if not batch:
                break

            for post in batch:
                # get text content
                text = ""
                if post.get("type") == "text":
                    text = post.get("body", "")
                elif post.get("type") == "photo":
                    text = post.get("caption", "")
                elif post.get("type") == "quote":
                    text = post.get("text", "")

                # get tags on the post
                post_tags = " ".join(post.get("tags", []))

                combined = f"{text} {post_tags}".strip()

                if len(combined) > 30:
                    # convert timestamp to year
                    ts = post.get("timestamp", 0)
                    year = datetime.fromtimestamp(ts).year if ts else None

                    posts.append({
                        "aesthetic_tag": tag,
                        "text": combined[:1000],  # cap at 1000 chars
                        "year": year,
                        "post_type": post.get("type"),
                        "note_count": post.get("note_count", 0)
                    })

            # pagination — go before the oldest post
            before = batch[-1].get("timestamp")
            print(f"  [{tag}] collected {len(posts)} posts so far...")
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error scraping '{tag}': {e}")
            break

    return posts

# ── main ──────────────────────────────────────────────────────────────────────
all_posts = []

for tag in TAGS:
    print(f"\nScraping tag: #{tag}")
    posts = scrape_tag(tag, API_KEY, limit=200)
    all_posts.extend(posts)
    print(f"  Done — {len(posts)} posts")
    time.sleep(1)

df = pd.DataFrame(all_posts)
print(f"\nTotal posts collected: {len(df)}")
print(f"Year distribution:\n{df['year'].value_counts().sort_index()}")
print(f"Tag distribution:\n{df['aesthetic_tag'].value_counts()}")


df.to_csv(os.path.join(OUT_DIR, "tumblr_aesthetic_posts.csv"), index=False)
print("Saved to data/raw/tumblr_aesthetic_posts.csv")