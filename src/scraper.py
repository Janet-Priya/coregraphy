import requests
import re
import pandas as pd
import time
import os

API_URL = "https://aesthetics.fandom.com/api.php"
KB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "aesthetics_kb.csv")

def get_all_aesthetic_pages():
    pages = []
    params = {
        "action": "query",
        "list": "allpages",
        "aplimit": "500",
        "apnamespace": "0",
        "format": "json"
    }
    while True:
        response = requests.get(API_URL, params=params,timeout=10)
        data = response.json()
        pages.extend(data['query']['allpages'])
        if 'continue' in data:
            params['apcontinue'] = data['continue']['apcontinue']
        else:
            break
        time.sleep(0.5)
    return pages

def clean_wikitext(raw):
    # remove templates {{...}}
    raw = re.sub(r'\{\{[^}]*\}\}', '', raw)
    # remove file/image links
    raw = re.sub(r'\[\[File:[^\]]*\]\]', '', raw)
    raw = re.sub(r'\[\[Image:[^\]]*\]\]', '', raw)
    # convert [[link|text]] to just text
    raw = re.sub(r'\[\[[^\]]*\|([^\]]*)\]\]', r'\1', raw)
    # convert [[link]] to just link text
    raw = re.sub(r'\[\[([^\]]*)\]\]', r'\1', raw)
    # remove external links
    raw = re.sub(r'\[http[^\]]*\]', '', raw)
    # remove headers == text ==
    raw = re.sub(r'={2,}[^=]+=={2,}', '', raw)
    # remove bold/italic
    raw = re.sub(r"'{2,}", '', raw)
    # remove HTML tags
    raw = re.sub(r'<[^>]+>', '', raw)
    # remove citation markers
    raw = re.sub(r'\[\d+\]', '', raw)
    # collapse whitespace
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    raw = re.sub(r'[ \t]+', ' ', raw)
    return raw.strip()

def scrape_aesthetic_page(title):
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json"
    }

    response = requests.get(API_URL, params=params,timeout=10)
    data = response.json()

    pages = data['query']['pages']
    page = next(iter(pages.values()))

    if 'revisions' not in page:
        return None

    raw = page['revisions'][0]['slots']['main']['*']
    description = clean_wikitext(raw)

    if len(description) < 100:
        return None

    return {
        "aesthetic": title,
        "description": description,
        "url": f"https://aesthetics.fandom.com/wiki/{title.replace(' ', '_')}"
    }

def build_knowledge_base():
    print("Fetching all aesthetic pages...")
    pages = get_all_aesthetic_pages()
    print(f"Found {len(pages)} pages")

    skip_exact = [
        'Main Page', 'List of Aesthetics', 'Guidelines', 'Community',
        'Help', 'Rules', '-core', '-ism', '-punk', '-wave'
    ]
    skip_prefixes = [
        '190', '191', '192', '193', '194', '195', '196', '197', '198',
        'List', 'Help', 'Rules', 'Guide', 'Main', 'Community',
        'Template', 'User', 'Talk'
    ]

    pages = [
        p for p in pages
        if p['title'] not in skip_exact
        and not any(p['title'].startswith(prefix) for prefix in skip_prefixes)
        and not p['title'][0].isdigit()
    ]
    print(f"Filtered to {len(pages)} relevant pages")

    already_done = set()
    results = []

    if os.path.exists(KB_PATH) and os.path.getsize(KB_PATH) > 0:
        existing = pd.read_csv(KB_PATH)
        already_done = set(existing['aesthetic'].tolist())
        results = existing.to_dict('records')
        print(f"Resuming — {len(already_done)} already scraped")

    for i, page in enumerate(pages):
        title = page['title']

        if title in already_done:
            continue

        print(f"[{i+1}/{len(pages)}] Scraping: {title}")

        try:
            data = scrape_aesthetic_page(title)
            length = len(data['description']) if data else 0
            print(f"  length: {length}")
            if data and length > 100:
                results.append(data)
        except Exception as e:
            print(f"  Skipping {title} — {e}")
            continue

        if len(results) > 0 and len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(KB_PATH, index=False)
            print(f"  Checkpoint saved — {len(results)} aesthetics so far")

        time.sleep(0.3)

    df = pd.DataFrame(results)
    df.to_csv(KB_PATH, index=False)
    print(f"Done — saved {len(df)} aesthetics to {KB_PATH}")
    return df

if __name__ == "__main__":
    df = build_knowledge_base()
    print(df.head())