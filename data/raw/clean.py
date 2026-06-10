import pandas as pd

df = pd.read_csv("data/raw/aesthetics_kb.csv")

junk = ['__NOTOC__', 'Welcome to', 'The purpose of this page', '#REDIRECT', 'The purpose of this page']
df = df[~df.description.apply(lambda x: any(x.startswith(j) for j in junk))]
df = df.drop_duplicates(subset='aesthetic').reset_index(drop=True)

df.to_csv("data/raw/aesthetics_kb_clean.csv", index=False)
print(f"Clean KB: {len(df)} aesthetics ready for embedding")