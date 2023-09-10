import pandas as pd

INPUT_FILE = "./data/KB articles.xlsx"
OUT_DIR = "./data/help-knowledgebase_2023-03-13/"

df = pd.read_excel(INPUT_FILE, sheet_name="Sheet1")

for index, row in df.iterrows():
    title, slug, topic = row["Title"], row["Slug"], row["Topic"]
    text = row["Text"].replace("\\n", "\n").replace('\\"', '"')
    with open(OUT_DIR + slug + ".md", "w") as f:
        f.write(f"---\ntitle: {title}\ntopic: {topic}\n---\n{text}")
