from joblib import Parallel, delayed
import pandas as pd

from preprocessing import load_lang, clean_text
from preprocessing.chunking import chunking
from preprocessing.date_extraction import extract_date

languages = ['fr', 'en']
max_docs_per_lang = 100  # taille du dataset (x2 car français + anglais)

results = Parallel(n_jobs=len(languages))(
    delayed(load_lang)(lang, max_docs_per_lang) for lang in languages
)

all_docs = [doc for lang_docs in results for doc in lang_docs]

df = pd.DataFrame(all_docs)
df["title"] = df["title"].apply(clean_text)
df["text"] = df["text"].apply(clean_text)

df_chunk = chunking(df, 0)

df_chunk = extract_date(df_chunk)

# save the dataframe to a CSV file
df_chunk.to_csv("df_chunk.csv", index=False)