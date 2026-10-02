from datasets import load_dataset
from itertools import islice
from joblib import Parallel, delayed
import unicodedata
import re
import pandas as pd
from dateparser.search import search_dates
from dateparser_data.settings import default_parsers
import numpy as np
import spacy
from time import time
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import torch
from functools import lru_cache

def load_lang(lang: str, max_docs: int) -> list:
    """
    Load the dataset for a specific language and return a list of documents.
    :param lang: language code (e.g., 'fr' or 'en')
    :param max_docs: maximum number of documents to a load
    :return: list of documents
    """
    print(f"[{lang.upper()}] Chargement...")
    dataset_stream = load_dataset('miracl/miracl-corpus', lang, split='train', streaming=True, trust_remote_code=True)
    sample = islice(dataset_stream, max_docs) if max_docs > 0 else dataset_stream

    docs = []
    for doc in sample:
        docs.append({
            "docid": doc["docid"],
            "lang": lang,
            "title": doc["title"],
            "text": doc["text"]
        })
    return docs

languages = ['fr', 'en']
max_docs_per_lang = 50000  # taille du dataset (x2 car français + anglais)

results = Parallel(n_jobs=len(languages))(
    delayed(load_lang)(lang, max_docs_per_lang) for lang in languages
)

print(f"[INFO] {len(results)} langues chargées avec succès.")

all_docs = [doc for lang_docs in results for doc in lang_docs]

def clean_text(text: str) -> str:
    """
    Clean the input text by performing various preprocessing steps.
    :param text: the text to clean
    :return: cleaned text
    """

    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Replacing typographical quotation marks with single quotation marks
    text = text.replace('“', '"').replace('”', '"').replace('«', '"').replace('»', '"').replace("’", "'")

    # Replace \n with a space (or a period + space if it breaks a sentence)
    text = text.replace('\n', '. ')

    # Delete invisible control characters (except those already managed)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # Delete multiple spaces
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)

    # Delete multiple spaces around punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)

    # Delete spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)

    # Delete HTML tags and HTML entities
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&\w+;', '', text)

    # Delete unprintable characters or orphan symbols
    text = re.sub(r'[^\x20-\x7EÀ-ÿ€£$¥•–—’“”…°²³µ·]', '', text)

    return text.strip()

print("[INFO] Nettoyage des textes...")

df = pd.DataFrame(all_docs)
df["title"] = df["title"].apply(clean_text)
df["text"] = df["text"].apply(clean_text)

print("[INFO] Nettoyage terminé.")

def chunking(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    """
    Chunking the dataframe into smaller chunks of a specified size.
    :param df: the dataframe to chunk
    :param chunk_size: the size of each chunk
    :return: a new dataframe with the chunks
    """

    # TODO : chunking by chunk_size

    chunk_text = []
    chunk_meta = []

    for index, row in df.iterrows():
        text = row["text"]
        title = row["title"]
        docid = row["docid"]
        lang = row["lang"]
        sentences = text.split(". ")

        for sentence in sentences:
            chunk_text.append(sentence)
            chunk_meta.append({
                "title": title,
                "docid": docid,
                "lang": lang
            })

    if len(chunk_text) != len(chunk_meta):
        raise ValueError("Chunk text and metadata lengths do not match.")

    df_chunk = pd.DataFrame({"text": chunk_text, "meta": chunk_meta})
    df_chunk = df_chunk.drop_duplicates(subset=["text"])
    df_chunk = df_chunk.dropna(subset=["text"])

    return df_chunk

print("[INFO] Découpage des textes en phrases...")

df_chunk = chunking(df, 0)

print("[INFO] Découpage terminé.")

def contains_explicit_1_january(text: str) -> bool:
    """
    Check if the text contains an explicit mention of 1st January.
    :param text: the text to check
    :return: True if the text contains an explicit mention of 1st January, False otherwise
    """
    patterns = [
        r"\b0?1[\/\-\. ]?0?1\b",  # 01/01, 1/1, 01-01, etc.
        r"\b(1er|1|01)[^\d]?(janvier|january)\b",  # 1 janvier, 1er janvier, 01 janvier
        r"\b(janvier|january)[^\d]*(1er|1|01)\b",  # janvier 1, january 1st
        r"\b(1st|first) of (january|janvier)\b"  # 1st of January
    ]
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def has_explicit_date_pattern(text: str) -> bool:
    """
    Check if the text contains an explicit date pattern. In case of ambiguous date.
    :param text: the text to check
    :return: True if the text contains an explicit date pattern, False otherwise
    """
    numeric_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}\b"
    ]

    textual_patterns = [
        r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{2,4}\b"
    ]

    all_patterns = numeric_patterns + textual_patterns

    for pattern in all_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def split_on_conjunctions(text: str) -> list:
    """
    Split the text on conjunctions (et, and, , or ;).
    :param text: the text to split
    :return: the list of segments
    """

    text = re.sub(r"\([^)]*\)", "", text)

    return re.split(r"\s+(et|and|,|;)\s+", text)


def is_pure_date_expression(text: str) -> bool:
    """
    Check if the text is a pure date expression.
    :param text: the text to check
    :return: True if the text is a pure date expression, False otherwise
    """
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b",  # 15/04[/2025]
        r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b"
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def extract_date(df: pd.DataFrame) -> pd.DataFrame:
    parsers = [parser for parser in default_parsers if parser != 'relative-time']

    for i, row in df.iterrows():
        lang = row["meta"]["lang"]
        res_list = []

        try:
            # On divise le texte sur les conjonctions
            segments = [seg.strip() for seg in split_on_conjunctions(row["text"]) if
                        seg.strip().lower() not in ["et", "and", ",", ";"] and seg.strip() != ""]

            for segment in segments:
                dates_found = search_dates(
                    segment,
                    languages=["fr", "en"],
                    settings={
                        'PREFER_DAY_OF_MONTH': 'first',
                        'PREFER_MONTH_OF_YEAR': 'first',
                        'DATE_ORDER': 'DMY' if lang == 'fr' else 'MDY',
                        'PARSERS': parsers,
                    }
                )

                if dates_found:
                    for matched_text, result in dates_found:
                        res = {}

                        if result.day == 1 and result.month == 1:
                            if contains_explicit_1_january(matched_text):
                                res["year"] = result.year if has_explicit_date_pattern(matched_text) else None
                                res["month"] = result.month
                                res["day"] = result.day
                            else:
                                res["year"] = result.year if has_explicit_date_pattern(matched_text) else None
                                res["month"] = None
                                res["day"] = None
                        else:
                            res["year"] = result.year if has_explicit_date_pattern(matched_text) else None
                            res["month"] = result.month
                            res["day"] = result.day

                        res["hour"] = result.hour if result.hour else None
                        res["minute"] = result.minute if result.minute else None
                        res["second"] = result.second if result.second else None

                        # False positive check
                        if res["day"] and res["month"] and not is_pure_date_expression(matched_text):
                            break

                        if res["day"] or res["month"] or res["year"]:
                            res_list.append(res)

        except Exception as e:
            res_list = [{"error": str(e)}]

        df.at[i, "meta"]["dates"] = res_list

    return df

print("[INFO] Extraction des dates...")

df_chunk = extract_date(df_chunk)

print("[INFO] Extraction des dates terminée.")

chunks = np.array_split(df_chunk, 2)

models = {
    "spacy-xx_ent_wiki_sm": ["multilingual", "xx_ent_wiki_sm", "spacy-first"],
    "wikineural-multilingual-ner": ["multilingual", "Babelscape/wikineural-multilingual-ner", "transformers"],
    "bert-base-multilingual-cased": ["multilingual", "google-bert/bert-base-multilingual-cased", "transformers"],
    "bert-base-historic-multilingual-cased": ["multilingual", "dbmdz/bert-base-historic-multilingual-cased", "transformers"]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] Device utilisé :", device)

@lru_cache(maxsize=None)
def load_spacy_model(model_name):
    return spacy.load(model_name)

transformers_pipelines = {
    name: pipeline(
        "ner",
        model=AutoModelForTokenClassification.from_pretrained(model_id).to(device).half(),
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        device=0,
        aggregation_strategy="simple",
        batch_size=128
    )
    for name, (_, model_id, fw) in models.items()
    if fw == "transformers"
}

def compute_tfidf(entities):
    tokens = [e[0] for e in entities]  # extraire les textes
    if not tokens or all(e.strip() == "" for e in tokens):
        return 0.0
    try:
        tfidf = TfidfVectorizer().fit_transform([" ".join(tokens)])
        return float(np.mean(tfidf.sum(axis=1)))
    except ValueError:
        return 0.0


def compute_seqscore_contextual(text, entities):
    tokens = [e[0] for e in entities]
    if not tokens:
        return {"avg_len": 0, "uniq_ratio": 0, "context_sim": 0}

    lengths = [len(e.split()) for e in tokens]
    uniq_ratio = len(set(tokens)) / len(tokens)

    try:
        context_vec = TfidfVectorizer().fit_transform([text])
        entity_vec = TfidfVectorizer().fit_transform(tokens)
        sim = float(np.mean(context_vec.dot(entity_vec.T).toarray()))
    except ValueError:
        sim = 0.0

    return {
        "avg_len": np.mean(lengths),
        "uniq_ratio": uniq_ratio,
        "context_sim": sim
    }


def compute_avg_entity_length(entities):
    return float(np.mean([len(e[0]) for e in entities if isinstance(e[0], str)])) if entities else 0.0


def compute_entity_stability(entities_base, entities_secondary):
    base_set = set(e[0] for e in entities_base)
    sec_set = set(e[0] for e in entities_secondary)
    return len(base_set & sec_set) / len(base_set | sec_set) if base_set | sec_set else 0.0

def run_spacy(model_name, text):
    nlp = load_spacy_model(model_name)
    if nlp is None or not isinstance(text, str):
        return []
    return [ent.text for ent in nlp(text).ents]

def run_transformers(model_name, text):
    pipe = transformers_pipelines[model_name]
    return [(ent["word"], ent["entity_group"]) for ent in pipe(text)]


def process_model(model_id, framework, text, model_name):
    start = time()
    if framework == "transformers":
        entities = run_transformers(model_name, text)
    else:
        entities = run_spacy(model_id, text)
    duration = time() - start

    tfidf = compute_tfidf(entities)
    seq = compute_seqscore_contextual(text, entities)

    return {
        "entities": entities,
        "duration": duration,
        "tfidf_score": tfidf,
        "seq_avg_len": seq["avg_len"],
        "seq_uniq_ratio": seq["uniq_ratio"],
        "seq_context_sim": seq["context_sim"],
        "avg_entity_length": compute_avg_entity_length(entities)
    }

def process_row(row):
    text = row["text"]
    lang = row["meta"]["lang"]
    docid = row["meta"]["docid"]
    row_results = []
    for name, (model_lang, model_id, framework) in models.items():
        if model_lang != lang and model_lang != "multilingual":
            continue
        result = process_model(model_id, framework, text, name)
        row_results.append({
            "docid": docid,
            "text": text,
            "text_lang": lang,
            "model": name,
            "duration": result["duration"],
            "entity_count": len(result["entities"]),
            "tfidf_score": result["tfidf_score"],
            "seq_avg_len": result["seq_avg_len"],
            "seq_uniq_ratio": result["seq_uniq_ratio"],
            "seq_context_sim": result["seq_context_sim"],
            "avg_entity_length": result["avg_entity_length"],
            "entity_stability": 1.0,
            "entities": result["entities"]
        })

    return row_results

max_rows = len(df_chunk) if device == "cuda" else 20
n_jobs = -1 if device == "cuda" else 1

def benchmark_ner_parallel(df, max_rows=None, n_jobs=1):
    df = df.head(max_rows)
    results_nested = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Parallel Benchmarking")
    )
    return pd.DataFrame([item for sublist in results_nested for item in sublist])

print("[INFO] Lancement du benchmark NER...")

results_df = benchmark_ner_parallel(df_chunk, max_rows=max_rows, n_jobs=n_jobs)
results_df.to_csv("./NER_benchmark_results.csv", index=False)

print("[INFO] Benchmark terminé.")