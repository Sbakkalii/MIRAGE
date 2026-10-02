import ast
import glob
import os

from datasets import load_dataset
from itertools import islice
from joblib import Parallel, delayed
import unicodedata
import re
import pandas as pd
from dateparser.search import search_dates
from dateparser_data.settings import default_parsers
import numpy as np
from time import time
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch
import faiss
from sentence_transformers import SentenceTransformer
import json

torch.cuda.empty_cache()

if os.path.exists("checkpoint__df_chunk_ready.json"):
    print("[INFO] Chargement des données déjà prétraitées...")
    df_chunk = pd.read_parquet("df_chunk_ready.parquet")

else :

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
    max_docs_per_lang = 100000  # taille du dataset (x2 car français + anglais)

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

    chunks = np.array_split(df_chunk, 5000)

    def post_treatment_bert_entities(entities: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """
        Post-process the entities detected by BERT to clean them up before sending them to spaCy.
        :param entities: the list of entities detected by BERT
        :return: the cleaned list of entities
        """
        cleaned = []
        current = ""
        label = None

        for word, tag in entities:
            if word.startswith("##"):
                current += word[2:]
            else:
                if current:
                    cleaned.append((current.strip(), label))
                current = word
                label = tag

        if current:
            cleaned.append((current.strip(), label))

        cleaned = [(w, t) for w, t in cleaned if len(w) >= 3 and (len(w) > 3 or t != "MISC")]
        return cleaned


    def enrich_df_with_ner_pipe(df_chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich the DataFrame with Named Entity Recognition (NER) using BERT and spaCy, storing results into meta["entities"].
        :param df_chunk: the DataFrame to enrich
        :return: the enriched DataFrame
        """
        tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
        model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
        bert_ner = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            batch_size=128
        )

        if isinstance(df_chunk.iloc[0]["meta"], str):
            df_chunk["meta"] = df_chunk["meta"].apply(lambda x: ast.literal_eval(x))

        for index, row in df_chunk.iterrows():
            text = row["text"]

            ents = [(ent['word'], ent['entity_group']) for ent in bert_ner(text)]
            ents = post_treatment_bert_entities(ents)

            final_ents = []
            for ent_text, ent_label in ents:
                if ent_label is None or ent_label == "None" or ent_label == "DATE":
                    continue

                final_ents.append((ent_text, ent_label))

            meta = df_chunk.at[index, "meta"]
            meta["entities"] = final_ents
            df_chunk.at[index, "meta"] = meta

        return df_chunk


    print("[INFO] Enrichissement des données avec NER...")
    df_chunk = enrich_df_with_ner_pipe(df_chunk)
    print("[INFO] Enrichissement terminé.")

    df_chunk.to_parquet("df_chunk_ready.parquet")
    with open("checkpoint__df_chunk_ready.json", "w") as f:
        f.write("done")

models = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "e5-small-v2": "intfloat/e5-small-v2",
    "multilingual-e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
    "linq-embed-mistral": "Linq-AI-Research/Linq-Embed-Mistral",
    "SFR-Embedding-Mistral": "Salesforce/SFR-Embedding-Mistral"
}

available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def get_last_completed_batch(model_name):
    batch_dir = f"embedding_benchmark_results/{model_name}"
    if not os.path.exists(batch_dir):
        return -1

    existing_files = glob.glob(f"{batch_dir}/similarities_step_*.npy")
    if not existing_files:
        return -1

    indices = [int(os.path.basename(f).split("_")[-1].replace(".npy", "")) for f in existing_files]
    return max(indices)


def save_partial_results(model_name, step_idx, D_batch, I_i):
    save_dir = f"embedding_benchmark_results/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Sauvegarde des similarités du batch
    np.save(f"{save_dir}/similarities_step_{step_idx}.npy", D_batch)
    np.save(f"{save_dir}/indices_step_{step_idx}.npy", I_i)


def save_final_results(model_name, doc_embeddings, index, results_dict):
    save_dir = f"embedding_benchmark_results/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Embeddings
    np.save(f"{save_dir}/embeddings.npy", doc_embeddings)

    # Index FAISS
    faiss.write_index(index, f"{save_dir}/faiss_index.idx")

    # Résumé résultats
    df_result = pd.DataFrame([results_dict])
    df_result.to_csv(f"{save_dir}/summary.csv", index=False)

    with open(f"{save_dir}/summary.json", "w") as f:
        clean_dict = {k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) for k, v in results_dict.items()}
        json.dump(clean_dict, f, indent=4)


def encode_on_gpu(texts, model_path, device, batch_size=128):
    torch.cuda.set_device(device)
    model = SentenceTransformer(model_path, device=device)
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)


def evaluate_model_massive_multigpu(model_name, model_path, corpus, meta, k=5):
    print(f"\n[INFO] Évaluation du modèle {model_name} sur {len(corpus)} textes...")
    start_time = time()

    last_batch_completed = get_last_completed_batch(model_name)
    print(f"[INFO] Reprise à partir du batch {last_batch_completed + 1}...")

    num_devices = len(available_devices)
    chunks = np.array_split(corpus, num_devices)
    devices = available_devices[:num_devices]

    encoded_chunks = Parallel(n_jobs=num_devices)(
        delayed(encode_on_gpu)(chunk, model_path, device) for chunk, device in zip(chunks, devices)
    )

    doc_embeddings = np.vstack(encoded_chunks)
    time_docs = time() - start_time

    dim = doc_embeddings.shape[1]
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)

    # Recherche par similarité en batch
    batch_size = 5000

    def faiss_search_batch(start_idx):
        batch = doc_embeddings[start_idx:start_idx + batch_size]
        return index.search(batch, k)

    save_every = 1
    start_batch = last_batch_completed + 1

    def safe_batch_faiss(i, batch_num):
        D_i, I_i = faiss_search_batch(i)
        save_partial_results(model_name, batch_num, D_i, I_i)
        print(f"[Checkpoint] Batch {batch_num} sauvegardé")
        return D_i

    results = Parallel(n_jobs=min(32, num_devices * 4))(
        delayed(safe_batch_faiss)(i, batch_num)
        for batch_num, i in
        enumerate(range(start_batch * batch_size, len(doc_embeddings), batch_size), start=start_batch)
    )

    if not results:
        print("[INFO] Tous les batchs ont déjà été traités. Skip final stats.")
        return {}

    D = np.vstack(results)

    similarities_top1 = D[:, 0]
    similarities_top5 = np.mean(D[:, :5], axis=1)
    diff_top1_top2 = D[:, 0] - D[:, 1]

    corpus_langs = [m['lang'] for m in meta]

    results_dict = {
        "Model": model_name,
        "Mean Top-1 Similarity": round(np.mean(similarities_top1), 4),
        "Mean Top-5 Similarity": round(np.mean(similarities_top5), 4),
        "Mean Drop Top-1 to Top-2": round(np.mean(diff_top1_top2), 4),
        "Doc Encoding Time (s)": round(time_docs, 2),
        "Embedding Dimension": dim,
        "Corpus Languages": list(set(corpus_langs))
    }

    # Sauvegarde des résultats dans un fichier CSV
    save_final_results(model_name, doc_embeddings, index, results_dict)

    return results_dict


final_results = []

for model_name, model_path in models.items():
    print(f"[INFO] Evaluation du modèle {model_name}...")
    torch.cuda.empty_cache()
    results = evaluate_model_massive_multigpu(model_name, model_path, df_chunk["text"].tolist(),
                                              df_chunk["meta"].tolist())
    final_results.append(results)
    print(f"[INFO] Évaluation du modèle {model_name} terminée.")

print("[INFO] Évaluation de tous les modèles terminée.")