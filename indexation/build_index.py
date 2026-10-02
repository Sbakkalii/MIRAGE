"""Rebuild the rag_v1 Milvus index from the MIRACL fr/en subset.

Recipe follows the cluster pipeline (preprocessing/pretraitment.py,
preprocessing/text_cleaning.py, preprocessing/chunking.py, preprocessing/NER.py,
embedding/vectorisation.py) that produced the paper's 688,992-chunk corpus:
  1. stream the first MAX_DOCS documents per language from miracl/miracl-corpus
  2. clean text with preprocessing.text_cleaning.clean_text (NFKC normalization,
     quote normalization, newlines -> sentence boundaries, punctuation spacing,
     HTML tag/entity removal, unprintable stripping)
  3. sentence-boundary segmentation (split on ". "), exact-duplicate removal
  4. NER enrichment with Babelscape/wikineural-multilingual-ner (paper Sec. 4.2)
  5. embed raw chunk text with intfloat/multilingual-e5-large-instruct
     (embedding/vectorisation.py encodes df["text"])
  6. insert into a local Milvus collection (COSINE, HNSW M=48 efC=200)
     with fields: id, vector, text, docid, title, lang, dates, entities
     (dates left empty: no evaluated path reads them, and the paper's
     Limitations notes MIRACL passages carry no reliable date metadata)

Intermediate artifacts are cached under data/ (chunks.parquet, embeddings.npy).
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time
from itertools import islice

import numpy as np
import pandas as pd
from datasets import load_dataset
from joblib import Parallel, delayed
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

sys.path.insert(0, REPO_ROOT)
from preprocessing.text_cleaning import clean_text  # noqa: E402

MAX_DOCS_PER_LANG = 100000
LANGUAGES = ["fr", "en"]
COLLECTION = "rag_v1"
ENCODER = "intfloat/multilingual-e5-large-instruct"


def load_lang(lang: str, max_docs: int) -> list[dict]:
    print(f"[{lang.upper()}] Chargement...", flush=True)
    dataset_stream = load_dataset(
        "miracl/miracl-corpus", lang, split="train", streaming=True,
        trust_remote_code=True,
    )
    docs = []
    for doc in islice(dataset_stream, max_docs):
        docs.append(
            {"docid": doc["docid"], "lang": lang, "title": doc["title"], "text": doc["text"]}
        )
    return docs


def chunk_df(df: pd.DataFrame) -> pd.DataFrame:
    chunk_text = []
    chunk_meta = []
    for _, row in df.iterrows():
        sentences = row["text"].split(". ")
        for sentence in sentences:
            chunk_text.append(sentence)
            chunk_meta.append(
                {"title": row["title"], "docid": row["docid"], "lang": row["lang"]}
            )
    df_chunk = pd.DataFrame({"text": chunk_text, "meta": chunk_meta})
    df_chunk = df_chunk.drop_duplicates(subset=["text"])
    df_chunk = df_chunk.dropna(subset=["text"])
    return df_chunk.reset_index(drop=True)


def enrich_entities(df_chunk: pd.DataFrame, batch_size: int = 128) -> pd.DataFrame:
    """NER enrichment with Babelscape/wikineural-multilingual-ner (preprocessing/NER.py)."""
    from preprocessing.NER import NERProcessor

    processor = NERProcessor()
    try:
        texts = df_chunk["text"].fillna("").tolist()
        all_entities = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            all_entities.extend(processor.bert_ner(batch))
            if start % (batch_size * 100) == 0:
                print(f"[ner] {start}/{len(texts)}", flush=True)
        entities = [processor._post_process_entities(e) for e in all_entities]
        df_chunk = df_chunk.copy()
        df_chunk["meta"] = [
            {**m, "entities": e}
            for m, e in zip(df_chunk["meta"].tolist(), entities)
        ]
        return df_chunk
    finally:
        processor.release()


def embed_chunks(df_chunk: pd.DataFrame, batch_size: int = 64) -> np.ndarray:
    encoder = SentenceTransformer(ENCODER, trust_remote_code=True)
    texts = df_chunk["text"].tolist()
    embeddings = encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    return np.asarray(embeddings, dtype=np.float32)


def insert_into_milvus(df_chunk, embeddings, db_path, collection, batch_size=1000):
    client = MilvusClient(db_path)
    if client.has_collection(collection):
        client.drop_collection(collection)
    client.create_collection(
        collection_name=collection,
        dimension=int(embeddings.shape[1]),
        metric_type="COSINE",
        index_type="HNSW",
        params={"M": 48, "efConstruction": 200},
    )
    n = len(df_chunk)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = []
        for i in range(start, end):
            meta = df_chunk["meta"].iloc[i]
            batch.append(
                {
                    "id": i,
                    "vector": embeddings[i].tolist(),
                    "text": df_chunk["text"].iloc[i],
                    "docid": meta.get("docid", ""),
                    "title": meta.get("title", ""),
                    "lang": meta.get("lang", ""),
                    "dates": [],
                    "entities": meta.get("entities", []),
                }
            )
        client.insert(collection_name=collection, data=batch)
    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-docs", type=int, default=MAX_DOCS_PER_LANG)
    parser.add_argument("--skip-ner", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--no-insert", action="store_true")
    parser.add_argument("--db", default=os.path.join(REPO_ROOT, f"{COLLECTION}_milvus.db"))
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--limit-smoke", type=int, default=0)
    args = parser.parse_args()

    chunks_path = os.path.join(args.data_dir, "chunks.parquet")
    embeddings_path = os.path.join(args.data_dir, "embeddings.npy")
    os.makedirs(args.data_dir, exist_ok=True)
    t0 = time.time()

    use_cache_read = not args.recompute
    use_cache_write = not args.recompute and args.limit_smoke == 0

    df_chunk = None
    embeddings = None
    if use_cache_read and os.path.exists(chunks_path):
        print(f"[cache] loading {chunks_path}", flush=True)
        df_chunk = pd.read_parquet(chunks_path)
        df_chunk["meta"] = df_chunk["meta"].apply(json.loads)
        print(f"[cache] {len(df_chunk)} chunks", flush=True)
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            print(f"[cache] embeddings {embeddings.shape}", flush=True)

    if df_chunk is None:
        results = Parallel(n_jobs=len(LANGUAGES))(
            delayed(load_lang)(lang, args.max_docs) for lang in LANGUAGES
        )
        df = pd.DataFrame([d for lang_docs in results for d in lang_docs])
        print(f"[load] {len(df)} documents in {time.time() - t0:.1f}s", flush=True)

        df["title"] = df["title"].apply(clean_text)
        df["text"] = df["text"].apply(clean_text)

        t1 = time.time()
        df_chunk = chunk_df(df)
        print(f"[chunk] {len(df_chunk)} chunks in {time.time() - t1:.1f}s", flush=True)

        if not args.skip_ner:
            t1 = time.time()
            df_chunk = enrich_entities(df_chunk)
            print(f"[ner] {len(df_chunk)} chunks in {time.time() - t1:.1f}s", flush=True)
        else:
            df_chunk["meta"] = df_chunk["meta"].apply(
                lambda m: {**m, "entities": []}
            )

        if use_cache_write:
            pd.DataFrame(
                {"text": df_chunk["text"], "meta": df_chunk["meta"].apply(json.dumps)}
            ).to_parquet(chunks_path, index=False)
            print(f"[cache] saved {chunks_path}", flush=True)

    if embeddings is None:
        t1 = time.time()
        embeddings = embed_chunks(df_chunk)
        print(f"[embed] {embeddings.shape} in {time.time() - t1:.1f}s", flush=True)
        if use_cache_write:
            np.save(embeddings_path, embeddings)
            print(f"[cache] saved {embeddings_path}", flush=True)

    if args.limit_smoke > 0:
        df_chunk = df_chunk.iloc[: args.limit_smoke].reset_index(drop=True)
        embeddings = embeddings[: args.limit_smoke]

    if args.no_insert:
        print("[done] skipping insert")
        return

    t1 = time.time()
    client = insert_into_milvus(df_chunk, embeddings, args.db, args.collection)
    print(
        f"[insert] {len(df_chunk)} rows into {args.collection} ({args.db}) "
        f"in {time.time() - t1:.1f}s",
        flush=True,
    )

    stats = client.get_collection_stats(args.collection)
    print(f"[stats] {stats}", flush=True)

    probe = SentenceTransformer(ENCODER, trust_remote_code=True)
    qv = probe.encode("Qui est Antoine Meillet ?", normalize_embeddings=True)
    hits = client.search(
        collection_name=args.collection,
        data=[qv.tolist()],
        anns_field="vector",
        limit=3,
        output_fields=["text", "title", "docid"],
        params={"metric_type": "COSINE"},
    )[0]
    for h in hits:
        print(f"[probe] {h['distance']:.4f} | {h['entity']['title']} | {h['entity']['text'][:90]}")
    print(f"[done] total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
