"""Export everything the Colab T4 notebook needs into one JSON blob.

Produces (next to this script):
  colab_data.json          raw payload
  colab_data.b64           gzip+base64 of the payload (embedded into the .ipynb)

Payload contents
  queries    : 50 rows from benchmarks/evaluation/queries_50.csv
  expansions : cached r2_expansions.json (5 reformulations per query)
  r2         : per qid {dense: [10 texts], fusion: [10 texts]} — exact
               retrieval_benchmark.py semantics (RRF k=60, limit K*3,
               e5-large-instruct encoder, local fp32 encode)
  r3         : per qid {question, contexts: [{text, meta:{docid,title}}]} —
               multi_query_fusion top-10 (set+sorted query list) reranked by
               multilingual-e5-large via the deployed rerank_documents logic

Run from repo root:  .venv/bin/python notebooks/export_colab_data.py
"""

import base64
import csv
import gzip
import json
import os
import sys
from typing import List, Dict

import numpy as np
from pymilvus import MilvusClient
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

QUERIES_PATH = os.path.join(REPO, "benchmarks", "evaluation", "queries_50.csv")
EXPANSIONS_PATH = os.path.join(REPO, "benchmarks", "retrieval", "r2_expansions.json")
DB_PATH = os.path.join(REPO, "rag_v1_milvus.db")
COLLECTION = "rag_v1"
ENCODER = "intfloat/multilingual-e5-large-instruct"
TOP_K = 10
K_RRF = 60
OUT_JSON = os.path.join(HERE, "colab_data.json")
OUT_B64 = os.path.join(HERE, "colab_data.b64")


def milvus_search(client, vec, limit):
    res = client.search(
        collection_name=COLLECTION,
        data=[vec.tolist()],
        anns_field="vector",
        search_params={"metric_type": "COSINE"},
        limit=limit,
        output_fields=["text", "title", "docid"],
    )[0]
    return res


def dense_topk(client, encoder, query):
    vec = encoder.encode(query, normalize_embeddings=True)
    return [h["entity"]["text"] for h in milvus_search(client, vec, TOP_K)]


def fusion_r2(client, encoder, query, variations):
    """retrieval_benchmark.py order: [query] + variations, no dedup/sort."""
    all_queries = [query] + list(variations)
    vectors = encoder.encode(all_queries, normalize_embeddings=True)
    scores, texts = {}, {}
    for vec in vectors:
        for rank, hit in enumerate(milvus_search(client, vec, TOP_K * 3)):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (K_RRF + rank + 1)
            texts[doc_id] = hit["entity"]["text"]
    ordered = sorted(scores.items(), key=lambda x: -x[1])[:TOP_K]
    return [texts[d] for d, _ in ordered]


def fusion_r3_docs(client, encoder, query, variations):
    """multi_query_fusion semantics: sorted(set([query]+variations))."""
    all_queries = sorted(set([query] + list(variations)))
    vectors = encoder.encode(all_queries, normalize_embeddings=True)
    scores, docs = {}, {}
    for vec in vectors:
        for rank, hit in enumerate(milvus_search(client, vec, TOP_K * 3)):
            doc_id = str(hit["id"])
            if doc_id not in docs:
                ent = hit.get("entity", {})
                docs[doc_id] = {
                    "id": doc_id,
                    "text": ent.get("text", "Texte non trouvé"),
                    "title": ent.get("title") or "Titre inconnu",
                    "meta": {
                        "docid": ent.get("docid") or "Source_inconnue_docid",
                        "title": ent.get("title") or "Titre inconnu",
                    },
                }
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (K_RRF + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: -x[1])[:TOP_K]
    return [docs[d] for d, _ in ordered if d in docs]


def rerank_documents(
        query: str,
        docs: List[Dict],
        model: SentenceTransformer,
        top_k: int = 10
) -> List[Dict]:
    """
    Rerank documents based on their similarity to the query using a SentenceTransformer model.
    :param query: The query string to compare against the documents.
    :param docs: A list of documents, where each document is a dictionary containing at least a "text" key.
    :param model: The SentenceTransformer model to use for encoding the query and documents.
    :param top_k: The number of top documents to return after reranking.
    :return:
    """
    query_vec = model.encode(query, normalize_embeddings=True)
    doc_texts = [doc["text"] for doc in docs]
    doc_vecs = model.encode(doc_texts, normalize_embeddings=True)

    scores = [1 - cosine(query_vec, doc_vec) for doc_vec in doc_vecs]
    sorted_indices = sorted(range(len(scores)), key=lambda i: -scores[i])

    reranked = []
    for idx in sorted_indices[:top_k]:
        doc = docs[idx].copy()
        doc["rerank_score"] = round(scores[idx], 4)
        reranked.append(doc)

    return reranked


def main():
    with open(QUERIES_PATH, newline="") as f:
        queries = list(csv.DictReader(f))
    with open(EXPANSIONS_PATH) as f:
        expansions = json.load(f)
    assert len(queries) == 50 and len(expansions) == 50

    client = MilvusClient(DB_PATH)
    client.load_collection(COLLECTION)
    encoder = SentenceTransformer(ENCODER, trust_remote_code=True)
    print(f"[export] {len(queries)} queries", flush=True)

    r2, r3 = {}, []
    for i, q in enumerate(queries):
        qid, qt = q["id"], q["query"]
        var = expansions[qid]
        r2[qid] = {
            "dense": dense_topk(client, encoder, qt),
            "fusion": fusion_r2(client, encoder, qt, var),
        }
        fusion_docs = fusion_r3_docs(client, encoder, qt, var)
        reranked = rerank_documents(qt, fusion_docs, encoder, top_k=TOP_K)
        r3.append({
            "id": qid,
            "category": q["category"],
            "lang": q["lang"],
            "question": qt,
            "contexts": [
                {"text": d["text"], "meta": d["meta"]} for d in reranked
            ],
        })
        if (i + 1) % 5 == 0:
            print(f"[export] {i + 1}/50", flush=True)

    payload = {
        "meta": {
            "n_queries": 50,
            "encoder": ENCODER,
            "top_k": TOP_K,
            "k_rrf": K_RRF,
            "source_db": "rag_v1_milvus.db (688992 rows, cluster recipe)",
            "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "queries": queries,
        "expansions": expansions,
        "r2": r2,
        "r3": r3,
    }
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with open(OUT_JSON, "wb") as f:
        f.write(raw)
    b64 = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
    with open(OUT_B64, "w") as f:
        f.write(b64)
    print(f"[export] raw={len(raw)/1e6:.2f}MB b64={len(b64)/1e6:.2f}MB -> {OUT_JSON}, {OUT_B64}")


if __name__ == "__main__":
    main()
