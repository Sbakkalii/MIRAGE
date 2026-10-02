from typing import List, Dict

from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


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
