from typing import List

from pymilvus import MilvusClient
import torch
import os
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from time import time
from joblib import Parallel, delayed
import glob

client = MilvusClient("../../rag_v1_milvus.db")
collection_name = "rag_v1"

ebd_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

def is_model_compatible(model_path: str) -> bool:
    try:
        _ = SentenceTransformer(model_path, device="cuda:0", trust_remote_code=True)
        return True
    except Exception as e:
        print(f"[SKIPPED] {model_path} → {e}")
        return False


full_reranker_models = {
    # "NV-Embed-v2": "nvidia/NV-Embed-v2",
    "Linq-Embed-Mistral": "Linq-AI-Research/Linq-Embed-Mistral",
    "SFR-Embedding-Mistral": "Salesforce/SFR-Embedding-Mistral",
    "multilingual-e5-large-instruct": "intfloat/multilingual-e5-large-instruct",
    "e5-mistral-7b-instruct": "intfloat/e5-mistral-7b-instruct",
}

reranker_models = {
    name: path for name, path in full_reranker_models.items() if is_model_compatible(path)
}

def benchmark_rerankers_dual_retrieval_multigpu_safe(
    query: str,
    variations: List[str],
    top_k: int = 10
) -> pd.DataFrame:
    benchmarks = []
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    query_vec = ebd_model.encode(query, normalize_embeddings=True).tolist()
    dense_results = client.search(
        collection_name="rag_v1",
        data=[query_vec],
        anns_field="vector",
        search_params={"metric_type": "COSINE"},
        limit=top_k,
        output_fields=["text", "vector"]
    )[0]
    dense_texts = [hit["entity"]["text"] for hit in dense_results]

    all_queries = [query] + variations
    all_vectors = ebd_model.encode(all_queries, normalize_embeddings=True)
    fusion_scores = {}
    fusion_texts = {}
    k_rrf = 60
    fusion_limit = top_k * 3

    for vec in all_vectors:
        results = client.search(
            collection_name="rag_v1",
            data=[vec.tolist()],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            limit=fusion_limit,
            output_fields=["text", "vector"]
        )[0]

        for rank, hit in enumerate(results):
            doc_id = hit["id"]
            text = hit["entity"]["text"]
            fusion_scores[doc_id] = fusion_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
            fusion_texts[doc_id] = text

    sorted_fusion = sorted(fusion_scores.items(), key=lambda x: -x[1])[:top_k]
    fusion_texts_topk = [fusion_texts[doc_id] for doc_id, _ in sorted_fusion]

    for label, model_path in reranker_models.items():
        success = False
        for device in devices:
            print(f"[INFO] Trying model {label} on {device}...")
            try:
                torch.cuda.set_device(device)
                torch.cuda.empty_cache()

                start = time()
                reranker = SentenceTransformer(model_path, device=device)

                q_vec = reranker.encode(query, normalize_embeddings=True)

                # Rerank dense
                d_vecs_dense = reranker.encode(dense_texts, normalize_embeddings=True)
                scores_dense = [1 - cosine(q_vec, doc_vec) for doc_vec in d_vecs_dense]
                i_dense = np.argsort(scores_dense)[::-1]
                dense_cos1 = scores_dense[i_dense[0]]
                dense_cos5 = np.mean([scores_dense[i] for i in i_dense[:5]])
                dense_drop = scores_dense[i_dense[0]] - scores_dense[i_dense[1]]

                # Rerank fusion
                d_vecs_fusion = reranker.encode(fusion_texts_topk, normalize_embeddings=True)
                scores_fusion = [1 - cosine(q_vec, doc_vec) for doc_vec in d_vecs_fusion]
                i_fusion = np.argsort(scores_fusion)[::-1]
                fusion_cos1 = scores_fusion[i_fusion[0]]
                fusion_cos5 = np.mean([scores_fusion[i] for i in i_fusion[:5]])
                fusion_drop = scores_fusion[i_fusion[0]] - scores_fusion[i_fusion[1]]

                duration = round(time() - start, 4)

                benchmarks.append({
                    "model": label,
                    "device": device,
                    "cosine@1_dense": round(dense_cos1, 4),
                    "mean@5_dense": round(dense_cos5, 4),
                    "drop_1to2_dense": round(dense_drop, 4),
                    "cosine@1_fusion": round(fusion_cos1, 4),
                    "mean@5_fusion": round(fusion_cos5, 4),
                    "drop_1to2_fusion": round(fusion_drop, 4),
                    "time(s)": duration
                })

                del reranker
                torch.cuda.empty_cache()
                success = True
                break

            except torch.cuda.OutOfMemoryError:
                print(f"[WARNING] OOM sur {device} pour {label}. On passe au suivant...")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"[ERROR] Échec modèle {label} sur {device} → {e}")
                torch.cuda.empty_cache()
                continue

        if not success:
            print(f"[FAIL] Aucun GPU n'a pu traiter le modèle {label}. Modèle ignoré.")

    return pd.DataFrame(benchmarks)


variations = [
    "Pourquoi l’Empire romain s’est-il effondré ?",
    "Quelles sont les causes de la disparition de l’Empire romain ?",
    "Quels événements ont conduit à la fin de l’Empire romain ?"
]

torch.cuda.empty_cache()

df = benchmark_rerankers_dual_retrieval_multigpu_safe(
    query="Quels ont été les facteurs de la chute de l’Empire romain ?",
    variations=variations,
    top_k=10
)
df.to_csv("./benchmark.csv", index=False)