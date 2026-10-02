"""Stage-2 R2: 50-query dense-vs-fusion benchmark (Table 7 re-run).

Protocol mirrors benchmarks/retrieval/retrieval_benchmark.py:
  - dense condition (D): top-K search for the original query with
    intfloat/multilingual-e5-large-instruct
  - fusion condition (F): original query + N=5 Mistral-7B reformulations,
    each searched with limit K*3, RRF fusion (k=60), fused top-K
  - each listed model re-encodes the query and each candidate set; Sim@1,
    Sim@5 and the top-1/top-2 gap are averaged over the 50 queries in
    benchmarks/evaluation/queries_50.csv

Differences from the original script (recorded in the Stage-2 changelog):
  - 50 queries instead of a single query with hardcoded variations
  - reformulations generated with the deployed prompt
    (verbatim copy of ragas_eval.generate_query_variations: French template,
    temperature 0.7, N=5) and cached in r2_expansions.json
  - 7B models attempt fp32 first, fall back to 4-bit, then CPU; the precision
    actually used is recorded per model
  - Time is candidate re-encoding time; model load time is reported
    separately

Outputs (benchmarks/retrieval/): r2_results.csv, r2_summary.json,
r2_expansions.json.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from pymilvus import MilvusClient
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUERIES_PATH = os.path.join(REPO_ROOT, "benchmarks", "evaluation", "queries_50.csv")
DB_PATH = os.path.join(REPO_ROOT, "rag_v1_milvus.db")
COLLECTION = "rag_v1"
ENCODER = "intfloat/multilingual-e5-large-instruct"
LLM_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
RERANKER_MODELS = {
    "multilingual-e5-large": "intfloat/multilingual-e5-large-instruct",
    "e5-mistral-7b-instruct": "intfloat/e5-mistral-7b-instruct",
    "SFR-Embedding-Mistral": "Salesforce/SFR-Embedding-Mistral",
    "Linq-Embed-Mistral": "Linq-AI-Research/Linq-Embed-Mistral",
}


def generate_query_variations(original_query: str, llm_model, llm_tokenizer,
                              num_variations: int = 5, max_length: int = 64) -> list:
    """Verbatim copy of the deployed expansion prompt (ragas_eval.py)."""
    prompt_template = f"""
            Reformule la question suivante de {num_variations} manières différentes, 
            en conservant son sens original, dans le but de rechercher des informations dans des archives historiques.
            Chaque reformulation doit être concise et sur une nouvelle ligne. Ne numérote pas les reformulations.

            Question originale : {original_query}

            Reformulations :
        """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = llm_tokenizer(prompt_template, return_tensors="pt").to(device)
    generated_ids = llm_model.generate(
        **inputs,
        max_new_tokens=max_length * num_variations,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        pad_token_id=llm_tokenizer.pad_token_id,
    )
    response_text = llm_tokenizer.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    if "Reformulations :" in response_text:
        response_part = response_text.split("Reformulations :")[-1].strip()
        variations = [v.strip() for v in response_part.split("\n")
                      if v.strip() and v.lower() != original_query.lower()]
    else:
        variations = [v.strip() for v in response_text.split("\n")
                      if v.strip() and v.lower() != original_query.lower()]
    return variations[:num_variations]


def load_expansion_llm(precision: str):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    ) if precision == "4bit" else None
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    kwargs = {"device_map": "auto"}
    if quant is not None:
        kwargs["quantization_config"] = quant
    model = AutoModelForCausalLM.from_pretrained(LLM_NAME, torch_dtype=torch.float16, **kwargs)
    model.eval()
    return model, tokenizer


def load_reranker(path: str):
    """fp32 GPU -> fp32 CPU -> 4-bit; returns (model, precision_label, load_seconds).

    The 7B models do not fit in fp32 on the 8 GB GPU; running them in fp32 on
    CPU keeps numerics equivalent to the cluster fp32 run, so CPU is preferred
    over 4-bit quantization as the first fallback.
    """
    t0 = time.time()
    try:
        model = SentenceTransformer(path, device="cuda:0", trust_remote_code=True)
        return model, "fp32-cuda", round(time.time() - t0, 2)
    except Exception as e:
        print(f"[rerank] fp32-cuda load failed for {path}: {e}", flush=True)
        torch.cuda.empty_cache()
    try:
        model = SentenceTransformer(path, device="cpu", trust_remote_code=True)
        return model, "fp32-cpu", round(time.time() - t0, 2)
    except Exception as e:
        print(f"[rerank] fp32-cpu load failed for {path}: {e}", flush=True)
    try:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = SentenceTransformer(
            path, device="cuda:0", trust_remote_code=True,
            model_kwargs={"quantization_config": quant},
        )
        return model, "4bit-cuda", round(time.time() - t0, 2)
    except Exception as e:
        raise RuntimeError(f"no viable device for {path}: {e}")


def retrieve(client, encoder, query: str, variations: list, top_k: int,
             k_rrf: int) -> tuple:
    q_vec = encoder.encode(query, normalize_embeddings=True)
    dense_hits = client.search(
        collection_name=COLLECTION,
        data=[q_vec.tolist()],
        anns_field="vector",
        search_params={"metric_type": "COSINE"},
        limit=top_k,
        output_fields=["text", "vector"],
    )[0]
    dense_texts = [h["entity"]["text"] for h in dense_hits]

    all_queries = [query] + variations
    all_vectors = encoder.encode(all_queries, normalize_embeddings=True)
    fusion_scores = {}
    fusion_texts = {}
    fusion_limit = top_k * 3
    for vec in all_vectors:
        results = client.search(
            collection_name=COLLECTION,
            data=[vec.tolist()],
            anns_field="vector",
            search_params={"metric_type": "COSINE"},
            limit=fusion_limit,
            output_fields=["text", "vector"],
        )[0]
        for rank, hit in enumerate(results):
            doc_id = hit["id"]
            fusion_scores[doc_id] = fusion_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
            fusion_texts[doc_id] = hit["entity"]["text"]
    sorted_fusion = sorted(fusion_scores.items(), key=lambda x: -x[1])[:top_k]
    fusion_topk = [fusion_texts[doc_id] for doc_id, _ in sorted_fusion]
    return dense_texts, fusion_topk


def rescore(model, query: str, texts: list) -> dict:
    q_vec = model.encode(query, normalize_embeddings=True)
    doc_vecs = model.encode(texts, normalize_embeddings=True)
    scores = [1 - cosine(q_vec, dv) for dv in doc_vecs]
    order = np.argsort(scores)[::-1]
    return {
        "cos1": float(scores[order[0]]),
        "mean5": float(np.mean([scores[i] for i in order[:5]])),
        "drop": float(scores[order[0]] - scores[order[1]]),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-rrf", type=int, default=60)
    parser.add_argument("--num-variations", type=int, default=5)
    parser.add_argument("--expansion-precision", choices=["4bit", "fp16"], default="4bit")
    parser.add_argument("--regenerate-expansions", action="store_true")
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    with open(QUERIES_PATH) as f:
        queries = list(csv.DictReader(f))
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    print(f"[r2] {len(queries)} queries, top_k={args.top_k}, k_rrf={args.k_rrf}", flush=True)

    expansions_path = os.path.join(args.out_dir, "r2_expansions.json")
    expansions = {}
    if os.path.exists(expansions_path) and not args.regenerate_expansions:
        with open(expansions_path) as f:
            expansions = json.load(f)
        print(f"[r2] loaded {len(expansions)} cached expansion sets", flush=True)

    missing = [q for q in queries if q["id"] not in expansions]
    if missing:
        print(f"[r2] generating expansions for {len(missing)} queries "
              f"({args.expansion_precision})", flush=True)
        llm, tok = load_expansion_llm(args.expansion_precision)
        t0 = time.time()
        for q in missing:
            variations = generate_query_variations(
                q["query"], llm, tok, num_variations=args.num_variations
            )
            expansions[q["id"]] = variations
            print(f"  {q['id']}: {variations}", flush=True)
        gen_time = round(time.time() - t0, 1)
        with open(expansions_path, "w") as f:
            json.dump(expansions, f, ensure_ascii=False, indent=2)
        del llm
        torch.cuda.empty_cache()
        print(f"[r2] expansions generated in {gen_time}s -> {expansions_path}", flush=True)

    client = MilvusClient(DB_PATH)
    if not client.has_collection(COLLECTION):
        raise RuntimeError(f"collection {COLLECTION} missing in {DB_PATH}; run build_index.py")
    client.load_collection(COLLECTION)
    encoder = SentenceTransformer(ENCODER, trust_remote_code=True)

    retrieved = {}
    t0 = time.time()
    for q in queries:
        retrieved[q["id"]] = retrieve(
            client, encoder, q["query"], expansions[q["id"]], args.top_k, args.k_rrf
        )
    retrieve_time = round(time.time() - t0, 1)
    print(f"[r2] retrieval done in {retrieve_time}s", flush=True)

    rows = []
    summary = {
        "n_queries": len(queries),
        "top_k": args.top_k,
        "k_rrf": args.k_rrf,
        "num_variations": args.num_variations,
        "expansion_precision": args.expansion_precision,
        "retrieval_time_s": retrieve_time,
        "encoder": ENCODER,
        "models": {},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    for label, path in RERANKER_MODELS.items():
        print(f"[r2] reranker {label}", flush=True)
        model, precision, load_time = load_reranker(path)
        t0 = time.time()
        per_query = []
        for q in queries:
            dense_texts, fusion_topk = retrieved[q["id"]]
            t_enc = time.time()
            d = rescore(model, q["query"], dense_texts)
            fu = rescore(model, q["query"], fusion_topk)
            enc_time = round(time.time() - t_enc, 4)
            row = {
                "qid": q["id"], "category": q["category"], "lang": q["lang"],
                "model": label, "precision": precision,
                "cosine@1_dense": round(d["cos1"], 4),
                "mean@5_dense": round(d["mean5"], 4),
                "drop_1to2_dense": round(d["drop"], 4),
                "cosine@1_fusion": round(fu["cos1"], 4),
                "mean@5_fusion": round(fu["mean5"], 4),
                "drop_1to2_fusion": round(fu["drop"], 4),
                "rerank_time_s": enc_time,
            }
            rows.append(row)
            per_query.append(row)
        rerank_total = round(time.time() - t0, 1)

        def col(name):
            return np.array([r[name] for r in per_query], dtype=float)

        delta1 = col("cosine@1_fusion") - col("cosine@1_dense")
        delta5 = col("mean@5_fusion") - col("mean@5_dense")
        summary["models"][label] = {
            "precision": precision,
            "load_time_s": load_time,
            "rerank_time_total_s": rerank_total,
            "mean": {k: round(float(col(k).mean()), 4) for k in
                     ["cosine@1_dense", "mean@5_dense", "drop_1to2_dense",
                      "cosine@1_fusion", "mean@5_fusion", "drop_1to2_fusion"]},
            "delta_fusion_minus_dense": {
                "cosine@1": round(float(delta1.mean()), 4),
                "cosine@1_ci95": [round(v, 4) for v in bootstrap_ci(delta1)],
                "mean@5": round(float(delta5.mean()), 4),
                "mean@5_ci95": [round(v, 4) for v in bootstrap_ci(delta5)],
            },
        }
        del model
        torch.cuda.empty_cache()
        print(f"[r2]   {label}: precision={precision} load={load_time}s "
              f"rerank={rerank_total}s", flush=True)

    results_path = os.path.join(args.out_dir, "r2_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = os.path.join(args.out_dir, "r2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[r2] wrote {results_path} and {summary_path}", flush=True)
    print(json.dumps(summary["models"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if os.environ.get("R2_NO_TQDM") is None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
