from typing import List, Dict

from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import torch


def generate_query_variations(original_query: str, llm_model, llm_tokenizer, num_variations: int = 3,
                              max_length: int = 64) -> list[str]:
    """
    Generate variations of a query using a language model.
    :param original_query: The original query to be reformulated.
    :param llm_model: The language model used for generating variations.
    :param llm_tokenizer: The tokenizer for the language model.
    :param num_variations: The number of variations to generate.
    :param max_length: The maximum length of the generated variations.
    :return:
    """
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
        pad_token_id=llm_tokenizer.pad_token_id
    )
    response_text = llm_tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    if "Reformulations :" in response_text:
        response_part = response_text.split("Reformulations :")[-1].strip()
        variations = [v.strip() for v in response_part.split("\n") if v.strip() and v.lower() != original_query.lower()]
        variations = variations[:num_variations]
    else:
        variations = [v.strip() for v in response_text.split("\n") if v.strip() and v.lower() != original_query.lower()]
        variations = variations[:num_variations]

    print(f"[INFO] Generated variations: {variations} for query: {original_query}")
    return variations

def multi_query_fusion(
        query: str,
        collection_name: str,
        client: MilvusClient,
        llm_variation_model: any,
        llm_variation_tokenizer: any,
        num_variations_to_generate: int = 3,
        generate_variations_flag: bool = True,
        top_k: int = 10,
        k_rrf: int = 60,
        encoder: SentenceTransformer = SentenceTransformer("intfloat/multilingual-e5-large-instruct"),

) -> List[Dict]:
    """
    Perform multi-query fusion to retrieve documents from a Milvus collection.
    :param k_rrf: Retrieval rank fusion parameter.
    :param generate_variations_flag: Flag to indicate whether to generate variations.
    :param num_variations_to_generate: The number of variations to generate for the query.
    :param llm_variation_tokenizer: The tokenizer for the LLM used to generate query variations.
    :param llm_variation_model: The LLM model used to generate query variations.
    :param collection_name: The name of the Milvus collection to search.
    :param client: The Milvus client instance.
    :param encoder: The encoder to use for encoding queries.
    :param query: The main query string.
    :param top_k: The number of top documents to retrieve.
    :return: A list of dictionaries containing document IDs, texts, and scores.
    """

    if generate_variations_flag and llm_variation_model and llm_variation_tokenizer:
        print(f"[INFO] Requête originale : {query}")
        variations = generate_query_variations(
            query,
            llm_variation_model,
            llm_variation_tokenizer,
            num_variations_to_generate
        )
        all_queries = [query] + variations
    else:
        if generate_variations_flag and (not llm_variation_model or not llm_variation_tokenizer):
            print("[WARNING] LLM model or tokenizer not provided. Using only the original query.")
        all_queries = [query]

    all_queries = sorted(list(set(all_queries)))
    print(f"[INFO] Request use for Fusion : {all_queries}")

    query_vectors = encoder.encode(all_queries, normalize_embeddings=True)

    doc_rrf_scores: Dict[str, float] = {}
    doc_data_map: Dict[str, Dict] = {}
    milvus_search_limit = top_k * 3

    for i, vec in enumerate(query_vectors):
        current_query_text = all_queries[i]
        print(f"[INFO] Milvus Search for : \"{current_query_text}\"")

        try:
            results_for_query = client.search(
                collection_name=collection_name,
                data=[vec.tolist()],
                anns_field="vector",
                limit=milvus_search_limit,
                output_fields=["text"]
            )
        except Exception as e:
            print(f"[ERROR] Error during Milvus search: {e}")
            continue

        current_query_hits = results_for_query[0] if results_for_query and len(results_for_query) > 0 else []

        for rank, hit in enumerate(current_query_hits):
            doc_id = str(hit["id"])

            if doc_id not in doc_data_map:
                doc_data_map[doc_id] = {
                    "text": hit.get("entity", {}).get("text", "Texte non trouvé"),
                    "title": hit.get("entity", {}).get("title", "Titre inconnu"),
                }

            rrf_score_for_hit = 1.0 / (k_rrf + rank + 1)
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + rrf_score_for_hit

    if not doc_rrf_scores:
        print("[WARNING] No documents found in Milvus search.")
        return []

    sorted_docs_by_rrf = sorted(doc_rrf_scores.items(), key=lambda item: item[1], reverse=True)

    final_results: List[Dict] = []
    for doc_id, combined_score in sorted_docs_by_rrf[:top_k]:
        if doc_id in doc_data_map:
            result_doc = {
                "id": doc_id,
                "text": doc_data_map[doc_id]["text"],
                "score": round(combined_score, 6),
                "title": doc_data_map[doc_id].get("title"),
            }
            final_results.append(result_doc)
        else:
            print(f"[WARNING] Document ID {doc_id} not found in doc_data_map.")

    print(f"[INFO] Fusion RRF : {len(final_results)} documents found.")
    return final_results
