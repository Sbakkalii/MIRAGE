from time import time
from datetime import datetime
from typing import List, Dict

from datasets import Dataset
from pymilvus import MilvusClient
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import csv
import json
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""
if os.getenv("OPENAI_BASE_URL"):
    os.environ.setdefault("OPENAI_API_BASE", os.environ["OPENAI_BASE_URL"])

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
                output_fields=["text", "title", "docid"]
            )
        except Exception as e:
            print(f"[ERROR] Error during Milvus search: {e}")
            continue

        current_query_hits = results_for_query[0] if results_for_query and len(results_for_query) > 0 else []

        for rank, hit in enumerate(current_query_hits):
            doc_id = str(hit["id"])

            if doc_id not in doc_data_map:
                entity = hit.get("entity", {})
                doc_data_map[doc_id] = {
                    "text": entity.get("text", "Texte non trouvé"),
                    "title": entity.get("title") or "Titre inconnu",
                    "docid": entity.get("docid") or "Source_inconnue_docid",
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
                "meta": {
                    "docid": doc_data_map[doc_id].get("docid", "Source_inconnue_docid"),
                    "title": doc_data_map[doc_id].get("title", "Titre inconnu"),
                },
            }
            final_results.append(result_doc)
        else:
            print(f"[WARNING] Document ID {doc_id} not found in doc_data_map.")

    print(f"[INFO] Fusion RRF : {len(final_results)} documents found.")
    return final_results

def build_structured_context(context_docs: list, max_chars_per_doc_chunk: int = 1000) -> str:
    """
    Build a structured context from the provided documents.
    :param context_docs: The list of context documents to be processed.
    :param max_chars_per_doc_chunk: The maximum number of characters per document chunk.
    :return:
    """
    final_context_parts = []

    processed_articles = {}

    for doc in context_docs:
        meta = doc.get('meta', {})
        doc_id_full = meta.get('docid', 'Source_inconnue_docid')
        doc_id_base = doc_id_full.split('#')[0] if isinstance(doc_id_full, str) else doc_id_full

        doc_title = meta.get('title', 'Titre inconnu')
        text_chunk = doc.get('text', '')

        if doc_id_base not in processed_articles:
            processed_articles[doc_id_base] = {'title': doc_title, 'texts': []}

        processed_articles[doc_id_base]['texts'].append(text_chunk[:max_chars_per_doc_chunk])

    for doc_id_base, article_data in processed_articles.items():
        article_title = article_data['title']
        concatenated_texts = "\n".join(article_data['texts'])
        final_context_parts.append(
            f"Extrait de l'article \"{article_title}\" (Source ID: {doc_id_base}):\n{concatenated_texts}")

    return "\n\n---\n\n".join(final_context_parts)


def generate_response(query: str, context_docs: list, llm_model, llm_tokenizer, max_new_tokens=500) -> str:
    """
    Generate a response using the LLM model based on the provided query and context documents.
    :param query: The input query string.
    :param context_docs: The context documents to be used for generating the response.
    :param llm_model: The LLM model to be used for generation.
    :param llm_tokenizer: The tokenizer for the LLM model.
    :param max_new_tokens: The maximum number of new tokens to generate.
    :return: The generated response string.
    """

    context_text = build_structured_context(context_docs)

    prompt_content = f"""
        Tu es un expert en histoire et tu dois répondre à la question suivante en prenant en compte les extraits de journaux historiques fournis.
        Ta tâche est de répondre à la question posée en utilisant EXCLUSIVEMENT les informations contenues dans les "Extraits de journaux" fournis ci-dessous.
        Ne fais aucune supposition et n'utilise aucune connaissance extérieure.
        Si les extraits ne contiennent pas l'information nécessaire pour répondre à la question, tu DOIS explicitement indiquer : "Je ne peux pas répondre à cette question en me basant uniquement sur les informations fournies."
        Réponds dans la langue de la question.
        Vérifie attentivement que chaque information que tu extrais concerne bien l'événement principal de la question et non un autre événement mentionné dans le contexte, sauf si le lien de causalité est explicite.
        Si tu identifies des acteurs, assure-toi que leurs relations sont explicitement décrites dans les extraits avant de les affirmer.
        Ne fais pas référence à toi-même en tant que "modèle d'IA".
        Une conséquence est un résultat ou un effet postérieur à un événement. Un événement qui déclenche ou cause un autre événement n'est pas une conséquence de cet événement lui-même.

        Extraits de journaux :
        ---
        {context_text}
        ---
        Question : {query}
    """

    messages = [
        {"role": "user", "content": prompt_content.strip()}]

    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print("Prompt formaté pour le modèle (début) :")
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)

    print("\n[INFO] Génération de la réponse...")
    start_time = time()
    generated_ids = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        pad_token_id=llm_tokenizer.pad_token_id
    )

    response_text = llm_tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    end_time = time()
    print(f"[INFO] Réponse générée en {end_time - start_time:.2f} secondes.")

    return response_text.strip()

_queries_csv = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "queries_50.csv"
)
if os.path.exists(_queries_csv):
    with open(_queries_csv, newline="") as _f:
        test_questions = [_row["query"] for _row in csv.DictReader(_f)]
else:
    test_questions = [
        "Expliquez en détail comment les voyages interstellaires des Romains ont influencé l'architecture des temples égyptiens"
    ]


def run_complete_rag_pipeline_for_question(query_text, llm_generator_model, llm_generator_tokenizer,
                                          milvus_client, embedding_encoder,
                                          llm_variation_model, llm_variation_tokenizer,
                                          cross_encoder_model_for_reranking):

    retrieved_docs = multi_query_fusion(
        query=query_text,
        collection_name="rag_v1",
        client=milvus_client,
        encoder=embedding_encoder,
        llm_variation_model=llm_variation_model,
        llm_variation_tokenizer=llm_variation_tokenizer,
        num_variations_to_generate=5,
    )

    reranked_docs = rerank_documents(
        query=query_text,
        docs=retrieved_docs,
        model=cross_encoder_model_for_reranking,
    )

    contexts_list = [doc['text'] for doc in reranked_docs]

    llm_answer = generate_response(
        query=query_text,
        context_docs=reranked_docs,
        llm_model=llm_generator_model,
        llm_tokenizer=llm_generator_tokenizer,
        max_new_tokens=2000
    )
    return {"question": query_text, "answer": llm_answer, "contexts": contexts_list}

results_data = []

_script_dir = os.path.dirname(os.path.abspath(__file__))
_answers_json = os.path.join(_script_dir, "r3_answers.json")
_progress_path = os.path.join(_script_dir, "r3_progress.json")
if os.path.exists(_answers_json):
    # Answers generated on the T4 notebook (notebooks/colab_t4_r2_r3.ipynb):
    # use them directly, skip local generation — and skip loading the
    # generation models entirely (they are not needed for judging).
    with open(_answers_json, encoding="utf-8") as _f:
        results_data = json.load(_f)
    print(f"[INFO] {len(results_data)} réponses chargées depuis r3_answers.json (génération T4)")
elif os.path.exists(_progress_path):
    with open(_progress_path, encoding="utf-8") as _f:
        results_data = json.load(_f)
    print(f"[INFO] Reprise: {len(results_data)} questions déjà évaluées")
_done = {item["question"] for item in results_data}

_missing = [q for q in test_questions if q not in _done]
if _missing:
    print(f"[INFO] {len(_missing)} questions sans réponse -> chargement des modèles de génération...")

    print("[INFO] Loading models...")

    model_name = "intfloat/multilingual-e5-large-instruct"
    collection_name = "rag_v1"

    model = SentenceTransformer(model_name, trust_remote_code=True)
    client = MilvusClient(f"../../{collection_name}_milvus.db")
    if client.has_collection(collection_name):
        client.load_collection(collection_name)

    llm_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Load LLM : {llm_model_name} on {device}...")

    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if device == "cuda" else None,
    )

    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
else:
    print("[INFO] Réponses complètes: chargement des modèles de génération ignoré")

for i, q in enumerate(test_questions):
    if q in _done:
        continue
    print(f"Processing question: {q}")
    rag_output = run_complete_rag_pipeline_for_question(
        q,                          # query_text
        llm_model,                  # llm_generator_model
        llm_tokenizer,              # llm_generator_tokenizer
        client,                     # milvus_client
        model,                      # embedding_encoder (SentenceTransformer)
        # Arguments pour la génération de variations :
        llm_model,                  # llm_variation_model
        llm_tokenizer,              # llm_variation_tokenizer
        # Argument pour le reranking :
        model                       # cross_encoder_model_for_reranking
    )
    results_data.append(rag_output)
    with open(_progress_path, "w") as _f:
        json.dump(results_data, _f, ensure_ascii=False, indent=2)

if not results_data:
    print("Aucune donnée collectée pour l'évaluation. Vérifie ton pipeline.")
else:
    # Convertir en Dataset Hugging Face
    dataset_dict = {
        "question": [item["question"] for item in results_data],
        "answer": [item["answer"] for item in results_data],
        "contexts": [item["contexts"] for item in results_data],

    }

    dataset = Dataset.from_dict(dataset_dict)

    metrics_to_evaluate = [
        faithfulness,  # Est-ce que la réponse est basée sur le contexte ?
        answer_relevancy,  # La réponse est-elle pertinente par rapport à la question ?
    ]

    # D3 amendments (2026-09-29): judge = DeepSeek API (DEEPSEEK_API_KEY;
    # DEEPSEEK_BASE_URL overrides) or any OpenAI-compatible endpoint
    # (OPENAI_BASE_URL/OPENAI_API_KEY, e.g. the local FREELLMAPI gateway);
    # embeddings = local Ollama (OLLAMA_BASE_URL + RAGAS_EMBED_MODEL) — the
    # DeepSeek API has no embeddings endpoint. Model ids stay env-overridable
    # because neither endpoint serves the ragas defaults (gpt-4o-mini /
    # text-embedding-ada-002).
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    _judge_model = os.getenv("RAGAS_JUDGE_MODEL", "deepseek-v4-flash")
    _embed_model = os.getenv("RAGAS_EMBED_MODEL", "qwen3-embedding:4b")
    if os.getenv("DEEPSEEK_API_KEY"):
        _judge_base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        _judge_key = os.environ["DEEPSEEK_API_KEY"]
    else:
        _judge_base = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        _judge_key = os.getenv("OPENAI_API_KEY") or None
    _embed_base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")

    _judge_kwargs = dict(model=_judge_model, base_url=_judge_base,
                         api_key=_judge_key, temperature=0)
    if "deepseek" in _judge_base:
        # V4 flash thinks by default; judge calls are short classifications.
        _judge_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    judge_llm = LangchainLLMWrapper(ChatOpenAI(**_judge_kwargs))
    embedder = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=_embed_model,
        openai_api_base=_embed_base,
        openai_api_key=os.getenv("OLLAMA_API_KEY") or "ollama",
        # Ollama's OpenAI-compat endpoint accepts only string inputs; the
        # default "length-safe" path sends token-id arrays -> HTTP 400.
        check_embedding_ctx_length=False,
    ))

    print(f"Début de l'évaluation RAGAS sur {len(dataset)} exemples...")
    print(f"[R3] judge={_judge_model} @ {_judge_base} | "
          f"embed={_embed_model} @ {_embed_base}")
    result = evaluate(
        dataset,
        metrics=metrics_to_evaluate,
        llm=judge_llm,
        embeddings=embedder,
    )

    print("\n===== Résultats de l'Évaluation RAGAS =====")
    print(result)

    df_results = result.to_pandas()
    print("\n===== Résultats RAGAS (Pandas DataFrame) =====")
    print(df_results.head())

    _qmeta = {}
    if os.path.exists(_queries_csv):
        with open(_queries_csv, newline="") as _f:
            for _row in csv.DictReader(_f):
                _qmeta[_row["query"]] = (_row["id"], _row["category"], _row["lang"])
    # ragas 0.2.15 to_pandas() renames the input column question -> user_input
    _qid_col = "user_input" if "user_input" in df_results.columns else "question"
    df_results["qid"] = df_results[_qid_col].map(lambda x: _qmeta.get(x, ("", "", ""))[0])
    df_results["category"] = df_results[_qid_col].map(lambda x: _qmeta.get(x, ("", "", ""))[1])
    df_results["lang"] = df_results[_qid_col].map(lambda x: _qmeta.get(x, ("", "", ""))[2])
    df_results.to_csv(os.path.join(_script_dir, "ragas_evaluation_results.csv"), index=False)

    def _agg(frame):
        out = {"n": int(len(frame))}
        for col in ("faithfulness", "answer_relevancy"):
            if col in frame.columns:
                out[col] = {
                    "mean": round(float(frame[col].mean()), 4),
                    "sd": round(float(frame[col].std(ddof=1)), 4) if len(frame) > 1 else 0.0,
                }
        return out

    summary = {
        "overall": _agg(df_results),
        "by_category": {k: _agg(g) for k, g in df_results.groupby("category") if k},
        "by_lang": {k: _agg(g) for k, g in df_results.groupby("lang") if k},
        "judge": _judge_base,
        "judge_model": _judge_model,
        "embed_model": _embed_model,
        "embed_base": _embed_base,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with open(os.path.join(_script_dir, "r3_summary.json"), "w") as _f:
        json.dump(summary, _f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))