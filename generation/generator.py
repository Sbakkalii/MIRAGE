import torch
from time import time


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
