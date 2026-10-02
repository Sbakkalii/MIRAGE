from itertools import islice

from datasets import load_dataset


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