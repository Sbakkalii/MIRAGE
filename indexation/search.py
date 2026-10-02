import numpy as np
from pymilvus import MilvusClient


def test_search(query_vector, client:MilvusClient, collection_name:str, top_k:int=5) -> list[
    list[dict]]:
    """
    Test the search function of the Milvus client.
    :param query_vector: query vector to search for.
    :param top_k: number of top results to return.
    :param client: Milvus client to use to search the data.
    :return: None
    """

    results = client.search(
        collection_name=collection_name,
        data=query_vector,
        limit=top_k,
        output_fields=["text", "title", "lang"],
        params={
            "metric_type": "COSINE",
            "params": {
                "ef": 64
            }
        },
    )

    return results

def delete_by_doc_id(client, collection_name, doc_id):
    """
    Supprime un document de la collection en fonction de son doc_id.
    :param client: Client Milvus.
    :param collection_name: Nom de la collection.
    :param doc_id: Identifiant du document à supprimer.
    """
    filter_query = f'docid == "{doc_id}"'
    client.delete(
        collection_name=collection_name,
        filter=filter_query
    )
    print(f"Document avec doc_id={doc_id} supprimé.")