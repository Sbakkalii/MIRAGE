import numpy as np
import pandas as pd
from pymilvus import MilvusClient
from tqdm import tqdm
import datetime

from preprocessing import preprocess_data
from embedding import vectorisation

def indexation(df:pd.DataFrame, embeddings:np.ndarray, client:MilvusClient, collection_name:str) -> None:
    """
    Indexation of the data in the database.
    :param collection_name: the name of the collection to index the data in.
    :param embeddings: the embeddings to index.
    :param df: the dataframe to index.
    :param client: the client to use to index the data.
    :return: None
    """

    if len(df) != len(embeddings):
        raise ValueError(f"[ERROR - {datetime.datetime.now()}]  Mismatch: len(df) = {len(df)}, len(embeddings) = {len(embeddings)}")


    if not client.has_collection(collection_name):
        raise ValueError(f"[ERROR - {datetime.datetime.now()}]  Collection {collection_name} does not exist in Milvus.")

    df = df.reset_index(drop=True)

    data = []

    print(f"[INFO - {datetime.datetime.now()}]  Indexing data in Milvus...")
    time = datetime.datetime.now()

    for i, row in df.iterrows():

        meta = row["meta"]
        docid = meta.get("docid")

        existing_docs = client.query(
            collection_name=collection_name,
            filter=f'docid == "{docid}"',
            output_fields=["docid"]
        )

        if existing_docs:
            continue

        entry = {
            "id": i,
            "vector": embeddings[i].tolist(),
            "text": row["text"],
            "docid": meta.get("docid"),
            "title": meta.get("title"),
            "lang": meta.get("lang"),
            "dates": meta.get("dates", []),
            "entities": meta.get("entities", [])
        }

        data.append(entry)

    batch_size = 5000

    for i in tqdm(range(0, len(data), batch_size), desc="Insertion dans Milvus"):
        batch = data[i:i + batch_size]
        client.insert(collection_name=collection_name, data=batch)

    print(f"[INFO - {datetime.datetime.now()}] {len(data)} documents indexed in {datetime.datetime.now() - time} seconds.")


def add_new_documents(documents, client, collection_name, model_name, framework, batch_size):
    """
    Ajoute de nouveaux documents à la base de données.
    :param documents: Liste de nouveaux documents à ajouter.
    :param client: Client Milvus.
    :param collection_name: Nom de la collection.
    :param model_name: Modèle d'embedding.
    :param framework: Framework utilisé pour l'embedding.
    :param batch_size: Taille des lots pour l'embedding.
    """
    # Prétraitement
    processed_data = preprocess_data(documents) # TODO

    # Génération des embeddings
    embeddings = vectorisation(
        processed_data,
        model_name,
        framework,
        batch_size
    )

    # Indexation
    indexation(
        df=processed_data,
        embeddings=embeddings,
        client=client,
        collection_name=collection_name,
    )