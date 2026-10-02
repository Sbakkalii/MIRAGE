import os
from typing import Literal
import datetime
import torch

import pandas as pd
import numpy as np
from numpy import ndarray
from sentence_transformers import SentenceTransformer


def vectorisation(df: pd.DataFrame, model_name: str, backend: Literal["torch", "onnx", "openvino"],
                  batch_size: int) -> np.ndarray:
    """
    Vectorises the text data in the dataframe using the specified model and backend.
    :param batch_size: The batch size to use for vectorization
    :param model_name: the name of the model to use for vectorization
    :param backend: the backend to use for vectorization (e.g. 'onnx', 'torch')
    :param df: the dataframe containing the text data to vectorise
    :return: a numpy array of the vectorised text data
    """

    if os.path.exists("data/embeddings.npy"):
        print(f"[INFO - {datetime.datetime.now()}] Embeddings already exist. Loading from file...")
        embeddings = np.load("data/embeddings.npy")
        return embeddings

    print(f"[INFO - {datetime.datetime.now()}] Loading model {model_name} with backend {backend}...")
    time = datetime.datetime.now()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(model_name, trust_remote_code=True, backend=backend)
    embedding_model.to(device)

    print(f"[INFO - {datetime.datetime.now()}] Model loaded in {datetime.datetime.now() - time}.")

    print(f"[INFO - {datetime.datetime.now()}] Vectorising data...")

    # TODO : récupérer la taille moyenne du text

    df["chunk"] = df["meta"].apply(lambda x: "À propos de " + x.get("title", "") + "\n" + df["text"] if x.get(
        "lang") == "fr" else "About " + x.get("title", "") + "\n" + df["text"])
    texts = df["text"].tolist()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings_batch = embedding_model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            device=device
        )
        embeddings.append(embeddings_batch)

    print(f"[INFO - {datetime.datetime.now()}] Data vectorised in {datetime.datetime.now() - time}.")

    save_dir = "data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    embeddings = np.vstack(embeddings)
    np.save("data/embeddings.npy", embeddings)
    return embeddings
