# preprocessing/pretraitment.py
import ast
import os
import pandas as pd
from tqdm import tqdm
import torch
import datetime
import math
import torch.multiprocessing as mp
import gc

from .qa_handler import qa_generation_worker, log_gpu_memory
from .chunking import chunking
from .content_extraction import load_lang
from .text_cleaning import clean_text
from .date_extraction import extract_date
from .NER import enrich_df_with_ner_pipe


def run_ner_in_process(df_input_path: str, df_output_path: str):
    """
    A function to run the NER process in a separate multiprocessing context.
    :param df_input_path: The path to the input DataFrame in Parquet format.
    :param df_output_path: The path to save the output DataFrame in Parquet format.
    :return: None
    """
    print("[NER PROCESS] Démarrage du processus NER isolé sur GPU...")
    try:
        df_to_process = pd.read_parquet(df_input_path)
        print(f"[NER PROCESS] Fichier d'entrée chargé, {len(df_to_process)} lignes à traiter.")

        if df_to_process.empty:
            df_to_process.to_parquet(df_output_path)
            return

        df_enriched = enrich_df_with_ner_pipe(df_to_process)
        df_enriched.to_parquet(df_output_path)
        print(f"[NER PROCESS] Traitement NER terminé. {len(df_enriched)} lignes enrichies sauvegardées.")

    except Exception as e:
        print(f"[NER PROCESS] [ERREUR] Le processus NER a échoué: {e}")
        pd.DataFrame().to_parquet(df_output_path)
    finally:
        print("[NER PROCESS] Processus NER terminé.")


def process_segments_multi_gpu(segments_for_qab: list, llm_model_name: str, qa_params: dict) -> list:
    """
    A function to process segments for Q/A generation using multiple GPUs.
    :param segments_for_qab: List of segments to process for Q/A generation
    :param llm_model_name: The name of the LLM model to use for Q/A generation
    :param qa_params: Parameters for Q/A generation such as batch size and max tokens
    :return: List of processed segments with Q/A generation results
    """
    all_processed_segments = []
    processes = []

    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0: return []

        num_workers = num_gpus
        print(f"[Q/A ORCHESTRATEUR] Utilisation de {num_workers} workers sur {num_gpus} GPUs.")

        manager = mp.Manager()
        tasks_queue = manager.Queue()
        results_queue = manager.Queue()
        stop_event = manager.Event()

        chunk_size = qa_params.get('batch_size_llm', 128)
        total_chunks = math.ceil(len(segments_for_qab) / chunk_size)
        if total_chunks == 0: return []

        for i in range(0, len(segments_for_qab), chunk_size):
            tasks_queue.put(segments_for_qab[i:i + chunk_size])

        for _ in range(num_workers):
            tasks_queue.put(None)

        for worker_id in range(num_workers):
            p = mp.Process(target=qa_generation_worker,
                           args=(tasks_queue, results_queue, stop_event, worker_id, worker_id, llm_model_name,
                                 qa_params))
            processes.append(p)
            p.start()

        with tqdm(total=total_chunks, desc="Génération Q/A") as pbar:
            for _ in range(total_chunks):
                if stop_event.is_set():
                    raise RuntimeError("Erreur OOM signalée par un worker Q/A.")
                all_processed_segments.extend(results_queue.get(timeout=None))
                pbar.update(1)

    finally:
        print("[Q/A ORCHESTRATEUR] Nettoyage des workers...")
        for p in processes:
            p.join(timeout=10)
            if p.is_alive(): p.terminate()
        print("[Q/A ORCHESTRATEUR] Workers arrêtés.")

    return all_processed_segments


def preprocess_data(
        languages: list[str],
        max_docs_per_lang: int,
        nb_chunk_for_parallel_ner: int,
        llm_model_name_for_qa: str,
        new_docs=False
) -> pd.DataFrame:
    """
    Main function to preprocess data for Q/A generation.
    :param languages: The list of languages to process
    :param max_docs_per_lang: Maximum number of documents to load per language
    :param nb_chunk_for_parallel_ner: Number of chunks for parallel NER processing
    :param llm_model_name_for_qa: The name of the LLM model to use for Q/A generation
    :param new_docs: Boolean flag to indicate if new documents should be processed
    :return: pd.DataFrame containing the final processed data for Q/A generation
    """

    t_time = datetime.datetime.now()
    save_dir = 'data'
    os.makedirs(save_dir, exist_ok=True)

    final_output_path = os.path.join(save_dir, "final_qab_data.parquet")

    if not new_docs and os.path.exists(final_output_path):
        print(f"[MAIN] Chargement depuis le checkpoint : {final_output_path}")
        return pd.read_parquet(final_output_path)

    print(f"[MAIN] Début du prétraitement complet.")

    print(f"[MAIN] Chargement des documents...")
    all_docs = [doc for lang in languages for doc in load_lang(lang, max_docs_per_lang)]
    df = pd.DataFrame(all_docs)
    del all_docs
    gc.collect()

    if df.empty: return df
    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)
    df_initial_chunks = chunking(df, 0)
    del df
    gc.collect()

    if df_initial_chunks.empty: return df_initial_chunks
    df_initial_chunks = extract_date(df_initial_chunks)
    df_initial_chunks['meta'] = df_initial_chunks['meta'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, dict) else {}))

    ner_input_path = os.path.join(save_dir, "temp_ner_input.parquet")
    ner_output_path = os.path.join(save_dir, "temp_ner_output.parquet")
    df_initial_chunks.to_parquet(ner_input_path)
    del df_initial_chunks
    gc.collect()

    print("[MAIN] Lancement du processus NER isolé sur GPU...")
    ner_process = mp.Process(target=run_ner_in_process, args=(ner_input_path, ner_output_path))
    ner_process.start()
    ner_process.join()
    print("[MAIN] Le processus NER est terminé.")

    if not os.path.exists(ner_output_path) or os.path.getsize(ner_output_path) == 0:
        print("[ERREUR FATALE] Le processus NER a échoué.")
        if os.path.exists(ner_input_path): os.remove(ner_input_path)
        if os.path.exists(ner_output_path): os.remove(ner_output_path)
        return pd.DataFrame()

    df_ner_enriched = pd.read_parquet(ner_output_path)
    os.remove(ner_input_path)
    os.remove(ner_output_path)

    if torch.cuda.is_available(): log_gpu_memory("[MAIN]", "Après la fin du processus NER")

    if df_ner_enriched.empty:
        return df_ner_enriched

    segments_for_qab = [{**row.to_dict()} for _, row in df_ner_enriched.iterrows()]
    del df_ner_enriched
    gc.collect()

    qa_params = {"batch_size_llm": 128, "batch_size_bertscore": 256, "max_new_tokens_qa": 350, "temperature_qa": 0.3}
    print(f"[MAIN] Lancement de la génération Q/A...")

    try:
        all_processed_segments = process_segments_multi_gpu(segments_for_qab=segments_for_qab,
                                                            llm_model_name=llm_model_name_for_qa, qa_params=qa_params)
    except RuntimeError as e:
        print(f"[ERREUR FATALE] {e}")
        return pd.DataFrame()

    df_contextual_chunks = pd.DataFrame(all_processed_segments)
    df_contextual_chunks.to_parquet(final_output_path, index=False)
    print(f"[MAIN] Données finales sauvegardées dans {final_output_path}")

    print(f"[MAIN] Prétraitement complet terminé en {datetime.datetime.now() - t_time}")
    return df_contextual_chunks
