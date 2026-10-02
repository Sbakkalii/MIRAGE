# tests/test_preprocessing.py
import torch
import torch.multiprocessing as mp
import sys
import os
import pandas as pd


def run_preprocessing():
    """
    Fonction principale contenant la logique de l'application.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from preprocessing.pretraitment import preprocess_data

    print("[INFO] - Démarrage de l'application...")

    languages = ['fr', 'en']
    max_docs_per_lang = 10000
    nb_chunk_for_parallel_ner = 200  # n'est plus utilisé pour le GPU, mais conservé pour la logique CPU si besoin
    llm_model_name_for_qa = "mistralai/Mistral-7B-Instruct-v0.3"

    if not torch.cuda.is_available():
        print("[ERREUR] CUDA n'est pas disponible.")
        sys.exit(1)

    print(f"[INFO] Nombre de GPUs détectés : {torch.cuda.device_count()}")
    print(f"[INFO] Lancement du prétraitement...")

    res_df = preprocess_data(
        languages=languages,
        max_docs_per_lang=max_docs_per_lang,
        nb_chunk_for_parallel_ner=nb_chunk_for_parallel_ner,
        llm_model_name_for_qa=llm_model_name_for_qa,
        new_docs=True,
    )

    if res_df is not None and not res_df.empty:
        print("\n[INFO] Aperçu des résultats finaux :")
        print(res_df.head())
        output_csv_path = "test_preprocessing_output_qab.csv"
        res_df.to_csv(output_csv_path, index=False)
        print(f"\n[INFO] Résultats complets sauvegardés dans : {output_csv_path}")
    else:
        print("[INFO] Le traitement n'a produit aucun résultat ou a échoué.")


if __name__ == "__main__":
    try:
        cache_dir = os.path.join(os.getcwd(), 'tmp_mpl_cache')
        os.environ['MPLCONFIGDIR'] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        mp.set_start_method('spawn', force=True)
        print("[INFO] Méthode de démarrage du multiprocessing définie sur 'spawn'.")
    except RuntimeError as e:
        print(f"[AVERTISSEMENT] La méthode de démarrage ne pouvait pas être changée: {e}")

    run_preprocessing()
