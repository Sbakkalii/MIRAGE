# NER.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import ast
import gc


class NERProcessor:
    """
    A class to process Named Entity Recognition (NER) using a pre-trained model.
    This class initializes a NER pipeline with a specified model and device,
    processes text data in a DataFrame, and extracts named entities.
    """

    def __init__(self, model_name: str = "Babelscape/wikineural-multilingual-ner", device: str = "cuda:0"):
        """
        Initialize the NERProcessor with a specified model and device.
        :param model_name: The name of the pre-trained model to use for NER.
        :param device: The device to run the model on, e.g., "cuda:0" for GPU or "cpu" for CPU.
        """
        print(f"[NERProcessor] Initialisation sur le device : {device}")
        if not torch.cuda.is_available():
            raise RuntimeError("Le calcul sur GPU est demandé, mais CUDA n'est pas disponible.")

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)

        self.bert_ner = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device,
            batch_size=256
        )
        print("[NERProcessor] Pipeline NER prête.")

    def _post_process_entities(self, entities: list) -> list[tuple[str, str]]:
        """
        Post-process the entities extracted by the NER model.
        :param entities: A list of entities extracted by the NER model.
        :return: A cleaned list of tuples containing the word and its corresponding entity tag.
        """
        cleaned = []
        for entity in entities:
            word = entity['word']
            tag = entity['entity_group']
            if word and len(word) >= 3 and tag != "MISC" and tag is not None and tag != "None":
                cleaned.append((word.strip(), tag))
        return cleaned

    def process_dataframe(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame chunk to extract named entities using the NER pipeline.
        :param df_chunk: A DataFrame containing a 'text' column with text data to process.
        :return: A DataFrame with an additional 'entities' column containing the extracted named entities.
        """
        if df_chunk.empty or 'text' not in df_chunk.columns:
            return df_chunk

        print("[NERProcessor] Extraction des textes pour le traitement par lot...")
        texts = df_chunk['text'].fillna('').tolist()

        print(f"[NERProcessor] Traitement de {len(texts)} textes par la pipeline NER...")
        all_ents_batches = self.bert_ner(texts)
        print("[NERProcessor] Traitement par lot terminé.")

        if 'meta' not in df_chunk.columns:
            df_chunk['meta'] = [{} for _ in range(len(df_chunk))]
        else:
            df_chunk['meta'] = df_chunk['meta'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, dict) else {}))

        final_entities_list = [self._post_process_entities(ents) for ents in all_ents_batches]

        df_chunk['entities'] = final_entities_list
        df_chunk['meta'] = df_chunk.apply(lambda row: {**row['meta'], 'entities': row['entities']}, axis=1)
        df_chunk = df_chunk.drop(columns=['entities'])

        return df_chunk

    def release(self):
        """
        Release resources used by the NERProcessor.
        :return: None
        """
        print("[NERProcessor] Libération des ressources NER...")
        del self.bert_ner, self.model, self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def enrich_df_with_ner_pipe(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a DataFrame chunk with named entity recognition (NER) results.
    :param df_chunk: A DataFrame containing a 'text' column with text data to process.
    :return: A DataFrame with an additional 'entities' column containing the extracted named entities.
    """
    if df_chunk.empty:
        return df_chunk
    processor = None
    try:
        processor = NERProcessor()
        return processor.process_dataframe(df_chunk)
    finally:
        if processor:
            processor.release()
