import pandas as pd


def chunking(df : pd.DataFrame, chunk_size:int) -> pd.DataFrame:
    """
    Chunking the dataframe into smaller chunks of a specified size.
    :param df: the dataframe to chunk
    :param chunk_size: the size of each chunk
    :return: a new dataframe with the chunks
    """

    # TODO : chunking by chunk_size

    chunk_text = []
    chunk_meta = []

    for index, row in df.iterrows():
        text = row["text"]
        title = row["title"]
        docid = row["docid"]
        lang = row["lang"]
        sentences = text.split(". ")

        for sentence in sentences:

            chunk_text.append(sentence)
            chunk_meta.append({
                "title": title,
                "docid": docid,
                "lang": lang
            })

    if len(chunk_text) != len(chunk_meta):
        raise ValueError("Chunk text and metadata lengths do not match.")

    df_chunk = pd.DataFrame({"text": chunk_text, "meta": chunk_meta})
    df_chunk = df_chunk.drop_duplicates(subset=["text"])
    df_chunk = df_chunk.dropna(subset=["text"])

    return df_chunk