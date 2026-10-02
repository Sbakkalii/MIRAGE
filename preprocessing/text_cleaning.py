import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean the input text by performing various preprocessing steps.
    :param text: the text to clean
    :return: cleaned text
    """

    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Replacing typographical quotation marks with single quotation marks
    text = text.replace('“', '"').replace('”', '"').replace('«', '"').replace('»', '"').replace("’", "'")

    # Replace \n with a space (or a period + space if it breaks a sentence)
    text = text.replace('\n', '. ')

    # Delete invisible control characters (except those already managed)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # Delete multiple spaces
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)

    # Delete multiple spaces around punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)

    # Delete spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)

    # Delete HTML tags and HTML entities
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&\w+;', '', text)

    # Delete unprintable characters or orphan symbols
    text = re.sub(r'[^\x20-\x7EÀ-ÿ€£$¥•–—’“”…°²³µ·]', '', text)

    return text.strip()
