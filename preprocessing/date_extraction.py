import re

import pandas as pd
from dateparser.search import search_dates
from dateparser_data.settings import default_parsers


def contains_explicit_1_january(text: str) -> bool:
    """
    Check if the text contains an explicit mention of 1st January.
    :param text: the text to check
    :return: True if the text contains an explicit mention of 1st January, False otherwise
    """
    patterns = [
        r"\b0?1[\/\-\. ]?0?1\b",  # 01/01, 1/1, 01-01, etc.
        r"\b(1er|1|01)[^\d]?(janvier|january)\b",  # 1 janvier, 1er janvier, 01 janvier
        r"\b(janvier|january)[^\d]*(1er|1|01)\b",  # janvier 1, january 1st
        r"\b(1st|first) of (january|janvier)\b"  # 1st of January
    ]
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def has_explicit_date_pattern(text: str) -> bool:
    """
    Check if the text contains an explicit date pattern. In case of ambiguous date.
    :param text: the text to check
    :return: True if the text contains an explicit date pattern, False otherwise
    """
    numeric_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}\b"
    ]

    textual_patterns = [
        r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{2,4}\b"
    ]

    all_patterns = numeric_patterns + textual_patterns

    for pattern in all_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def split_on_conjunctions(text: str) -> list:
    """
    Split the text on conjunctions (et, and, , or ;).
    :param text: the text to split
    :return: the list of segments
    """

    text = re.sub(r"\([^)]*\)", "", text)

    return re.split(r"\s+(et|and|,|;)\s+", text)


def is_pure_date_expression(text: str) -> bool:
    """
    Check if the text is a pure date expression.
    :param text: the text to check
    :return: True if the text is a pure date expression, False otherwise
    """
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b",  # 15/04[/2025]
        r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\b",
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b"
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def extract_date(df: pd.DataFrame) -> pd.DataFrame:
    parsers = [parser for parser in default_parsers if parser != 'relative-time']

    for i, row in df.iterrows():
        lang = row["meta"]["lang"]
        res_list = []

        try:
            # On divise le texte sur les conjonctions
            segments = [seg.strip() for seg in split_on_conjunctions(row["text"]) if
                        seg.strip().lower() not in ["et", "and", ",", ";"] and seg.strip() != ""]

            for segment in segments:
                dates_found = search_dates(
                    segment,
                    languages=["fr", "en"],
                    settings={
                        'PREFER_DAY_OF_MONTH': 'first',
                        'PREFER_MONTH_OF_YEAR': 'first',
                        'DATE_ORDER': 'DMY' if lang == 'fr' else 'MDY',
                        'PARSERS': parsers,
                    }
                )

                if dates_found:
                    for matched_text, result in dates_found:
                        res = {}

                        if result.day == 1 and result.month == 1:
                            if contains_explicit_1_january(matched_text):
                                res["year"] = result.year if has_explicit_date_pattern(matched_text) else None
                                res["month"] = result.month
                                res["day"] = result.day
                            else:
                                res["year"] = result.year if has_explicit_date_pattern(matched_text) else None
                                res["month"] = None
                                res["day"] = None
                        else:
                            res["year"] = result.year if has_explicit_date_pattern(matched_text) else None
                            res["month"] = result.month
                            res["day"] = result.day

                        res["hour"] = result.hour if result.hour else None
                        res["minute"] = result.minute if result.minute else None
                        res["second"] = result.second if result.second else None

                        # False positive check
                        if res["day"] and res["month"] and not is_pure_date_expression(matched_text):
                            break

                        if res["day"] or res["month"] or res["year"]:
                            res_list.append(res)

        except Exception as e:
            res_list = [{"error": str(e)}]

        df.at[i, "meta"]["dates"] = res_list

    return df
