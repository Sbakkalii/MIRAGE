"""Grounding check for benchmarks/evaluation/queries_50.csv against data/chunks.parquet.

Answerable queries (entity/event) must have corpus support (accent/case-insensitive
phrase hits in chunk text or title). Unanswerable queries must have no chunk
containing both components of their absurd conjunction.

Run after indexation/build_index.py has written data/chunks.parquet (the file is
saved as soon as the NER phase completes).

Exit code 0 = all checks pass; 1 = failures listed on stdout.
"""

import csv
import json
import os
import sys
import unicodedata

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS = os.path.join(REPO_ROOT, "data", "chunks.parquet")
QUERIES = os.path.join(REPO_ROOT, "benchmarks", "evaluation", "queries_50.csv")

# Phrase that must be present in the corpus for each answerable query id.
# Unanswerable queries list component pairs that must NOT co-occur in one chunk.
TARGETS = {
    # entity, fr
    "q01": ["meillet"],
    "q02": ["paul valery", "valery"],
    "q03": ["marie curie"],
    "q04": ["champollion"],
    "q05": ["clemenceau"],
    "q06": ["pasteur"],
    "q07": ["victor hugo"],
    "q08": ["jean jaures", "jaures"],
    "q09": ["napoleon"],
    "q10": ["averroes"],
    # entity, en
    "q11": ["abraham lincoln", "lincoln"],
    "q12": ["winston churchill", "churchill"],
    "q13": ["marie curie"],
    "q14": ["charles darwin", "darwin"],
    "q15": ["isaac newton", "newton"],
    "q16": ["leonard de vinci", "leonardo da vinci"],
    "q17": ["alan turing", "turing"],
    "q18": ["nelson mandela", "mandela"],
    "q19": ["albert einstein", "einstein"],
    "q20": ["florence nightingale", "nightingale"],
    # event, fr (accept FR or EN surface form in corpus)
    "q21": ["premiere guerre mondiale", "first world war"],
    "q22": ["revolution francaise", "french revolution"],
    "q23": ["seconde guerre mondiale", "world war ii", "second world war"],
    "q24": ["empire romain", "roman empire"],
    "q25": ["constantinople", "byzance"],
    "q26": ["croisade", "crusade"],
    "q27": ["independance des etats-unis", "american revolution", "war of independence"],
    "q28": ["decolonisation", "decolonization"],
    "q29": ["peste noire", "black death"],
    "q30": ["reforme protestante", "protestant reformation"],
    # event, en
    "q31": ["american civil war", "guerre de secession"],
    "q32": ["premiere guerre mondiale", "first world war"],
    "q33": ["chute de l'empire romain", "fall of the roman empire"],
    "q34": ["revolution francaise", "french revolution"],
    "q35": ["reforme protestante", "protestant reformation"],
    "q36": ["grande depression", "great depression"],
    "q37": ["mur de berlin", "berlin wall"],
    "q38": ["apartheid"],
    "q39": ["peste noire", "black death"],
    "q40": ["revolution d'octobre", "october revolution"],
}

# Unanswerable: the two components must never appear together in one chunk.
PAIRS = {
    "q41": ["interstella", "romain"],       # interstellar + romans (pinned)
    "q42": ["chevalier", "internet"],
    "q43": ["pharaon", "chemin de fer"],
    "q44": ["guerre de troie", "banque centrale"],
    "q45": ["marie curie", "reseaux sociaux"],
    "q46": ["interstellar", "roman trade"],
    "q47": ["pyramides", "moon landing"],   # moon landing in EN text
    "q48": ["napoleon", "smartphone"],
    "q49": ["magna carta", "internet"],
    "q50": ["cleopatre", "satellite"],
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.casefold()


def main() -> int:
    if not os.path.exists(CHUNKS):
        print(f"missing {CHUNKS} — run build_index.py first")
        return 2
    df = pd.read_parquet(CHUNKS, columns=["text", "meta"])
    norm_text = df["text"].map(normalize)
    norm_title = df["meta"].map(lambda m: normalize(json.loads(m).get("title", "")))
    haystack = norm_text + " " + norm_title
    del df, norm_text, norm_title

    with open(QUERIES) as f:
        rows = list(csv.DictReader(f))

    failures = []
    for row in rows:
        qid, category = row["id"], row["category"]
        if category in ("entity", "event"):
            phrases = TARGETS.get(qid)
            if not phrases:
                failures.append(f"{qid}: no target phrases defined")
                continue
            counts = {p: int(haystack.str.contains(p, regex=False).sum()) for p in phrases}
            best = max(counts.values())
            status = "PASS" if best >= 3 else "FAIL"
            if status == "FAIL":
                failures.append(f"{qid}: hits={counts}")
            print(f"{qid} [{category}/{row['lang']}] {status} hits={counts}")
        else:
            pair = PAIRS.get(qid)
            if not pair:
                failures.append(f"{qid}: no absurd pair defined")
                continue
            a = haystack.str.contains(normalize(pair[0]), regex=False)
            b = haystack.str.contains(normalize(pair[1]), regex=False)
            both = int((a & b).sum())
            status = "PASS" if both == 0 else "FAIL"
            if status == "FAIL":
                failures.append(f"{qid}: co-occurring chunks={both}")
            print(f"{qid} [unanswerable/{row['lang']}] {status} co-occurrence={both}")

    print()
    if failures:
        print("FAILURES:")
        for line in failures:
            print(" -", line)
        return 1
    print("all 50 queries pass grounding checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
