"""Build notebooks/colab_t4_r2_r3.ipynb (self-contained Colab T4 notebook).

Cells embed the colab_data.b64 blob produced by export_colab_data.py.
Every code cell is AST-validated before the notebook is written.
"""

import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__)
)
B64_PATH = os.path.join(HERE, "colab_data.b64")
OUT_PATH = os.path.join(HERE, "colab_t4_r2_r3.ipynb")

with open(B64_PATH) as f:
    B64 = f.read().strip()

MD_INTRO = """\
# Multilingual RAG — T2 re-runs on a T4 GPU (Kaggle or Colab)

Self-contained notebook (no uploads needed). Produces:

| Output | Section | What it is |
|---|---|---|
| `r2_results.csv` + `r2_summary.json` | 1 | Table 7 re-run: 50 queries × dense-vs-fusion × 4 rerankers |
| `r3_answers.json` | 2 | Answers + contexts for the 50 queries (RAGAS judging runs **locally afterwards**: DeepSeek judge + local Ollama embeddings — no judge reachable from the notebook) |

**How to run (Kaggle):** Settings → Accelerator → **GPU T4 x2** (or T4) →
**Run all**. Everything is written to `/kaggle/working`. Finish with
**Save Version → Save & Run All (Commit)** and download the three files from
the new version's **Output** tab.

**How to run (Colab):** Runtime → Change runtime type → **T4 GPU** → Run all,
then use the download cell.

Expected wall time ≈ 30–50 min (≈15–30 min of that is downloading the four
rerankers + Mistral ≈ 45–60 GB from the HuggingFace Hub).

**Provenance notes (recorded in the Stage-2 changelog):**
- Retrieval candidates were computed locally on the rebuilt cluster-recipe
  index (688,992 rows, e5-large-instruct fp32) and embedded in this notebook.
- Reranker scoring runs here in **fp16 on a T4** (cluster ran fp32; the local
  8 GB laptop fallback would have been CPU fp32 / 4-bit). `precision` is
  recorded per model in the outputs.
- Answer generation runs here in **Mistral-7B fp16** (the paper's A-14 local
  fallback was 4-bit).

**Bring back:** `r2_results.csv`, `r2_summary.json`, `r3_answers.json`.
"""

CELL_SETUP = """\
# Environment check + installs
import os, subprocess, sys, torch
assert torch.cuda.is_available(), (
    "Enable the GPU: Kaggle Settings -> Accelerator -> GPU T4, "
    "or Colab Runtime -> Change runtime type -> T4 GPU")
print("GPU:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
subprocess.check_call([sys.executable, "-m", "pip", "-q", "install",
                       "sentencepiece", "sentence-transformers", "scipy"])
BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
os.makedirs(BASE, exist_ok=True)
print("installs OK | BASE:", BASE)
"""

CELL_DATA = """\
# Decode the embedded experiment data (queries, retrieval candidates, expansions, R3 contexts)
import base64, gzip, json, os
B64 = "__B64__"
payload = gzip.decompress(base64.b64decode(B64))
open(f"{BASE}/colab_data.json", "wb").write(payload)
DATA = json.loads(payload)
assert len(DATA["queries"]) == 50 and len(DATA["r2"]) == 50 and len(DATA["r3"]) == 50
langs = {l: sum(1 for q in DATA["queries"] if q["lang"] == l) for l in ("fr", "en")}
cats = {c: sum(1 for q in DATA["queries"] if q["category"] == c)
        for c in ("entity", "event", "unanswerable")}
print("data OK:", DATA["meta"])
print("langs:", langs, "categories:", cats)
"""

MD_R2 = """\
## Section 1/2 — R2: dense vs fusion re-run (Table 7)

Exact `retrieval_benchmark.py` metric semantics: per query and per reranker,
re-encode the query and the 10 candidates (dense condition D and RRF-fusion
condition F), cosine similarity, `np.argsort(scores)[::-1]`, report
Sim@1 (`cosine@1`), mean top-5 (`mean@5`), gap top1−top2 (`drop_1to2`),
plus fusion−dense deltas with bootstrap 95% CIs (10,000 resamples, seed 42).
`rerank_time_s` = re-encoding time of one query (both conditions).
"""

CELL_R2 = """\
# === R2: 50 queries x 4 rerankers (fp16 on T4) ===
import csv, json, time, gc, os, datetime, shutil
import numpy as np, torch
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

QUERIES = DATA["queries"]
R2DATA = DATA["r2"]
MODELS = [
    ("multilingual-e5-large", "intfloat/multilingual-e5-large-instruct", 32),
    ("e5-mistral-7b-instruct", "intfloat/e5-mistral-7b-instruct", 8),
    ("SFR-Embedding-Mistral", "Salesforce/SFR-Embedding-Mistral", 8),
    ("Linq-Embed-Mistral", "Linq-AI-Research/Linq-Embed-Mistral", 8),
]
CSV_PATH = f"{BASE}/r2_results.csv"
SUMMARY_PATH = f"{BASE}/r2_summary.json"

def rescore(model, query, texts, batch_size):
    q_vec = model.encode(query, normalize_embeddings=True)
    d_vecs = model.encode(texts, normalize_embeddings=True, batch_size=batch_size)
    scores = [1 - cosine(q_vec, dv) for dv in d_vecs]
    order = np.argsort(scores)[::-1]
    return {
        "cos1": float(scores[order[0]]),
        "mean5": float(np.mean([scores[i] for i in order[:5]])),
        "drop": float(scores[order[0]] - scores[order[1]]),
    }

def bootstrap_ci(values, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

def purge_hub_cache(repo_path):
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        shutil.rmtree(os.path.join(HF_HUB_CACHE, "models--" + repo_path.replace("/", "--")),
                      ignore_errors=True)
    except Exception as e:
        print("[cache]", e)

rows = []
summary = {
    "n_queries": len(QUERIES),
    "top_k": 10,
    "k_rrf": 60,
    "encoder": "intfloat/multilingual-e5-large-instruct (local fp32 encode of candidates)",
    "device": f"{torch.cuda.get_device_name(0)} fp16 (Colab)",
    "models": {},
    "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
}

for label, path, bs in MODELS:
    print(f"[r2] reranker {label} (batch={bs})", flush=True)
    t0 = time.time()
    model = SentenceTransformer(path, device="cuda", trust_remote_code=True,
                                model_kwargs={"torch_dtype": torch.float16})
    load_time = round(time.time() - t0, 2)
    t_all = time.time()
    per_query = []
    for q in QUERIES:
        dense_texts = R2DATA[q["id"]]["dense"]
        fusion_texts = R2DATA[q["id"]]["fusion"]
        t_enc = time.time()
        d = rescore(model, q["query"], dense_texts, bs)
        fu = rescore(model, q["query"], fusion_texts, bs)
        enc_time = round(time.time() - t_enc, 4)
        row = {
            "qid": q["id"], "category": q["category"], "lang": q["lang"],
            "model": label, "precision": "fp16-cuda",
            "cosine@1_dense": round(d["cos1"], 4),
            "mean@5_dense": round(d["mean5"], 4),
            "drop_1to2_dense": round(d["drop"], 4),
            "cosine@1_fusion": round(fu["cos1"], 4),
            "mean@5_fusion": round(fu["mean5"], 4),
            "drop_1to2_fusion": round(fu["drop"], 4),
            "rerank_time_s": enc_time,
        }
        rows.append(row)
        per_query.append(row)
    rerank_total = round(time.time() - t_all, 1)

    def col(name, qr=per_query):
        return np.array([r[name] for r in qr], dtype=float)

    delta1 = col("cosine@1_fusion") - col("cosine@1_dense")
    delta5 = col("mean@5_fusion") - col("mean@5_dense")
    summary["models"][label] = {
        "precision": "fp16-cuda",
        "load_time_s": load_time,
        "rerank_time_total_s": rerank_total,
        "rerank_time_per_query_s": round(rerank_total / len(QUERIES), 3),
        "mean": {k: round(float(col(k).mean()), 4) for k in
                 ["cosine@1_dense", "mean@5_dense", "drop_1to2_dense",
                  "cosine@1_fusion", "mean@5_fusion", "drop_1to2_fusion"]},
        "delta_fusion_minus_dense": {
            "cosine@1": round(float(delta1.mean()), 4),
            "cosine@1_ci95": [round(v, 4) for v in bootstrap_ci(delta1)],
            "mean@5": round(float(delta5.mean()), 4),
            "mean@5_ci95": [round(v, 4) for v in bootstrap_ci(delta5)],
        },
    }
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[r2]   {label}: load={load_time}s rerank={rerank_total}s "
          f"cos@1 D={summary['models'][label]['mean']['cosine@1_dense']} "
          f"F={summary['models'][label]['mean']['cosine@1_fusion']}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    purge_hub_cache(path)

print("[r2] DONE ->", CSV_PATH, SUMMARY_PATH)
print(json.dumps({k: v["delta_fusion_minus_dense"] for k, v in summary["models"].items()},
                 indent=2))
"""

MD_R3 = """\
## Section 2/2 — R3: answer generation for the 50 queries (Mistral-7B)

Deployed pipeline prompt and generation parameters copied **verbatim** from
`benchmarks/generation/ragas_eval.py` (structured context builder, French
instruction prompt, chat template, `max_new_tokens=2000`, `temperature=0.3`,
`top_p=0.9`). Saves after every answer (resume-safe). RAGAS metrics
(faithfulness / answer_relevancy) are judged **locally** afterwards (DeepSeek
API judge + local Ollama embeddings).
"""

CELL_R3 = '''\
# === R3: generate answers for the 50 queries (Mistral-7B fp16) ===
import json, time, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM

MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
ANSWERS_PATH = f"{BASE}/r3_answers.json"

def build_structured_context(context_docs: list, max_chars_per_doc_chunk: int = 1000) -> str:
    """
    Build a structured context from the provided documents.
    :param context_docs: The list of context documents to be processed.
    :param max_chars_per_doc_chunk: The maximum number of characters per document chunk.
    :return:
    """
    final_context_parts = []

    processed_articles = {}

    for doc in context_docs:
        meta = doc.get('meta', {})
        doc_id_full = meta.get('docid', 'Source_inconnue_docid')
        doc_id_base = doc_id_full.split('#')[0] if isinstance(doc_id_full, str) else doc_id_full

        doc_title = meta.get('title', 'Titre inconnu')
        text_chunk = doc.get('text', '')

        if doc_id_base not in processed_articles:
            processed_articles[doc_id_base] = {'title': doc_title, 'texts': []}

        processed_articles[doc_id_base]['texts'].append(text_chunk[:max_chars_per_doc_chunk])

    for doc_id_base, article_data in processed_articles.items():
        article_title = article_data['title']
        concatenated_texts = "\\n".join(article_data['texts'])
        final_context_parts.append(
            f"Extrait de l'article \\"{article_title}\\" (Source ID: {doc_id_base}):\\n{concatenated_texts}")

    return "\\n\\n---\\n\\n".join(final_context_parts)

print("[r3] loading", MISTRAL, "fp16 ...", flush=True)
tok = AutoTokenizer.from_pretrained(MISTRAL)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
gen_model = AutoModelForCausalLM.from_pretrained(
    MISTRAL, torch_dtype=torch.float16, device_map="cuda")

answers = []
if os.path.exists(ANSWERS_PATH):
    answers = json.load(open(ANSWERS_PATH))
    print("[r3] resume:", len(answers), "answers")
done_ids = {a["id"] for a in answers}

for s in DATA["r3"]:
    if s["id"] in done_ids:
        continue
    context_text = build_structured_context(s["contexts"])
    prompt_content = f\"\"\"
        Tu es un expert en histoire et tu dois répondre à la question suivante en prenant en compte les extraits de journaux historiques fournis.
        Ta tâche est de répondre à la question posée en utilisant EXCLUSIVEMENT les informations contenues dans les "Extraits de journaux" fournis ci-dessous.
        Ne fais aucune supposition et n'utilise aucune connaissance extérieure.
        Si les extraits ne contiennent pas l'information nécessaire pour répondre à la question, tu DOIS explicitement indiquer : "Je ne peux pas répondre à cette question en me basant uniquement sur les informations fournies."
        Réponds dans la langue de la question.
        Vérifie attentivement que chaque information que tu extrais concerne bien l'événement principal de la question et non un autre événement mentionné dans le contexte, sauf si le lien de causalité est explicite.
        Si tu identifies des acteurs, assure-toi que leurs relations sont explicitement décrites dans les extraits avant de les affirmer.
        Ne fais pas référence à toi-même en tant que "modèle d'IA".
        Une conséquence est un résultat ou un effet postérieur à un événement. Un événement qui déclenche ou cause un autre événement n'est pas une conséquence de cet événement lui-même.

        Extraits de journaux :
        ---
        {context_text}
        ---
        Question : {s['question']}
    \"\"\"
    messages = [{"role": "user", "content": prompt_content.strip()}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.inference_mode():
        generated_ids = gen_model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
        )
    answer = tok.decode(generated_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True).strip()
    answers.append({
        "id": s["id"], "category": s["category"], "lang": s["lang"],
        "question": s["question"], "answer": answer,
        "contexts": [c["text"] for c in s["contexts"]],
    })
    json.dump(answers, open(ANSWERS_PATH, "w"), ensure_ascii=False, indent=2)
    print(f"[r3] {len(answers)}/50 {s['id']} ({time.time()-t0:.1f}s): {answer[:90]!r}",
          flush=True)

print("[r3] DONE ->", ANSWERS_PATH, f"({len(answers)} answers)")
'''

CELL_DOWNLOAD = """\
# Collect the three deliverables.
# Kaggle: files are in /kaggle/working -> Save Version -> Save & Run All
# (Commit), then download them from the new version's Output tab.
# Colab: browser downloads start below.
import os
FILES = [f"{BASE}/r2_results.csv", f"{BASE}/r2_summary.json", f"{BASE}/r3_answers.json"]
for p in FILES:
    e = os.path.exists(p)
    print(("OK      " if e else "MISSING "), p, os.path.getsize(p) if e else "")
if os.path.isdir("/kaggle/working"):
    print("Kaggle: commit this notebook (Save Version -> Save & Run All) and")
    print("fetch the files from the version's Output tab.")
else:
    from google.colab import files
    for p in FILES:
        if os.path.exists(p):
            files.download(p)
"""

MD_OUTRO = """\
### After the run

1. Get the three files off the runtime: on **Kaggle**, Save Version →
   Save & Run All (Commit) → open the new version → **Output** tab →
   download `r2_results.csv`, `r2_summary.json`, `r3_answers.json`; on
   **Colab**, the download cell above saves them directly. Place them at
   `benchmarks/retrieval/{r2_results.csv,r2_summary.json}` and
   `benchmarks/generation/r3_answers.json`.
2. R3 judging (faithfulness / answer_relevancy) runs **locally** afterwards:
   provide `DEEPSEEK_API_KEY` and a running Ollama with the embedding model
   (`OLLAMA_BASE_URL` defaults to `http://127.0.0.1:11434/v1`,
   `RAGAS_JUDGE_MODEL=deepseek-v4-flash`, `RAGAS_EMBED_MODEL=qwen3-embedding:4b`),
   then `.venv/bin/python ragas_eval.py` from `benchmarks/generation/` — it
   detects `r3_answers.json` and skips generation.
3. Table-7 timing provenance = T4 (§4.1.4 wording will be adapted in Stage 3
   together with the returned numbers).
"""


def code_cell(source):
    ast.parse(source)  # validate
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def to_source_lines(text):
    """nbformat accepts a plain string for source; keep string form."""
    return text


def main():
    cells = [
        md_cell(MD_INTRO),
        code_cell(CELL_SETUP),
        code_cell(CELL_DATA.replace("__B64__", B64)),
        md_cell(MD_R2),
        code_cell(CELL_R2),
        md_cell(MD_R3),
        code_cell(CELL_R3),
        code_cell(CELL_DOWNLOAD),
        md_cell(MD_OUTRO),
    ]
    nb = {
        "cells": [
            {**c, "source": to_source_lines(c["source"])} for c in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    # round-trip validation
    nb2 = json.load(open(OUT_PATH))
    n_code = sum(1 for c in nb2["cells"] if c["cell_type"] == "code")
    for c in nb2["cells"]:
        if c["cell_type"] == "code":
            ast.parse(c["source"])
    print(f"wrote {OUT_PATH}: {len(nb2['cells'])} cells ({n_code} code, all AST-OK), "
          f"{os.path.getsize(OUT_PATH)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
