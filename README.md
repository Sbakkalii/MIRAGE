# Multilingual RAG over Temporally Diverse Text Corpora

Code, results and manuscript source for an empirical study of multilingual
Retrieval-Augmented Generation (RAG) over temporally layered corpora — the French
and English subsets of MIRACL. The repository accompanies the paper *Multilingual
Retrieval-Augmented Generation for Temporally Diverse Text Corpora* (anonymous
submission) and is provided for artifact review and replication.

The study asks whether semantic query expansion (SQE) and multi-query fusion
improve retrieval for corpora where queries and passages are separated by lexical
and orthographic variation, and it evaluates the deployed pipeline end to end:
dense retrieval, structured generation with explicit abstention, an NER ablation
and an embedding-model ablation.

**Headline result** (reported in the paper and reproducible from `benchmarks/`):
single-query dense retrieval is the robust default under original-query
re-scoring — multi-query fusion, BM25 and a sparse+dense hybrid do not improve it —
while structured generation yields high faithfulness on well-supported queries and
refuses all unanswerable ones.

## Repository layout

| Path | Contents |
|---|---|
| `preprocessing/`, `indexation/` | corpus preprocessing, chunking and index construction |
| `retrieval/`, `reranking/`, `embedding/` | retrieval, re-ranking and embedding-selection code |
| `generation/` | generation, RAGAS evaluation and the assistant/API layer |
| `benchmarks/` | benchmark scripts, notebooks and **all released result files** |
| `notebooks/` | Colab/Kaggle notebooks used for the T4 runs |
| `tests/` | unit tests for preprocessing, indexing and the query set |
| `tools/` | small utility scripts |
| `paper/` | anonymous manuscript source (`main.tex`), style files and compiled PDF |

## Released results and where to find them

Every number in the paper can be traced to one of these files.

| Artifact | Path | Supports |
|---|---|---|
| Evaluation query set + grounding check | `benchmarks/evaluation/queries_50.csv`, `verify_queries.py` | the 50 evaluation queries (§4.1) |
| Dense vs. fusion retrieval benchmark | `benchmarks/retrieval/r2_results.csv`, `r2_summary.json` | Table 4 (§4.5.1) |
| Cached query reformulations | `benchmarks/retrieval/r2_expansions.json` | expansion/fusion conditions |
| Generated answers + RAGAS scores | `benchmarks/generation/r3_answers.json`, `r3_summary.json`, `ragas_evaluation_results.csv` | Table 5 and §4.5.2 |
| Dense replication, BM25 and sparse+dense hybrid | `benchmarks/retrieval/dense_*`, `bm25_*`, `hybrid_*` | Appendix baselines (Tables 9–10) |
| Baseline run script (incl. 4-bit loader and precision control) | `benchmarks/retrieval/run_r3_remaining.ipynb` | Appendix baselines |
| NER ablation | `benchmarks/preprocessing/NER_benchmark_results.csv` | §4.3 |
| Embedding-model ablation | `benchmarks/embedding/` | §4.4 |

Each per-query CSV carries one row per query × re-scoring model with the
`cosine@1` / `mean@5` / `drop_1to2` metrics for its conditions; the matching
`*_summary.json` holds the aggregates and the paired bootstrap confidence
intervals (10,000 resamples, seed 42) used in the paper.

## Reproducing the results

1. **Environment.** Python 3.10+ with `torch`, `transformers`,
   `sentence-transformers`, `pandas`, `pyarrow`, `numpy`, `scipy`,
   `rank_bm25`, `ragas`, `pymilvus` (only for the Milvus backend) and
   `bitsandbytes>=0.46.1` (only for the 4-bit re-scoring path).
2. **Corpus.** Download the MIRACL French and English collections, then run the
   scripts under `preprocessing/` and `indexation/` to produce the chunked corpus
   and the dense index. The derived files (a ~2.7 GB embedding matrix, the chunk
   store and the Milvus database) are **not** shipped here, both because of their
   size and because they regenerate from MIRACL; `indexation/` contains the build
   entry points.
3. **Retrieval benchmark** (paper Table 4):
   `python benchmarks/retrieval/run_r2_benchmark.py`.
4. **Generation + RAGAS** (paper Table 5): run `benchmarks/generation/ragas_eval.py`
   against a judge LLM and embedding model; the paper reports results obtained with
   `deepseek-v4-flash` (claim verification) and `qwen3-embedding:4b` (relevancy),
   hosted stand-ins for the RAGAS defaults `gpt-4o-mini` /
   `text-embedding-ada-002`. Credentials are read from the environment
   (`DEEPSEEK_API_KEY`) and are never stored in this repository.
5. **Appendix baselines** (BM25, sparse+dense hybrid, dense replication):
   `benchmarks/retrieval/run_r3_remaining.ipynb`, designed to run on a Kaggle
   notebook with a GPU and the dataset attached. The notebook builds the BM25
   index, runs the six conditions and prints the paired contrasts.

### Compute and precision

Encoding times in the embedding ablation were measured on the original cluster;
the retrieval benchmark and generation used a single NVIDIA T4 (16 GB) in fp16.
The appendix baselines were produced in one session on the same T4 class with the
selected retriever scored in fp32 and the three 7B re-scoring models in 4-bit
quantization. Because 4-bit shifts cosine similarity by roughly +0.007 to +0.009
(Sim@1) — larger than the effects being measured — every contrast in the appendix
is computed within a single session, and the notebook includes a same-session
dense replication that quantifies the shift.

## Query set

`benchmarks/evaluation/queries_50.csv` holds the 50 evaluation queries
(20 entity-focused, 20 event-oriented, 10 unanswerable; 25 French, 25 English).
They were authored by the authors following the protocol in the paper, and
`verify_queries.py` checks each answerable query against the corpus and each
unanswerable query for the absence of its (deliberately contradictory) components.

## License and contact

Released for research and artifact-review purposes. The manuscript is under
anonymous review; please cite it according to the venue's final publication.
