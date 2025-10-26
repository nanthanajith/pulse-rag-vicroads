# Pulse RAG – VicRoads Chatbot and RAG Evaluation

A Retrieval-Augmented Generation (RAG) project for VicRoads. It combines dense and BM25 retrieval over a small curated collection with a local LLM (via Ollama) to answer user questions. It also includes an evaluation pipeline that produces TREC-style run files and reports classic ranking metrics.

Key pieces:
- Retrieval using Pyserini: dense (FAISS over ColBERT embeddings) and BM25 (`src/search.py`).
- Data preparation and qrels creation (`src/data.py`).
- Evaluation using Ranx (NDCG and MRR) from TREC runs (`src/eval.py`).
- Local generation via Ollama (default model `llama3`).
- Streamlit chatbot UI with simple multi-thread conversations (`src/app.py`).

## Repository Structure
- `src/app.py` – Streamlit app for the VicRoads chatbot UI.
- `src/chatbot.py` – Minimal text-mode chatbot using the same retrieval/generation.
- `src/search.py` – Retrieval utilities, TREC run generation, and optional BM25 index build.
- `src/data.py` – Creates `data/qrels.txt` and `data/collection.jsonl` from CSVs.
- `src/eval.py` – Compares run files with qrels using Ranx; prints metrics and LaTeX table.
- `src/encode.sh` / `src/index.sh` – Scripts to encode collection and build a FAISS index.
- `src/index-bm25.sh` – Script to build a Lucene BM25 index from JSONL.
- `src/requirements.txt` – Python dependencies.
- `data/` – Toy dataset: `collection.csv`, `topics.csv`, `groundtruth.csv`, plus derived files.
- `target/` – Outputs: embeddings, indexes, and TREC run files.

## Prerequisites
- Python 3.10–3.12 (tested with 3.11).
- Java (for Pyserini’s Lucene/BM25; Java 17 or 21 recommended). The code sets safe Lucene JVM flags internally for newer JDKs.
- Ollama installed locally with a model available (default: `llama3`).


## Setup
1) Create and activate a virtual environment, then install dependencies:
   - `python -m venv .venv`
   - Windows PowerShell: ``.\.venv\Scripts\Activate``
   - macOS/Linux: `source .venv/bin/activate`
   - `pip install -r requirements.txt`

2) Ensure Java is available (`java -version`). If using Java 21, the repository already applies a Lucene JVM flag at runtime to avoid memory segment warnings.

3) Prepare Ollama and pull a model (default `llama3`):
   - Install Ollama: https://ollama.com
   - `ollama pull llama3`
   - Optional: set a custom endpoint via `OLLAMA_HOST` (default is `http://127.0.0.1:11434`).

## Data
The project includes a small toy dataset under `data/`:
- `collection.csv` with columns `passage_id,passage` (documents)
- `topics.csv` with columns `topic_id,Topic,question_id,question` (queries)
- `groundtruth.csv` with columns `topic_id,topic,passage_id,passage,relevance_judgment`

From these, we derive:
- `data/qrels.txt` – Relevance judgments in TREC qrels format
- `data/collection.jsonl` – JSONL collection for Pyserini’s BM25 indexer

Generate both with:
- `python src/data.py`

## Indexing and Retrieval
RAG retrieval supports two modes: dense and BM25.

Dense (FAISS over ColBERT embeddings):
- Encode collection embeddings: run `src/encode.sh` (requires Bash) or use the Python module directly:
  - Bash: `bash src/encode.sh`
  - Or Python equivalent (single-shard CPU example):
    - `python -m pyserini.encode input --corpus data/collection.jsonl --fields text --shard-id 0 --shard-num 1 output --embeddings target/embeddings/tct_colbert-v2-hnp-msmarco --to-faiss encoder --encoder castorini/tct_colbert-v2-hnp-msmarco --fields text --batch 32 --device cpu`
- Build FAISS index:
  - Bash: `bash src/index.sh`
  - Or Python equivalent: `python -m pyserini.index.faiss --input target/embeddings/tct_colbert-v2-hnp-msmarco --output target/indexes/tct_colbert-v2-hnp-msmarco-faiss --hnsw`

BM25 (Lucene):
- If missing, the Streamlit/chat pipelines can auto-create the Lucene index from `data/collection.jsonl` when BM25 mode is used.
- You can also build it explicitly:
  - Bash: `bash src/index-bm25.sh`
  - Or Python equivalent: `python -m pyserini.index.lucene --collection JsonCollection --input data/collection --language en --index target/indexes/bm25 --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw`

Generate TREC run files for evaluation (both modes):
- `python src/search.py`
- Outputs:
  - Dense: `target/runs/vicroads-dense.txt`
  - BM25: `target/runs/vicroads-bm25.txt`

## Evaluation
Compare runs against qrels using Ranx (prints NDCG and MRR variants and emits a LaTeX table):
- Ensure qrels are present: `python src/data.py`
- Example (inferred topics):
  - `python src/eval.py inferred data/qrels.txt target/runs/vicroads-dense.txt target/runs/vicroads-bm25.txt`

## Chatbot
Two options are available.

Streamlit UI:
- `streamlit run src/app.py`
- Defaults: model `llama3`, mode `dense`, top-k 3.
- Environment:
  - `OLLAMA_HOST` can override the default Ollama endpoint.
- Notes:
  - To use BM25 retrieval, change `DEFAULT_MODE` in `src/app.py`.
  - To show retrieved context passages in the UI, set `SHOW_CONTEXT = True` in `src/app.py`.
  - The header uses `src/Logo.png` as the page icon and banner.

CLI (text-only):
- `python src/chatbot.py`
- Type your question; type `exit` to quit.


## Configuration
Within `src/app.py`:
- `DEFAULT_MODEL` – Ollama model name (default: `llama3`).
- `DEFAULT_MODE` – Retrieval mode: `dense` or `bm25`.
- `DEFAULT_TOPK` – Number of passages to retrieve.
- `SHOW_CONTEXT` – Toggle display of retrieved passages.
- `LOGO_PATH` – Logo used for the Streamlit page.

Within `src/search.py`:
- Paths for collection, topics, and indexes can be adjusted at the top of the file.
- BM25 index is auto-created if missing when BM25 mode is requested.

## Troubleshooting
- Java/Lucene warnings with Java 21: mitigated via `JDK_JAVA_OPTIONS` flags set programmatically in `src/search.py`.
- Missing BM25 index: it will be created automatically on first BM25 search or you can run `src/index-bm25.sh`.
- No LLM response: verify Ollama is running and the model is pulled; set `OLLAMA_HOST` if running remotely.
- Windows shell scripts: use Git Bash/WSL, or run the Python command equivalents listed above.

## Acknowledgements
- Pyserini (Lucene/BM25 and dense retrieval), FAISS
- Castorini encoders (`tct_colbert-v2-hnp-msmarco`)
- Ranx (ranking evaluation)
- Streamlit (UI)
- Ollama (local LLM inference)


## Attribution
- This project reproduces and adapts ideas and components from Walert, a conversational information seeking chatbot presented at CHIIR 2024. Please cite the following work when referencing this repository:
- Pathiyan Cherumanal, S., Tian, L., Abushaqra, F. M., Magnossão de Paula, A. F., Ji, K., Ali, H., Hettiachchi, D., Trippas, J. R., Scholer, F., and Spina, D. (2024). Walert: Putting Conversational Information Seeking Knowledge into Action by Building and Evaluating a Large Language Model-Powered Chatbot. In Proceedings of the 2024 Conference on Human Information Interaction and Retrieval (CHIIR '24). ACM. https://doi.org/10.1145/3627508.3638309

### Citation
```
@inproceedings{10.1145/3627508.3638309,
  author = {Pathiyan Cherumanal, Sachin and Tian, Lin and Abushaqra, Futoon M. and Magnoss\~{a}o de Paula, Angel Felipe and Ji, Kaixin and Ali, Halil and Hettiachchi, Danula and Trippas, Johanne R. and Scholer, Falk and Spina, Damiano},
  title = {Walert: Putting Conversational Information Seeking Knowledge into Action by Building and Evaluating a Large Language Model-Powered Chatbot},
  year = {2024},
  isbn = {9798400704345},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3627508.3638309},
  doi = {10.1145/3627508.3638309},
  booktitle = {Proceedings of the 2024 Conference on Human Information Interaction and Retrieval},
  pages = {401--405},
  numpages = {5},
  keywords = {conversational information seeking, large language models, retrieval-augmented generation},
  location = {Sheffield, United Kingdom},
  series = {CHIIR '24}
}
```
