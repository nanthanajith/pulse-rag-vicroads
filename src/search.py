# search.py - VicRoads RAG retrieval (dense + BM25, for chatbot & eval)
import os
import pandas as pd
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexer
from pyserini.output_writer import OutputFormat, get_output_writer

# -------------------------
# Paths & constants
# -------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data")
COLLECTION = os.path.join(DATA_DIR, "collection.csv")
COLLECTION_JSONL = os.path.join(DATA_DIR, "collection.jsonl")
TOPICS = os.path.join(DATA_DIR, "topics.csv")

INDEX_DENSE = os.path.join(BASE_DIR, "../target/indexes/tct_colbert-v2-hnp-msmarco-faiss")
QUERY_ENCODER = 'facebook/dpr-question_encoder-multiset-base'

INDEX_BM25 = os.path.join(BASE_DIR, "../target/indexes/bm25")

OUTPUT_DIR = os.path.join(BASE_DIR, "../target/runs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH_DENSE = os.path.join(OUTPUT_DIR, "vicroads-dense.txt")
OUTPUT_PATH_BM25 = os.path.join(OUTPUT_DIR, "vicroads-bm25.txt")

TOP_K = 3  # Number of passages to retrieve for chatbot

# Load passage collection once
collection_df = pd.read_csv(COLLECTION)
topics_df = pd.read_csv(TOPICS)

# -------------------------
# Initialize searcher
# -------------------------
def init_searcher(mode="dense"):
    if mode == "dense":
        return FaissSearcher(INDEX_DENSE, QUERY_ENCODER)
    elif mode == "bm25":
        if not os.path.exists(INDEX_BM25):
            print("BM25 index not found! Creating it now...")
            create_bm25_index()
        return LuceneSearcher(INDEX_BM25)
    else:
        raise ValueError("Mode must be 'dense' or 'bm25'")

# -------------------------
# Create BM25 Lucene index (if missing)
# -------------------------
def create_bm25_index():
    from pyserini.index.lucene import IndexCollection
    if not os.path.exists(COLLECTION_JSONL):
        # Create JSONL from CSV
        import json
        with open(COLLECTION_JSONL, "w", encoding="utf-8") as f:
            for _, row in collection_df.iterrows():
                doc = {"id": row["passage_id"], "contents": row["passage"]}
                f.write(json.dumps(doc) + "\n")
        print("collection.jsonl created for BM25")

    # Create Lucene index
    print("Creating BM25 Lucene index...")
    indexer = LuceneIndexer(INDEX_BM25)
    indexer.index(COLLECTION_JSONL)
    print(f"BM25 index created at {INDEX_BM25}")

# -------------------------
# Get top-K passages for a question (chatbot)
# -------------------------
def get_context_passages(question, mode="dense", top_k=TOP_K):
    searcher = init_searcher(mode)
    hits = searcher.search(question, top_k)
    context_passages = []
    for d in hits:
        temp_passage = collection_df.loc[collection_df['passage_id'] == d.docid, 'passage'].values
        if len(temp_passage) > 0:
            context_passages.append(temp_passage[0])
    return context_passages

# -------------------------
# Generate TREC-style run file (for evaluation)
# -------------------------
def write_run_file(mode="dense", num_hits=100):
    searcher = init_searcher(mode)
    output_file = OUTPUT_PATH_DENSE if mode == "dense" else OUTPUT_PATH_BM25
    tag = f"vicroads-{mode}"

    output_writer = get_output_writer(
        output_file,
        OutputFormat('trec'),
        'w',
        max_hits=num_hits,
        tag=tag,
        topics=topics_df
    )

    with output_writer:
        for question_id, question in topics_df[['question_id', 'question']].values:
            hits = searcher.search(question, num_hits)
            output_writer.write(question_id, hits)

    print(f"Run file created: {output_file}")

# -------------------------
# Command-line execution
# -------------------------
if __name__ == "__main__":
    print("Generating TREC run files...")
    write_run_file("dense")
    write_run_file("bm25")
