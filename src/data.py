import pandas as pd
import json
import os

DATA_DIR = "../data"
COLLECTION = os.path.join(DATA_DIR, "collection.csv")
TOPICS = os.path.join(DATA_DIR, "topics.csv")
GROUNDTRUTH = os.path.join(DATA_DIR, "groundtruth.csv")
QRELS = os.path.join(DATA_DIR, "qrels.txt")
COLLECTION_JSONL = os.path.join(DATA_DIR, "collection.jsonl")


def create_qrels(topics_filename, groundtruth_filename, output_file=QRELS):
    topics = pd.read_csv(topics_filename)
    groundtruth = pd.read_csv(groundtruth_filename)

    # Merge on topic_id
    data = pd.merge(topics, groundtruth, on="topic_id")

    lines = []
    for _, row in data.iterrows():
        line = f"{row['question_id']} 0 {row['passage_id']} {row['relevance_judgment']}"
        lines.append(line)

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print("Qrels created:", pd.DataFrame(lines).shape)


def create_pyserini_collection(collection_filename, output_file=COLLECTION_JSONL):
    collection = pd.read_csv(collection_filename)

    # Save as JSONL for Pyserini
    with open(output_file, "w") as f:
        for _, row in collection.iterrows():
            doc = {"id": row["passage_id"], "contents": row["passage"]}
            f.write(json.dumps(doc) + "\n")

    print("Collection JSONL created")


if __name__ == "__main__":
    create_qrels(TOPICS, GROUNDTRUTH)
    create_pyserini_collection(COLLECTION)