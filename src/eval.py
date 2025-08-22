from ranx import compare, Qrels, Run
import pandas as pd
import argparse, sys
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topic_set', choices=['known', 'inferred'])
    parser.add_argument('qrel')
    parser.add_argument('runs', nargs='+')
    args = parser.parse_args()

    qrels_df = pd.read_csv(args.qrel, sep=' ', names=['q_id','ignore','doc_id','score'], header=None)

    # Convert IDs to string for Ranx
    qrels_df['doc_id'] = qrels_df['doc_id'].astype(str)
    qrels_df['q_id'] = qrels_df['q_id'].astype(str)

    # Convert score to int for Ranx
    qrels_df['score'] = qrels_df['score'].astype(np.int64)

    runs = [Run.from_file(run, kind="trec") for run in args.runs]

    qrels = Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")

    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=["ndcg@1","ndcg@3","ndcg@5"],
        max_p=0.01,
        make_comparable=True,
        stat_test="tukey",
        rounding_digits=4
    )

    print(report)
    print(report.to_latex())

if __name__ == "__main__":
    sys.exit(main())