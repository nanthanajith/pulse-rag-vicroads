from ranx import compare, Qrels, Run
import pandas as pd
import numpy as np
import argparse, sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topic_set', choices=['known', 'inferred'])
    parser.add_argument('qrel')
    parser.add_argument('runs', nargs='+')
    args = parser.parse_args()

    # --- Load and sanitize qrels ---
    qrels_df = pd.read_csv(
        args.qrel,
        sep=r"\s+",
        engine="python",
        names=['q_id', 'zero', 'doc_id', 'score'],
        header=None,
        comment='#',
        dtype={'q_id': str, 'zero': str, 'doc_id': str}
    )
    # Ensure correct dtypes for ranx
    qrels_df['doc_id'] = qrels_df['doc_id'].astype(str)
    qrels_df['q_id'] = qrels_df['q_id'].astype(str)
    qrels_df['score'] = pd.to_numeric(qrels_df['score'], errors='raise').astype('int64')
    
    # --- Load runs (TREC format) ---
    for run_path in args.runs:
        if not Path(run_path).exists():
            raise FileNotFoundError(f"Run file not found: {run_path}")
    runs = [Run.from_file(run, kind="trec") for run in args.runs]

    # --- Build Qrels ---
    qrels = Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")

    # --- Compare with rank-based metrics ---
    # Report classic NDCG in tandem with full and shallow-cutoff MRR.
    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=[
            "ndcg@1","ndcg@3","ndcg@5",
            "mrr",
            "mrr@3",
            "mrr@5"
        ],
        max_p=0.01,          # significance level for Tukey HSD
        make_comparable=True,
        stat_test="tukey",
        rounding_digits=4
    )

    print(report)
    print("\n========================")
    print("% Add in preamble")
    print("\\usepackage{graphicx}")
    print("\\usepackage{booktabs}")
    print("========================\n")
    print(report.to_latex())

if __name__ == "__main__":
    sys.exit(main())