import argparse
import os
import pandas as pd

from .data import load_phrasebank_75agree, split_df
from .models import train_eval
from .rag_knn import make_rag_inputs
from .utils import ensure_dir, save_json, plot_tradeoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="outputs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rag_k", type=int, default=3)
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    df = load_phrasebank_75agree(seed=args.seed)
    train_df, val_df, test_df = split_df(df, seed=args.seed)

    models = [
        "bert-base-uncased",
        "roberta-base",
        "distilbert-base-uncased",
    ]

    results = []
    for m in models:
        r = train_eval(m, train_df, val_df, test_df, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
        r["setting"] = "baseline"
        results.append(r)
        save_json(r, os.path.join(args.output_dir, f"{m.replace('/','_')}_baseline.json"))

    # RAG experiment (standout)
    rag_test_df = test_df.copy()
    rag_test_df["sentence"] = make_rag_inputs(train_df, test_df, k=args.rag_k)

    for m in models:
        r = train_eval(m, train_df, val_df, rag_test_df, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
        r["setting"] = f"rag_k{args.rag_k}"
        results.append(r)
        save_json(r, os.path.join(args.output_dir, f"{m.replace('/','_')}_ragk{args.rag_k}.json"))

    out_df = pd.DataFrame(results)
    out_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    plot_tradeoff(out_df[out_df["setting"] == "baseline"], os.path.join(args.output_dir, "tradeoff_baseline.png"),
                  "Baseline Accuracy–Latency Tradeoff")
    plot_tradeoff(out_df[out_df["setting"] != "baseline"], os.path.join(args.output_dir, "tradeoff_rag.png"),
                  "RAG (KNN) Accuracy–Latency Tradeoff")

    print("\nDONE ✅ Saved outputs/ with results.csv and plots.")


if __name__ == "__main__":
    main()
