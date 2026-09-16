"""Create a reproducible, class-balanced random sample from a large parquet
dataset (text + label columns, 1=malicious / 0=benign) for evaluation.

Same dedupe + balanced-sample + shuffle logic as evaluate_parquet.load_and_sample,
just pulled out so the sample is a frozen file that every backend (Groq, Colab,
local) evaluates the *identical* rows.

    python make_sample.py <source.parquet> --size 3000 --seed 42 --out eval_sample.parquet
"""
import argparse
from pathlib import Path

import pandas as pd


def make_sample(src: Path, size: int, seed: int, out: Path) -> None:
    df = pd.read_parquet(src)
    before = len(df)
    df = df.drop_duplicates(subset="text")
    df = df[df["text"].str.strip() != ""]
    deduped = len(df)

    per_class = size // 2
    parts = []
    for label in (0, 1):
        subset = df[df["label"] == label]
        n = min(per_class, len(subset))
        parts.append(subset.sample(n=n, random_state=seed))
    sampled = (
        pd.concat(parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)[["text", "label"]]
    )
    sampled.to_parquet(out, index=False)

    print(f"source rows:   {before}")
    print(f"after dedupe:  {deduped}")
    print(f"sampled:       {len(sampled)}  "
          f"({(sampled.label == 1).sum()} malicious / {(sampled.label == 0).sum()} benign)")
    print(f"text length:   mean {int(sampled.text.str.len().mean())}, max {sampled.text.str.len().max()}")
    print(f"written:       {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path)
    p.add_argument("--size", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("eval_sample.parquet"))
    args = p.parse_args()
    make_sample(args.source, args.size, args.seed, args.out)


if __name__ == "__main__":
    main()
