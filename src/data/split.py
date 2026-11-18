"""Stratified 80/10/10 split. All splits keep the natural label distribution;
class imbalance is handled with class-weighted losses (capping neutral in train
cost M1 ~3-4 macro-F1 points, so it was dropped).

    python -m src.data.split [--seed 42]

Asserts that no normalized text is shared between splits or with the hand-made
custom test set.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.unify import dedup_key

PROCESSED_DIR = Path("data/processed")
CUSTOM_TEST_DIR = Path("data/custom_test")


def split(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    strat = df["label"] + "_" + df["source"]
    train, rest = train_test_split(df, test_size=0.2, stratify=strat, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=strat.loc[rest.index], random_state=seed)
    return {"train": train, "val": val, "test": test}


def assert_no_overlap(named: dict[str, pd.DataFrame]) -> None:
    keys = {name: set(df["text"].map(dedup_key)) for name, df in named.items()}
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = keys[a] & keys[b]
            assert not shared, f"{len(shared)} texts shared between {a} and {b}, e.g. {next(iter(shared))!r}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(PROCESSED_DIR / "all.csv")
    splits = split(df, args.seed)
    stats = {k: v["label"].value_counts().to_dict() for k, v in splits.items()}

    check = dict(splits)
    for f in sorted(CUSTOM_TEST_DIR.glob("*.csv")):
        check[f"custom:{f.name}"] = pd.read_csv(f)
    assert_no_overlap(check)

    for name, part in splits.items():
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    (PROCESSED_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
