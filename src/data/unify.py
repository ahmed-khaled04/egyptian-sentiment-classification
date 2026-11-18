"""Merge the raw datasets into one table: text, label, source, dialect.

    python -m src.data.unify

- ASTD: OBJ and NEUTRAL -> neutral.
- ArSarcasm-v2: Egyptian rows only.
- Duplicates (by fully normalized text) are collapsed. If copies disagree on the
  label (ASTD and ArSarcasm re-annotate many of the same tweets) the text is dropped.
"""

import csv
from pathlib import Path

import pandas as pd

from src.preprocess import preprocess

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/all.csv")

LABELS = ("positive", "negative", "neutral")
ASTD_LABELS = {"POS": "positive", "NEG": "negative", "NEUTRAL": "neutral", "OBJ": "neutral"}
ARSARCASM_LABELS = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}


def dedup_key(text: str) -> str:
    return preprocess(text, mode="full")


def load_astd(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "astd/Tweets.txt", sep="\t", header=None,
                     names=["text", "label"], quoting=csv.QUOTE_NONE, dtype=str)
    df["label"] = df["label"].str.strip().map(ASTD_LABELS)
    return df.assign(source="astd", dialect="egypt")[["text", "label", "source", "dialect"]]


def load_arsarcasm(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    df = pd.concat(
        [pd.read_csv(raw_dir / f"arsarcasm/{f}", dtype=str) for f in ("training_data.csv", "testing_data.csv")],
        ignore_index=True,
    )
    df = df[df["dialect"] == "egypt"]
    df = df.assign(
        text=df["tweet"].str.strip().str.strip('"'),  # tweets are wrapped in extra quotes
        label=df["sentiment"].map(ARSARCASM_LABELS),
        source="arsarcasm",
    )
    return df[["text", "label", "source", "dialect"]]


def unify(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Concatenate, clean and deduplicate. Returns (table, stats)."""
    df = pd.concat(frames, ignore_index=True)
    stats = {"raw_rows": len(df)}

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].str.strip()
    df["key"] = df["text"].map(dedup_key)
    df = df[df["key"].str.len() > 0]
    stats["after_dropna_empty"] = len(df)

    n_labels = df.groupby("key")["label"].transform("nunique")
    conflicts = df[n_labels > 1]
    stats["conflicting_texts_dropped"] = conflicts["key"].nunique()
    df = df[n_labels == 1]

    sources = df.groupby("key")["source"].agg(lambda s: "|".join(sorted(set(s))))
    df = df.drop_duplicates("key").copy()
    df["source"] = df["key"].map(sources)
    stats["duplicate_rows_collapsed"] = stats["after_dropna_empty"] - len(conflicts) - len(df)
    stats["final_rows"] = len(df)
    stats["label_counts"] = df["label"].value_counts().to_dict()

    return df[["text", "label", "source", "dialect"]].reset_index(drop=True), stats


def main() -> None:
    df, stats = unify([load_astd(), load_arsarcasm()])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
