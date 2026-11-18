"""Build a compact embedding matrix for M2/M3 from fastText Arabic (cc.ar.300).

    python -m src.embeddings [--bin embeddings/raw/cc.ar.300.bin] [--top-n 100000]

Vocabulary = every token in the train split + the top-N most frequent fastText
words, all passed through our own `preprocess` so they match model inputs.
Train tokens fastText has never seen still get a vector from its character
n-grams (subword info), which matters with ~20% OOV on test. Only the train
split is read, so val/test vocabulary never influences the model.

Output: embeddings/fasttext_ar_300.npz with `tokens` and `vectors` (float32).
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import NUM_TOKEN, URL_TOKEN, USER_TOKEN, preprocess, tokenize
from src.vocab import PAD, UNK

DEFAULT_BIN = Path("embeddings/raw/cc.ar.300.bin")  # decompressed: gensim short-reads the .gz
DEFAULT_OUT = Path("embeddings/fasttext_ar_300.npz")
TRAIN_CSV = Path("data/processed/train.csv")


def train_token_counts(train_csv: Path = TRAIN_CSV) -> Counter:
    texts = pd.read_csv(train_csv)["text"]
    return Counter(t for text in texts for t in tokenize(preprocess(text)))


def build_matrix(kv, train_counts: Counter, top_n: int, seed: int = 0) -> tuple[list[str], np.ndarray]:
    """`kv` is a gensim FastTextKeyedVectors (anything with index_to_key and __getitem__)."""
    rng = np.random.default_rng(seed)
    dim = kv.vector_size
    tokens = [PAD, UNK, URL_TOKEN, USER_TOKEN, NUM_TOKEN]
    vectors = [np.zeros(dim), np.zeros(dim)] + [rng.normal(0, 0.1, dim) for _ in range(3)]
    seen = set(tokens)

    # Train tokens first (most frequent first), vectors from fastText incl. subwords.
    for tok, _ in train_counts.most_common():
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)
            vectors.append(kv[tok])

    # Then frequent fastText words, normalized; first (most frequent) raw form wins.
    added = 0
    for raw in kv.index_to_key:
        if added >= top_n:
            break
        for tok in tokenize(preprocess(raw)):
            if tok not in seen:
                seen.add(tok)
                tokens.append(tok)
                vectors.append(kv[raw] if tok == raw else kv[tok])
                added += 1

    return tokens, np.asarray(vectors, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--top-n", type=int, default=100_000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    from gensim.models.fasttext import load_facebook_vectors

    print(f"loading {args.bin} (several minutes, ~5 GB RAM)...")
    kv = load_facebook_vectors(str(args.bin))
    counts = train_token_counts()
    tokens, vectors = build_matrix(kv, counts, args.top_n)

    in_ft = sum(1 for t in counts if t in kv.key_to_index)
    print(f"train types: {len(counts):,} | exact fastText hits: {in_ft / len(counts):.1%} (rest use subwords)")
    print(f"vocab: {len(tokens):,} x {vectors.shape[1]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, tokens=np.array(tokens, dtype=object), vectors=vectors)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
