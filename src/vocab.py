"""Token <-> id mapping for the embedding models (M2, M3)."""

from pathlib import Path

import numpy as np

from src.preprocess import preprocess, tokenize

PAD, UNK = "<PAD>", "<UNK>"


class Vocab:
    def __init__(self, tokens: list[str]):
        assert tokens[0] == PAD and tokens[1] == UNK
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.unk_id = self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize(preprocess(text))]
        return (ids[:max_len] if max_len else ids) or [self.unk_id]


def load_embeddings(path: str | Path) -> tuple[Vocab, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return Vocab(data["tokens"].tolist()), data["vectors"]
