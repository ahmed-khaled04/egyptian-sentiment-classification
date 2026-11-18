from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch

from src import predict
from src.embeddings import build_matrix
from src.models.neural import EmbAvg, EmbLSTM, encode_batch
from src.predict import NeuralPredictor, load_predictor, model_seeds
from src.preprocess import preprocess
from src.train import run
from src.vocab import PAD, UNK, Vocab, load_embeddings
from tests.test_tfidf_lr import WORDS, toy_df


class FakeKV:
    """Stands in for gensim FastTextKeyedVectors: any string gets a deterministic vector."""

    vector_size = 8

    def __init__(self, words):
        self.index_to_key = list(words)
        self.key_to_index = {w: i for i, w in enumerate(words)}

    def __getitem__(self, word):
        return np.random.default_rng(abs(hash(word)) % 2**32).normal(size=self.vector_size)


def test_build_matrix_train_tokens_then_normalized_fasttext_words():
    kv = FakeKV(["الخدمة", "أحمد", "حلو"])
    tokens, vectors = build_matrix(kv, Counter({"حلو": 3, "زفت": 1}), top_n=10)
    assert tokens[:2] == [PAD, UNK]
    assert tokens[5:7] == ["حلو", "زفت"]           # train tokens, by frequency
    assert "الخدمه" in tokens and "احمد" in tokens   # fastText words normalized
    assert "الخدمة" not in tokens
    assert len(tokens) == len(set(tokens)) == vectors.shape[0]
    assert not vectors[0].any()                     # PAD is zero


def test_vocab_encode_normalizes_and_handles_unknown():
    v = Vocab([PAD, UNK, "الخدمه", "حلوه"])
    assert v.encode("الخدمة حلوة جدا") == [2, 3, 1]
    assert v.encode("", max_len=5) == [1]
    assert v.encode("الخدمة حلوة", max_len=1) == [2]


@pytest.mark.parametrize("cls", [EmbAvg, EmbLSTM])
def test_models_forward_shapes(cls):
    vectors = torch.randn(10, 8)
    model = cls(vectors, n_classes=3, freeze=True, dropout=0.0, hidden=4, layers=2)
    ids = torch.tensor([[2, 3, 4], [5, 0, 0]])
    assert model(ids, torch.tensor([3, 1])).shape == (2, 3)
    assert not model.emb.weight.requires_grad


def test_avg_ignores_padding():
    vectors = torch.randn(10, 8)
    model = EmbAvg(vectors, 3, freeze=True, dropout=0.0).eval()
    a = model(torch.tensor([[2, 3]]), torch.tensor([2]))
    b = model(torch.tensor([[2, 3, 0, 0]]), torch.tensor([2]))
    torch.testing.assert_close(a, b)


@pytest.fixture
def toy_embeddings(tmp_path):
    words = sorted({preprocess(w) for ws in WORDS.values() for w in ws})
    vocab = Vocab([PAD, UNK] + words)
    rng = np.random.default_rng(0)
    path = tmp_path / "emb.npz"
    np.savez(path, tokens=np.array(vocab.itos, dtype=object), vectors=rng.normal(size=(len(vocab), 16)).astype(np.float32))
    return path


@pytest.mark.parametrize("arch", ["avg", "lstm"])
def test_train_seeds_and_predictor(tmp_path, toy_embeddings, monkeypatch, arch):
    toy_df(seed=0).to_csv(tmp_path / "train.csv", index=False)
    toy_df(n_per_class=10, seed=1).to_csv(tmp_path / "val.csv", index=False)
    models_dir = tmp_path / "models"
    cfg = {
        "name": "emb_" + arch, "type": "neural", "arch": arch, "seeds": [1, 2],
        "data": {"train": str(tmp_path / "train.csv"), "val": str(tmp_path / "val.csv")},
        "embeddings": str(toy_embeddings), "freeze_embeddings": False, "hidden": 8, "layers": 1,
        "max_len": 16, "dropout": 0.0, "batch_size": 16, "lr": 1e-2, "epochs": 15, "patience": 15,
        "device": "cpu", "output_dir": str(models_dir / ("emb_" + arch)),
    }
    result = run(cfg)
    assert result["val"]["macro_f1"] > 0.9
    assert result["best_seed"] in (1, 2) and len(result["runs"]) == 2

    monkeypatch.setattr(predict, "MODELS_DIR", models_dir)
    name = "emb_" + arch
    assert model_seeds(name) == [1, 2]
    p = load_predictor(name)
    assert isinstance(p, NeuralPredictor)
    probs = p.predict_proba(["حلو جميل تحفة", "زفت وحش", "كلمة_جديدة"])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    assert p.predict_labels(["حلو جميل تحفة رائع"]) == ["positive"]


def test_encode_batch_pads_with_zero():
    v = Vocab([PAD, UNK, "حلو"])
    ids, lengths = encode_batch(v, ["حلو حلو", "حلو"], max_len=10)
    assert ids.tolist() == [[2, 2], [2, 0]] and lengths.tolist() == [2, 1]
