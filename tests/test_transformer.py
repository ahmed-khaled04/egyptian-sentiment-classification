import numpy as np
import pytest
from transformers import BertConfig, BertModel, BertTokenizerFast

from src import predict
from src.predict import TransformerPredictor, load_predictor, model_seeds
from src.preprocess import preprocess
from src.train import run
from tests.test_tfidf_lr import WORDS, toy_df


@pytest.fixture
def tiny_bert(tmp_path):
    """A tiny random BERT + word-level vocab, saved like a hub model (no download)."""
    words = sorted({preprocess(w, mode="light") for ws in WORDS.values() for w in ws})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    path = tmp_path / "tiny-bert"
    path.mkdir()
    # transformers v5 takes the vocab directly (`vocab_file=` is silently ignored)
    BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=False).save_pretrained(path)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64)
    BertModel(cfg).save_pretrained(path)  # no classifier head, like the real pretrained checkpoints
    return path


def test_train_and_predict_tiny_bert(tmp_path, tiny_bert, monkeypatch):
    toy_df(seed=0).to_csv(tmp_path / "train.csv", index=False)
    toy_df(n_per_class=10, seed=1).to_csv(tmp_path / "val.csv", index=False)
    models_dir = tmp_path / "models"
    cfg = {
        "name": "msa_bert", "type": "transformer", "model_name": str(tiny_bert), "seeds": [1],
        "data": {"train": str(tmp_path / "train.csv"), "val": str(tmp_path / "val.csv")},
        "max_len": 16, "lr": 1e-3, "batch_size": 16, "epochs": 8, "warmup_ratio": 0.1,
        "weight_decay": 0.0, "output_dir": str(models_dir / "msa_bert"),
    }
    result = run(cfg)
    assert result["val"]["macro_f1"] > 0.8
    assert not (models_dir / "msa_bert/seed_1/checkpoints").exists()  # intermediate checkpoints cleaned

    monkeypatch.setattr(predict, "MODELS_DIR", models_dir)
    assert model_seeds("msa_bert") == [1]
    p = load_predictor("msa_bert")
    assert isinstance(p, TransformerPredictor)
    probs = p.predict_proba(["حلو جميل", "زفت وحش بايظ", "كلام جديد"])
    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-4)
    assert p.predict_labels(["حلو جميل تحفة رائع"]) == ["positive"]
