import numpy as np
import pandas as pd
import pytest

from src.data.unify import LABELS
from src.evaluate import evaluate
from src.metrics import compute_metrics
from src.predict import MajorityPredictor, SklearnPredictor
from src.train import run

WORDS = {
    "positive": ["حلو", "جميل", "تحفة", "جامد", "رائع"],
    "negative": ["وحش", "زفت", "بايظ", "فاشل", "مقرف"],
    "neutral": ["الساعة", "المحل", "بكرة", "الشارع", "الاجتماع"],
}


def toy_df(n_per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for label, words in WORDS.items():
        for _ in range(n_per_class):
            rows.append({"text": " ".join(rng.choice(words, size=4)), "label": label})
    return pd.DataFrame(rows)


@pytest.fixture
def trained(tmp_path):
    toy_df(seed=0).to_csv(tmp_path / "train.csv", index=False)
    toy_df(n_per_class=10, seed=1).to_csv(tmp_path / "val.csv", index=False)
    cfg = {
        "name": "tfidf_lr", "type": "tfidf_lr", "seed": 0,
        "data": {"train": str(tmp_path / "train.csv"), "val": str(tmp_path / "val.csv")},
        "features": {"word_ngram": [1, 2], "char_ngram": [2, 4], "min_df": 1},
        "model": {"C_grid": [1.0, 10.0]},
        "output_dir": str(tmp_path / "model"),
    }
    result = run(cfg)
    return tmp_path, result


def test_training_learns_toy_task(trained):
    _, result = trained
    assert result["val"]["macro_f1"] > 0.9
    assert result["best_C"] in (1.0, 10.0)


def test_predictor_proba_in_label_order(trained):
    tmp_path, _ = trained
    pred = SklearnPredictor("tfidf_lr", "M1", tmp_path / "model/model.joblib")
    probs = pred.predict_proba(["الفيلم حلو جميل", "الخدمة زفت"])
    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert pred.predict_labels(["حلو جميل تحفة", "زفت وحش بايظ"]) == ["positive", "negative"]
    assert set(pred.predict("حلو")) == set(LABELS)


def test_evaluate_writes_outputs_and_per_category(tmp_path):
    test = pd.DataFrame({"text": ["a", "b", "c"], "label": ["neutral", "positive", "neutral"],
                         "category": ["x", "x", "y"]})
    summary = evaluate(["majority"], {"custom_egy": test}, results_dir=tmp_path)
    assert summary.loc[0, "accuracy"] == pytest.approx(2 / 3)
    assert (tmp_path / "predictions/majority__custom_egy.csv").exists()
    per_cat = pd.read_csv(tmp_path / "per_category.csv")
    assert dict(zip(per_cat.category, per_cat.accuracy)) == {"x": 0.5, "y": 1.0}


def test_majority_and_metrics():
    preds = MajorityPredictor().predict_labels(["x", "y"])
    m = compute_metrics(["neutral", "positive"], preds)
    assert m["accuracy"] == 0.5
    assert m["f1_neutral"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx((2 / 3) / 3)
