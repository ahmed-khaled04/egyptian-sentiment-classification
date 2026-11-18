"""M1: word + char TF-IDF features with Logistic Regression.

The saved pipeline takes raw text: `preprocess` runs inside the vectorizers, so
evaluation and the demo cannot drift from training.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.metrics import compute_metrics
from src.preprocess import preprocess, tokenize

MODEL_FILE = "model.joblib"


def build_pipeline(cfg: dict, C: float) -> Pipeline:
    feats = cfg["features"]
    word = TfidfVectorizer(
        preprocessor=preprocess, tokenizer=tokenize, token_pattern=None, lowercase=False,
        ngram_range=tuple(feats["word_ngram"]), min_df=feats["min_df"], sublinear_tf=True,
    )
    char = TfidfVectorizer(
        preprocessor=preprocess, analyzer="char_wb", lowercase=False,
        ngram_range=tuple(feats["char_ngram"]), min_df=feats["min_df"], sublinear_tf=True,
    )
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=cfg["seed"])
    return Pipeline([("features", FeatureUnion([("word", word), ("char", char)])), ("clf", clf)])


def train(cfg: dict) -> dict:
    train_df = pd.read_csv(cfg["data"]["train"])
    val_df = pd.read_csv(cfg["data"]["val"])

    best = None
    search = []
    for C in cfg["model"]["C_grid"]:
        pipe = build_pipeline(cfg, C).fit(train_df["text"], train_df["label"])
        metrics = compute_metrics(val_df["label"], pipe.predict(val_df["text"]))
        search.append({"C": C, "val_macro_f1": metrics["macro_f1"]})
        print(f"C={C:<6} val macro-F1={metrics['macro_f1']:.4f}")
        if best is None or metrics["macro_f1"] > best[1]["macro_f1"]:
            best = (pipe, metrics, C)

    pipe, metrics, C = best
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / MODEL_FILE)
    return {"best_C": C, "search": search, "val": metrics}
