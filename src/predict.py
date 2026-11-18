"""Unified prediction interface used by evaluation and the Gradio app.

Every model exposes `predict_proba(texts) -> (n, 3)` with columns in LABELS order,
and `predict(text) -> {label: prob}`.
"""

from pathlib import Path

import joblib
import numpy as np

from src.data.unify import LABELS

MODELS_DIR = Path("models")


class Predictor:
    name: str
    display_name: str
    labels = list(LABELS)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_labels(self, texts: list[str]) -> list[str]:
        probs = self.predict_proba(texts)
        return [self.labels[i] for i in probs.argmax(axis=1)]

    def predict(self, text: str) -> dict[str, float]:
        probs = self.predict_proba([text])[0]
        return {label: float(p) for label, p in zip(self.labels, probs)}


class MajorityPredictor(Predictor):
    """Reference baseline: always predicts the natural majority class."""

    name = "majority"
    display_name = "Baseline: always neutral"

    def predict_proba(self, texts):
        probs = np.zeros((len(texts), len(self.labels)))
        probs[:, self.labels.index("neutral")] = 1.0
        return probs


class SklearnPredictor(Predictor):
    def __init__(self, name: str, display_name: str, path: Path):
        self.name, self.display_name = name, display_name
        self.pipeline = joblib.load(path)
        # Reorder the classifier's columns to LABELS order.
        classes = list(self.pipeline.classes_)
        self._order = [classes.index(label) for label in self.labels]

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(list(texts))[:, self._order]


# name -> (display name, artifact that must exist, loader)
REGISTRY = {
    "tfidf_lr": (
        "M1: TF-IDF + LogReg",
        MODELS_DIR / "tfidf_lr/model.joblib",
        lambda: SklearnPredictor("tfidf_lr", "M1: TF-IDF + LogReg", MODELS_DIR / "tfidf_lr/model.joblib"),
    ),
}


def available_models() -> list[str]:
    return [name for name, (_, artifact, _) in REGISTRY.items() if artifact.exists()]


def load_predictor(name: str) -> Predictor:
    if name == "majority":
        return MajorityPredictor()
    _, artifact, loader = REGISTRY[name]
    if not artifact.exists():
        raise FileNotFoundError(f"{name}: {artifact} not found, train it first")
    return loader()
