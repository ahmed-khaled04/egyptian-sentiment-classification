"""Unified prediction interface used by evaluation and the Gradio app.

Every model exposes `predict_proba(texts) -> (n, 3)` with columns in LABELS order,
and `predict(text) -> {label: prob}`.

Neural models are trained with several seeds. `load_predictor(name)` returns the
seed with the best val macro-F1 (used by the app); evaluation iterates over all
seeds via `model_seeds(name)` to report mean ± std.
"""

import json
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


class NeuralPredictor(Predictor):
    def __init__(self, name: str, display_name: str, path: Path):
        from src.models.neural import load_checkpoint  # torch import only when needed

        self.name, self.display_name = name, display_name
        self.model, self.vocab, self.max_len = load_checkpoint(path)

    def predict_proba(self, texts):
        from src.models.neural import predict_proba

        return predict_proba(self.model, self.vocab, texts, self.max_len, device="cpu")


# name -> (display name, kind)
REGISTRY = {
    "tfidf_lr": ("M1: TF-IDF + LogReg", "sklearn"),
    "emb_avg": ("M2: Embedding avg + Linear", "neural"),
    "emb_lstm": ("M3: Embedding + BiLSTM", "neural"),
}
ARTIFACT = {"sklearn": "model.joblib", "neural": "model.pt"}
LOADERS = {"sklearn": SklearnPredictor, "neural": NeuralPredictor}


def _artifact(name: str, seed: int | None) -> Path:
    kind = REGISTRY[name][1]
    base = MODELS_DIR / name if seed is None else MODELS_DIR / name / f"seed_{seed}"
    return base / ARTIFACT[kind]


def model_seeds(name: str) -> list[int | None]:
    """Seeds with a saved checkpoint; [None] for single-run models."""
    if name == "majority" or _artifact(name, None).exists():
        return [None]
    return sorted(int(p.parent.name.split("_")[1]) for p in (MODELS_DIR / name).glob(f"seed_*/{ARTIFACT[REGISTRY[name][1]]}"))


def best_seed(name: str) -> int | None:
    result = MODELS_DIR / name / "train_result.json"
    if result.exists():
        return json.loads(result.read_text()).get("best_seed")
    return None


def available_models() -> list[str]:
    return [name for name in REGISTRY if (MODELS_DIR / name).exists() and model_seeds(name)]


def load_predictor(name: str, seed: int | None = None) -> Predictor:
    if name == "majority":
        return MajorityPredictor()
    display, kind = REGISTRY[name]
    seeds = model_seeds(name)
    if not seeds:
        raise FileNotFoundError(f"{name}: no checkpoint under {MODELS_DIR / name}, train it first")
    if seed is None and seeds != [None]:
        seed = best_seed(name) if best_seed(name) in seeds else seeds[0]
    return LOADERS[kind](name, display, _artifact(name, seed))
