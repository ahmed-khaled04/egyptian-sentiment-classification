"""Evaluate every trained model on every available test set.

    python -m src.evaluate [--models tfidf_lr ...]

Writes:
  results/summary.csv                     one row per (model, test set)
  results/per_category.csv                accuracy per category (custom test set only)
  results/metrics.json                    full metrics incl. confusion matrices
  results/predictions/<model>__<set>.csv  per-example predictions (for significance tests)
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.unify import LABELS
from src.metrics import compute_metrics
from src.predict import available_models, load_predictor

RESULTS_DIR = Path("results")
TEST_SETS = {
    "in_domain": Path("data/processed/test.csv"),
    "custom_egy": Path("data/custom_test/egy_test.csv"),
    "msa_parallel": Path("data/custom_test/msa_parallel.csv"),
}


def load_test_sets() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path) for name, path in TEST_SETS.items() if path.exists()}


def evaluate(model_names: list[str], test_sets: dict[str, pd.DataFrame], results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    summary, per_category, full = [], [], {}

    for model_name in model_names:
        predictor = load_predictor(model_name)
        for set_name, df in test_sets.items():
            probs = predictor.predict_proba(df["text"].tolist())
            preds = [LABELS[i] for i in probs.argmax(axis=1)]
            metrics = compute_metrics(df["label"], preds)
            full[f"{model_name}__{set_name}"] = metrics

            out = df.copy()
            out["pred"] = preds
            for i, label in enumerate(LABELS):
                out[f"p_{label}"] = probs[:, i]
            out.to_csv(pred_dir / f"{model_name}__{set_name}.csv", index=False)

            summary.append({
                "model": model_name, "test_set": set_name, "n": metrics["n"],
                "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                **{f"f1_{label}": metrics[f"f1_{label}"] for label in LABELS},
            })
            if "category" in out.columns:
                acc = (out["pred"] == out["label"]).groupby(out["category"]).agg(["mean", "size"])
                for cat, row in acc.iterrows():
                    per_category.append({"model": model_name, "test_set": set_name, "category": cat,
                                         "accuracy": row["mean"], "n": int(row["size"])})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    if per_category:
        pd.DataFrame(per_category).to_csv(results_dir / "per_category.csv", index=False)
    (results_dir / "metrics.json").write_text(json.dumps(full, indent=2))
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="default: majority baseline + all trained models")
    args = parser.parse_args()
    models = args.models or ["majority", *available_models()]
    test_sets = load_test_sets()
    print(f"models: {models}\ntest sets: {list(test_sets)}")
    summary = evaluate(models, test_sets)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
