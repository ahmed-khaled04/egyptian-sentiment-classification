"""M4 (MSA BERT) and M5 (dialect BERT): fine-tuned pretrained transformers.

Both use the same code and hyperparameters; only `model_name` differs, so the
comparison isolates the effect of pretraining data. Text gets the *light*
preprocessing (the pretrained tokenizers expect near-raw Arabic).

Each seed is saved to <output_dir>/seed_<s>/ as a regular HF model directory.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from src.data.unify import LABELS
from src.metrics import compute_metrics
from src.models.neural import class_weights, pick_device
from src.preprocess import preprocess

ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {l: i for i, l in ID2LABEL.items()}


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def to_dataset(df: pd.DataFrame, tokenizer, max_len: int) -> Dataset:
    ds = Dataset.from_dict({
        "text": [preprocess(t, mode="light") for t in df["text"]],
        "labels": [LABEL2ID[l] for l in df["label"]],
    })
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_len),
                  batched=True, remove_columns=["text"])


def _hf_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    m = compute_metrics([LABELS[i] for i in labels], [LABELS[i] for i in preds])
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}


def train_one_seed(cfg: dict, seed: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    out_dir = Path(cfg["output_dir"]) / f"seed_{seed}"
    ckpt_dir = out_dir / "checkpoints"
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        seed=seed,
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        num_train_epochs=cfg["epochs"],
        warmup_steps=cfg["warmup_ratio"],  # float < 1 = ratio of total steps
        weight_decay=cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        dataloader_pin_memory=False,
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=to_dataset(train_df, tokenizer, cfg["max_len"]),
        eval_dataset=to_dataset(val_df, tokenizer, cfg["max_len"]),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_hf_metrics,
        class_weights=class_weights(train_df["label"]),
    )
    trainer.train()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    history = [h for h in trainer.state.log_history if "eval_macro_f1" in h]
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    probs = predict_proba(trainer.model, tokenizer, val_df["text"], cfg["max_len"], trainer.model.device)
    metrics = compute_metrics(val_df["label"], [LABELS[i] for i in probs.argmax(1)])
    for h in history:
        print(f"  seed {seed} epoch {h['epoch']:.0f} val macro-F1={h['eval_macro_f1']:.4f}")
    return {"seed": seed, "history": history, "val": metrics}


def train(cfg: dict) -> dict:
    train_df = pd.read_csv(cfg["data"]["train"])
    val_df = pd.read_csv(cfg["data"]["val"])
    runs = [train_one_seed(cfg, s, train_df, val_df) for s in cfg["seeds"]]
    f1s = [r["val"]["macro_f1"] for r in runs]
    best = max(runs, key=lambda r: r["val"]["macro_f1"])
    return {"model_name": cfg["model_name"], "runs": runs, "best_seed": best["seed"], "val": best["val"],
            "val_macro_f1_mean": float(np.mean(f1s)), "val_macro_f1_std": float(np.std(f1s))}


@torch.no_grad()
def predict_proba(model, tokenizer, texts, max_len: int, device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    texts = [preprocess(t, mode="light") for t in texts]
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i : i + batch_size], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits, dim=-1).float().cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, len(LABELS)))
    # Reorder columns to LABELS order using the model's own id2label.
    order = [[model.config.id2label[i] for i in range(len(LABELS))].index(l) for l in LABELS]
    return probs[:, order]


def load_model(path: str | Path, device: str = "auto"):
    dev = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(dev).eval()
    return model, tokenizer, dev
