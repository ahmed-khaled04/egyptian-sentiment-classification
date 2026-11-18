"""M2 (embedding mean + linear) and M3 (embedding + BiLSTM), sharing one trainer.

Each seed in `cfg["seeds"]` is trained separately and saved to
<output_dir>/seed_<s>/model.pt with its vocab and architecture, so a checkpoint
is self-contained for evaluation and the demo.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.data.unify import LABELS
from src.metrics import compute_metrics
from src.vocab import Vocab, load_embeddings

MODEL_FILE = "model.pt"


class EmbAvg(nn.Module):
    def __init__(self, vectors: torch.Tensor, n_classes: int, freeze: bool, dropout: float, **_):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(vectors, freeze=freeze, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(vectors.shape[1], n_classes)

    def forward(self, ids, lengths):
        mask = (ids != 0).unsqueeze(-1).float()
        mean = (self.emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.out(self.dropout(mean))


class EmbLSTM(nn.Module):
    def __init__(self, vectors: torch.Tensor, n_classes: int, freeze: bool, dropout: float,
                 hidden: int = 128, layers: int = 2, **_):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(vectors, freeze=freeze, padding_idx=0)
        self.lstm = nn.LSTM(vectors.shape[1], hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.dropout(self.emb(ids)), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        final = torch.cat([h[-2], h[-1]], dim=1)  # last layer, forward + backward
        return self.out(self.dropout(final))


ARCHS = {"avg": EmbAvg, "lstm": EmbLSTM}


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def encode_batch(vocab: Vocab, texts, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [vocab.encode(t, max_len) for t in texts]
    lengths = torch.tensor([len(s) for s in seqs])
    ids = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, lengths


@torch.no_grad()
def predict_proba(model: nn.Module, vocab: Vocab, texts, max_len: int, device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    texts = list(texts)
    for i in range(0, len(texts), batch_size):
        ids, lengths = encode_batch(vocab, texts[i : i + batch_size], max_len)
        out.append(torch.softmax(model(ids.to(device), lengths), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


def class_weights(labels: pd.Series) -> torch.Tensor:
    counts = labels.value_counts()
    return torch.tensor([len(labels) / (len(LABELS) * counts.get(l, 1)) for l in LABELS], dtype=torch.float)


def build_model(arch_cfg: dict, vectors: torch.Tensor) -> nn.Module:
    return ARCHS[arch_cfg["arch"]](vectors, len(LABELS), **arch_cfg)


def train_one_seed(cfg: dict, seed: int, vocab: Vocab, vectors: np.ndarray, train_df, val_df) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device(cfg.get("device", "auto"))
    arch_cfg = {"arch": cfg["arch"], "freeze": cfg["freeze_embeddings"], "dropout": cfg["dropout"],
                "hidden": cfg.get("hidden", 128), "layers": cfg.get("layers", 2)}
    model = build_model(arch_cfg, torch.tensor(vectors)).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_df["label"]).to(device))

    texts = train_df["text"].tolist()
    y = torch.tensor([LABELS.index(l) for l in train_df["label"]])
    max_len, bs = cfg["max_len"], cfg["batch_size"]
    rng = np.random.default_rng(seed)

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = rng.permutation(len(texts))
        total = 0.0
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            ids, lengths = encode_batch(vocab, [texts[j] for j in idx], max_len)
            opt.zero_grad()
            loss = loss_fn(model(ids.to(device), lengths), y[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)

        probs = predict_proba(model, vocab, val_df["text"], max_len, device)
        f1 = compute_metrics(val_df["label"], [LABELS[i] for i in probs.argmax(1)])["macro_f1"]
        print(f"  seed {seed} epoch {epoch:2d} loss={total / len(texts):.4f} val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    probs = predict_proba(model, vocab, val_df["text"], max_len, device)
    metrics = compute_metrics(val_df["label"], [LABELS[i] for i in probs.argmax(1)])

    out_dir = Path(cfg["output_dir"]) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "arch_cfg": arch_cfg, "tokens": vocab.itos,
                "dim": vectors.shape[1], "max_len": max_len}, out_dir / MODEL_FILE)
    return {"seed": seed, "best_epoch": best_epoch, "val": metrics}


def train(cfg: dict) -> dict:
    vocab, vectors = load_embeddings(cfg["embeddings"])
    train_df = pd.read_csv(cfg["data"]["train"])
    val_df = pd.read_csv(cfg["data"]["val"])
    runs = [train_one_seed(cfg, s, vocab, vectors, train_df, val_df) for s in cfg["seeds"]]
    f1s = [r["val"]["macro_f1"] for r in runs]
    best = max(runs, key=lambda r: r["val"]["macro_f1"])
    return {"runs": runs, "best_seed": best["seed"], "val": best["val"],
            "val_macro_f1_mean": float(np.mean(f1s)), "val_macro_f1_std": float(np.std(f1s))}


def load_checkpoint(path: str | Path, device="cpu") -> tuple[nn.Module, Vocab, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocab(ckpt["tokens"])
    model = build_model(ckpt["arch_cfg"], torch.zeros(len(vocab), ckpt["dim"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), vocab, ckpt["max_len"]
