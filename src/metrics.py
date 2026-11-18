"""Classification metrics shared by training (val selection) and evaluation."""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.data.unify import LABELS


def compute_metrics(y_true, y_pred, labels=LABELS) -> dict:
    labels = list(labels)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": len(y_true),
        "macro_f1": float(f.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
    for i, label in enumerate(labels):
        out[f"precision_{label}"] = float(p[i])
        out[f"recall_{label}"] = float(r[i])
        out[f"f1_{label}"] = float(f[i])
        out[f"support_{label}"] = int(support[i])
    return out
