"""Validate the hand-made Egyptian test set and build the evaluation files.

    python -m src.data.custom_test

Reads  data/custom_test/annotation_sheet.csv  (id, text, category, domain, label_a1, label_a2, label, notes)
       data/custom_test/msa_sheet.csv         (id, text_msa)
Writes data/custom_test/egy_test.csv          (id, text, label, category, domain)
       data/custom_test/msa_parallel.csv      (id, text, label, category, domain)

Final label = `label` if filled (adjudicated), else label_a1 when both annotators agree.
Rows still in disagreement are reported and left out until adjudicated.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.data.split import assert_no_overlap
from src.data.unify import LABELS, dedup_key

CUSTOM_DIR = Path("data/custom_test")
PROCESSED_DIR = Path("data/processed")
CATEGORIES = ("egy_vocab", "negation", "sarcasm", "intensifier", "emoji", "arabizi", "neutral_factual", "general")
SHEET_COLUMNS = ["id", "text", "category", "domain", "label_a1", "label_a2", "label", "notes"]


def _clean(col: pd.Series) -> pd.Series:
    return col.fillna("").astype(str).str.strip().str.lower()


def build(sheet: pd.DataFrame, msa_sheet: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (egy_test, msa_parallel, report). Raises ValueError on invalid input."""
    sheet = sheet.copy()
    sheet["text"] = sheet["text"].fillna("").astype(str).str.strip()
    sheet = sheet[sheet["text"] != ""]
    for c in ("label_a1", "label_a2", "label", "category", "domain"):
        sheet[c] = _clean(sheet.get(c, pd.Series(index=sheet.index, dtype=str)))

    errors = []
    if sheet["id"].duplicated().any():
        errors.append(f"duplicate ids: {sheet.loc[sheet['id'].duplicated(), 'id'].tolist()}")
    bad_cat = sheet[~sheet["category"].isin(CATEGORIES)]
    if len(bad_cat):
        errors.append(f"unknown category in ids {bad_cat['id'].tolist()} (allowed: {CATEGORIES})")
    for c in ("label_a1", "label_a2", "label"):
        bad = sheet[(sheet[c] != "") & ~sheet[c].isin(LABELS)]
        if len(bad):
            errors.append(f"invalid {c} in ids {bad['id'].tolist()} (allowed: {LABELS})")
    keys = sheet["text"].map(dedup_key)
    if keys.duplicated().any():
        errors.append(f"duplicate texts (after normalization) in ids {sheet.loc[keys.duplicated(), 'id'].tolist()}")
    if errors:
        raise ValueError("annotation_sheet.csv:\n  " + "\n  ".join(errors))

    both = sheet[(sheet["label_a1"] != "") & (sheet["label_a2"] != "")]
    report = {"rows": len(sheet), "double_annotated": len(both)}
    if len(both):
        report["agreement"] = float((both["label_a1"] == both["label_a2"]).mean())
        report["cohen_kappa"] = float(cohen_kappa_score(both["label_a1"], both["label_a2"], labels=list(LABELS)))

    agreed = (sheet["label_a1"] != "") & (sheet["label_a1"] == sheet["label_a2"])
    sheet["final"] = sheet["label"].where(sheet["label"] != "", sheet["label_a1"].where(agreed, ""))
    missing = sheet["final"] == ""
    one_side = (sheet["label_a1"] == "") | (sheet["label_a2"] == "")
    report["pending_adjudication"] = sheet.loc[missing & ~one_side, "id"].tolist()
    report["not_fully_labeled"] = sheet.loc[missing & one_side, "id"].tolist()

    egy = sheet.loc[~missing].assign(label=lambda d: d["final"])[["id", "text", "label", "category", "domain"]]
    egy = egy.reset_index(drop=True)
    report["final_rows"] = len(egy)
    report["by_category"] = egy.groupby(["category", "label"]).size().unstack(fill_value=0).to_dict("index") if len(egy) else {}

    msa = pd.DataFrame(columns=["id", "text", "label", "category", "domain"])
    if msa_sheet is not None and len(msa_sheet):
        m = msa_sheet.assign(text_msa=msa_sheet["text_msa"].fillna("").astype(str).str.strip())
        m = m[m["text_msa"] != ""]
        unknown = set(m["id"]) - set(sheet["id"])
        if unknown:
            raise ValueError(f"msa_sheet.csv: ids not in annotation_sheet.csv: {sorted(unknown)}")
        msa = (m.merge(egy, on="id", how="inner")
               .drop(columns="text").rename(columns={"text_msa": "text"})[["id", "text", "label", "category", "domain"]])
        report["msa_rows"] = len(msa)
        report["msa_waiting_for_egy_label"] = sorted(set(m["id"]) - set(msa["id"]))
    return egy, msa, report


def main() -> None:
    sheet = pd.read_csv(CUSTOM_DIR / "annotation_sheet.csv", dtype=str)
    msa_path = CUSTOM_DIR / "msa_sheet.csv"
    msa_sheet = pd.read_csv(msa_path, dtype=str) if msa_path.exists() else None
    egy, msa, report = build(sheet, msa_sheet)

    named = {s: pd.read_csv(PROCESSED_DIR / f"{s}.csv") for s in ("train", "val", "test")}
    named["custom_egy"] = egy
    assert_no_overlap(named)

    egy.to_csv(CUSTOM_DIR / "egy_test.csv", index=False)
    msa.to_csv(CUSTOM_DIR / "msa_parallel.csv", index=False)
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"wrote {CUSTOM_DIR / 'egy_test.csv'} ({len(egy)}) and {CUSTOM_DIR / 'msa_parallel.csv'} ({len(msa)})")


if __name__ == "__main__":
    main()
