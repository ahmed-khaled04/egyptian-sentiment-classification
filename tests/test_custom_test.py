import pandas as pd
import pytest

from src.data.custom_test import build


def sheet(rows):
    cols = ["id", "text", "category", "domain", "label_a1", "label_a2", "label"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


ROWS = [
    ("e1", "الفيلم ده جامد فشخ", "egy_vocab", "movies_tv", "positive", "positive", ""),
    ("e2", "مش وحش خالص", "negation", "food", "positive", "neutral", ""),          # disagreement
    ("e3", "تحفة بجد الطلب وصل بعد ٣ ساعات", "sarcasm", "food", "positive", "negative", "negative"),  # adjudicated
    ("e4", "الماتش الساعة ٩", "neutral_factual", "football", "neutral", "neutral", ""),
    ("e5", "الخدمة زفت", "general", "services", "negative", "", ""),               # only one annotator
]


def test_final_labels_agreement_and_pending():
    egy, _, report = build(sheet(ROWS))
    assert dict(zip(egy["id"], egy["label"])) == {"e1": "positive", "e3": "negative", "e4": "neutral"}
    assert report["pending_adjudication"] == ["e2"]
    assert report["double_annotated"] == 4
    assert report["agreement"] == pytest.approx(0.5)
    assert -1 <= report["cohen_kappa"] <= 1


def test_labels_and_categories_are_case_insensitive():
    egy, _, _ = build(sheet([("x", "حلو", "General", "food", "Positive", "POSITIVE", "")]))
    assert egy.loc[0, "label"] == "positive" and egy.loc[0, "category"] == "general"


@pytest.mark.parametrize("bad_row, msg", [
    (("x", "حلو", "slang", "food", "positive", "positive", ""), "unknown category"),
    (("x", "حلو", "general", "food", "good", "positive", ""), "invalid label_a1"),
])
def test_invalid_values_rejected(bad_row, msg):
    with pytest.raises(ValueError, match=msg):
        build(sheet([bad_row]))


def test_duplicate_ids_and_normalized_texts_rejected():
    with pytest.raises(ValueError, match="duplicate ids"):
        build(sheet([ROWS[0], ROWS[0][:1] + ("نص تاني",) + ROWS[0][2:]]))
    with pytest.raises(ValueError, match="duplicate texts"):
        # differs only by a diacritic
        build(sheet([ROWS[0], ("e9", "الفيلمَ ده جامد فشخ", *ROWS[0][2:])]))


def test_msa_parallel_copies_egy_label_and_skips_unlabeled():
    msa_sheet = pd.DataFrame({"id": ["e1", "e2"], "text_msa": ["هذا الفيلم رائع جداً", "ليس سيئاً على الإطلاق"]})
    _, msa, report = build(sheet(ROWS), msa_sheet)
    assert msa[["id", "text", "label"]].values.tolist() == [["e1", "هذا الفيلم رائع جداً", "positive"]]
    assert report["msa_waiting_for_egy_label"] == ["e2"]


def test_msa_unknown_id_rejected():
    with pytest.raises(ValueError, match="not in annotation_sheet"):
        build(sheet(ROWS), pd.DataFrame({"id": ["zzz"], "text_msa": ["نص"]}))
