import pandas as pd
import pytest

from src.data.split import assert_no_overlap, split
from src.data.unify import unify


def frame(rows, source="astd"):
    return pd.DataFrame(
        [{"text": t, "label": l, "source": source, "dialect": "egypt"} for t, l in rows]
    )


class TestUnify:
    def test_agreeing_duplicates_collapsed_with_both_sources(self):
        df, stats = unify([frame([("الفيلم حلو", "positive")]),
                           frame([("الفيلم  حلو", "positive")], source="arsarcasm")])
        assert len(df) == 1
        assert df.loc[0, "source"] == "arsarcasm|astd"
        assert stats["duplicate_rows_collapsed"] == 1

    def test_conflicting_duplicates_dropped(self):
        df, stats = unify([frame([("الخدمة زفت", "negative"), ("تمام", "neutral")]),
                           frame([("الخدمة زفت", "neutral")], source="arsarcasm")])
        assert df["text"].tolist() == ["تمام"]
        assert stats["conflicting_texts_dropped"] == 1

    def test_normalization_variants_count_as_duplicates(self):
        df, _ = unify([frame([("حلووووو", "positive"), ("حلوو", "positive")])])
        assert len(df) == 1

    def test_empty_and_missing_dropped(self):
        df, _ = unify([frame([("   ", "neutral"), (None, "neutral"), ("كويس", None), ("كويس جدا", "positive")])])
        assert df["text"].tolist() == ["كويس جدا"]


def make_dataset(n_pos=40, n_neg=60, n_neu=300):
    rows = ([(f"pos {i}", "positive") for i in range(n_pos)]
            + [(f"neg {i}", "negative") for i in range(n_neg)]
            + [(f"neu {i}", "neutral") for i in range(n_neu)])
    # unique word suffixes so dedup keys differ (digits become <NUM> otherwise)
    rows = [(f"{t} {'ا' * (i % 2 + 1)}{chr(0x0628 + i % 20)}{chr(0x0628 + i // 20 % 20)}", l)
            for i, (t, l) in enumerate(rows)]
    return frame(rows)


class TestSplit:
    def test_proportions_and_disjoint(self):
        df = make_dataset()
        s = split(df, seed=0)
        assert len(s["train"]) == 320 and len(s["val"]) == 40 and len(s["test"]) == 40
        idx = [set(p.index) for p in s.values()]
        assert not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2])

    def test_stratified(self):
        s = split(make_dataset(), seed=0)
        assert (s["test"]["label"] == "neutral").sum() == 30

    def test_assert_no_overlap_detects_normalized_leak(self):
        a = frame([("الخدمة حلوة", "positive")])
        b = frame([("الخدمه حلوه", "positive")])
        with pytest.raises(AssertionError):
            assert_no_overlap({"train": a, "custom": b})
