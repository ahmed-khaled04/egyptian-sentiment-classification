import pytest

from src.preprocess import NUM_TOKEN, URL_TOKEN, USER_TOKEN, preprocess, tokenize


class TestFull:
    def test_elongation_reduced_to_two(self):
        assert preprocess("حلووووو اوي") == "حلوو اوي"

    def test_numbers_not_shortened_by_elongation(self):
        assert preprocess("1000 جنيه", mode="light") == "1000 جنيه"

    def test_alef_variants(self):
        assert preprocess("أحمد إبراهيم آخر") == "احمد ابراهيم اخر"

    def test_alef_maqsura_and_ta_marbuta(self):
        assert preprocess("على الخدمة") == "علي الخدمه"

    def test_ta_marbuta_optional(self):
        assert preprocess("الخدمة", normalize_ta_marbuta=False) == "الخدمة"

    def test_tashkeel_and_tatweel_removed(self):
        assert preprocess("جَمِيـــل") == "جميل"

    def test_url_mention_number_placeholders(self):
        out = preprocess("@ali_99 شوف https://t.co/abc الساعة ١٠")
        assert out == f"{USER_TOKEN} شوف {URL_TOKEN} الساعه {NUM_TOKEN}"

    def test_hashtag_keeps_text(self):
        assert preprocess("#الأهلي_بطل") == "الاهلي بطل"

    def test_emojis_kept(self):
        assert "😂" in preprocess("تحفة 😂😂😂😂")
        assert preprocess("تحفة 😂😂😂😂").endswith("😂😂")

    def test_arabizi_lowercased(self):
        assert preprocess("El Service 7ELW") == "el service 7elw"

    def test_non_string_returns_empty(self):
        assert preprocess(None) == ""


class TestLight:
    def test_keeps_arabic_forms(self):
        assert preprocess("أحمد على الخدمة", mode="light") == "أحمد على الخدمة"

    def test_replaces_url_and_mention_reduces_elongation(self):
        out = preprocess("@user حلوووو https://x.com", mode="light")
        assert out == f"{USER_TOKEN} حلوو {URL_TOKEN}"

    def test_bad_mode(self):
        with pytest.raises(ValueError):
            preprocess("x", mode="heavy")


def test_tokenize_splits_words_punct_emoji_placeholders():
    toks = tokenize(preprocess("مش وحش، بجد 👏 @x"))
    assert toks == ["مش", "وحش", "،", "بجد", "👏", USER_TOKEN]
