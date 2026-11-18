"""Arabic text preprocessing shared by training, evaluation and the demo.

Two modes:
- "full":  aggressive normalization for the classical models (M1-M3).
- "light": minimal cleanup for pretrained transformers (M4, M5), whose
           tokenizers expect near-raw text.
"""

import re

URL_TOKEN = "<URL>"
USER_TOKEN = "<USER>"
NUM_TOKEN = "<NUM>"

_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = re.compile(r"\u0640")
_ALEF = re.compile(r"[إأآٱ]")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
# Standalone Western / Arabic-Indic / Extended Arabic-Indic numbers. Digits glued to
# letters are left alone so Arabizi words like "7elw" survive.
_DIGITS = r"[0-9\u0660-\u0669\u06F0-\u06F9]"
_NUMBER = re.compile(rf"(?<![^\W\d_]){_DIGITS}+(?:[.,\u066B]{_DIGITS}+)*(?![^\W\d_])")
# 3+ repeats of any non-digit char -> 2 (keeps "حلوو" distinct from "حلو"; leaves numbers intact).
_ELONGATION = re.compile(r"(\D)\1{2,}")
# Invisible joiners / emoji variation selectors that only fragment tokens.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufe0e\ufe0f]")
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"<URL>|<USER>|<NUM>|\w+|[^\w\s]")


def _common(text: str) -> str:
    text = _INVISIBLE.sub("", text)
    text = _URL.sub(f" {URL_TOKEN} ", text)
    text = _MENTION.sub(f" {USER_TOKEN} ", text)
    return text


def preprocess(text: str, mode: str = "full", normalize_ta_marbuta: bool = True) -> str:
    """Clean a single text. `mode` is "full" or "light"."""
    if mode not in ("full", "light"):
        raise ValueError(f"unknown mode: {mode!r}")
    if not isinstance(text, str):
        return ""

    if mode == "full":
        text = text.lower()  # before placeholders so <URL>/<USER>/<NUM> stay uppercase
    text = _common(text)

    if mode == "full":
        text = _HASHTAG.sub(lambda m: " " + m.group(1).replace("_", " ") + " ", text)
        text = _TASHKEEL.sub("", text)
        text = _TATWEEL.sub("", text)
        text = _ALEF.sub("ا", text)
        text = text.replace("ى", "ي")
        if normalize_ta_marbuta:
            text = text.replace("ة", "ه")
        text = _NUMBER.sub(f" {NUM_TOKEN} ", text)
    else:
        text = _TATWEEL.sub("", text)

    text = _ELONGATION.sub(r"\1\1", text)
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split preprocessed text into words, placeholder tokens, punctuation and emojis."""
    return _TOKEN.findall(text)
