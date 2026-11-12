"""Data loading and cleaning (paper Sec. III-A and III-B).

Cleaning removes HTML tags, hyperlinks, punctuation, square-bracket text and
numeric characters, then lower-cases the text.
"""
import re
import string

import pandas as pd

_HTML = re.compile(r"<.*?>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_BRACKETS = re.compile(r"\[.*?\]")
_DIGITS = re.compile(r"\w*\d\w*")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")          # "unidentified symbols"
_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply the paper's cleaning steps to one tweet."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _BRACKETS.sub(" ", text)
    text = _DIGITS.sub(" ", text)
    text = _NON_ASCII.sub(" ", text)
    text = _PUNCT.sub(" ", text)          # also strips '#' from hashtags, keeps the word
    return _SPACES.sub(" ", text).strip()


def load_dataset(path) -> pd.DataFrame:
    """Load a Kaggle disaster-tweets CSV and add a `clean_text` column."""
    df = pd.read_csv(path)
    df["keyword"] = df["keyword"].fillna("")
    df["location"] = df["location"].fillna("")   # many tweets have no location
    df["clean_text"] = df["text"].apply(clean_text)
    return df
