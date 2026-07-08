"""
preprocess.py – Reusable text-cleaning utilities for VeriNews AI.

All functions are pure (no side effects) so they can be used
independently in training, evaluation, and the API inference path.

Milestone 1B adds: remove_publisher_datelines() to strip wire-service
attribution strings that cause publisher-leakage in TF-IDF features.
"""
import re
import string
from typing import Optional

import pandas as pd


# ── Publisher dateline patterns (Milestone 1B) ────────────────────────────────
# These patterns match wire-service attributions like:
#   "WASHINGTON (Reuters) - "  |  "(Reuters) -"  |  "Reuters:"
# They are the #1 source of domain bias (reuters coeff = +21.7).
_DATELINE_PATTERNS = [
    # City + wire agency in parens + dash  e.g. "WASHINGTON (Reuters) - "
    # Works on both original case and lowercased text
    r"^[a-zA-Z][a-zA-Z\s,\.]{0,40}\([a-zA-Z\s]+\)\s*[-\u2013\u2014]\s*",
    # Plain wire agency + dash e.g. "(Reuters) - " or "(AP) - "
    r"^\([a-zA-Z\s/]+\)\s*[-\u2013\u2014]\s*",
    # "reuters -" "ap -" at start (lowercased)
    r"^(reuters|ap|afp|upi|pti|ani|ians|ndtv|bbc|cnn|fox news)\s*[:\-\u2013\u2014]\s*",
    # "By FIRSTNAME LASTNAME, Reuters" bylines (lowercased: "by ...")
    r"^by\s+[a-z]+\s+[a-z]+[,\s]+[a-z]+\s*",
]
_DATELINE_RE = re.compile("|".join(_DATELINE_PATTERNS), re.IGNORECASE)



def remove_publisher_datelines(text: str) -> str:
    """
    Strip wire-service datelines from the START of an article.

    Examples removed:
      "WASHINGTON (Reuters) - The president said..."
      "(AP) - Authorities confirmed..."
      "Reuters: Oil prices rose..."

    This is the primary fix for publisher-leakage bias identified in
    Milestone 1.5, where 'reuters' had a coefficient of +21.7.
    """
    return _DATELINE_RE.sub("", text, count=1).strip()


# ── Individual cleaning helpers ───────────────────────────────────────────────

def to_lowercase(text: str) -> str:
    """Convert all characters to lowercase."""
    return text.lower()


def remove_html_tags(text: str) -> str:
    """Strip HTML/XML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs and bare www. addresses."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    return text


def remove_punctuation(text: str) -> str:
    """Remove all punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def remove_numbers(text: str) -> str:
    """Remove standalone numeric tokens (keeps alphanumeric words intact)."""
    return re.sub(r"\b\d+\b", " ", text)


# ── Composite pipeline ────────────────────────────────────────────────────────

def clean_text(text: Optional[str]) -> str:
    """
    Full cleaning pipeline applied in order:
      1. Handle None / NaN
      2. Remove publisher datelines  ← NEW (Milestone 1B)
      3. Lowercase
      4. Remove HTML tags
      5. Remove URLs
      6. Remove punctuation
      7. Remove standalone numbers
      8. Normalize whitespace

    Parameters
    ----------
    text : str or None
        Raw article text.

    Returns
    -------
    str
        Cleaned text string (may be empty string if input was null).
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = remove_publisher_datelines(text)   # Step 2 — strip before lowercasing
    text = to_lowercase(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = normalize_whitespace(text)
    return text


# ── DataFrame-level helpers ───────────────────────────────────────────────────

def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply the cleaning pipeline to a DataFrame column and handle missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at minimum a `text_col` column.
    text_col : str
        Name of the column holding raw article text.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with:
        - `text_col` replaced by cleaned text.
        - Rows with empty post-cleaning text removed.
        - Duplicates removed (based on `text_col`).
        - Reset integer index.
    """
    df = df.copy()

    # Fill NaN with empty string before cleaning
    df[text_col] = df[text_col].fillna("")

    # Apply cleaning
    df[text_col] = df[text_col].apply(clean_text)

    # Drop rows where text became empty after cleaning
    df = df[df[text_col].str.strip() != ""].copy()

    # Drop exact duplicates on the text column
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)

    return df
