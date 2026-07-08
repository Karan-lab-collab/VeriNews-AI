"""
features.py – TF-IDF feature engineering for VeriNews AI.

Provides helpers to fit, transform, save, and load the vectorizer
so that the exact same vocabulary is used during training and inference.
"""
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from train.config import (
    MAX_FEATURES,
    MIN_DF,
    NGRAM_RANGE,
    VECTORIZER_PATH,
)


# ── Publisher stop words (Milestone 1B) ──────────────────────────────────────
# These tokens encode WHICH publisher wrote the article, not WHAT the article says.
# Feature analysis showed: reuters coeff = +21.7, washington reuters = +9.1.
# Adding them to stop_words forces the model to learn content-based features.
_PUBLISHER_STOP_WORDS = [
    # Wire agencies — appear in almost all real articles in the ISOT training set
    "reuters", "associated press", "washington reuters", "york reuters",
    "moscow reuters", "london reuters", "beijing reuters", "paris reuters",
    "reporting", "written by", "editing by", "compiled by",
    # Dateline cities that appear with "(Reuters)" — after dateline removal,
    # the city name often remains and co-occurs with Reuters articles.
    # Note: we keep common city names (they appear in both real and fake news).
    # US publication signals
    "nytimes", "washingtonpost", "foxnews", "huffpost", "breitbart",
    # Byline tokens common in real news corpus
    "reuters staff", "bureau",
]


def build_vectorizer() -> TfidfVectorizer:
    """
    Instantiate a TF-IDF vectorizer with project-wide defaults.

    Milestone 1B: Added _PUBLISHER_STOP_WORDS to eliminate source-identity
    leakage (reuters coeff was +21.7 — the #1 REAL predictor).

    Returns
    -------
    TfidfVectorizer
        An unfitted vectorizer ready to call `.fit_transform()`.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    combined_stop = list(ENGLISH_STOP_WORDS) + _PUBLISHER_STOP_WORDS

    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=True,      # Replace raw TF with 1 + log(tf) — helps with long docs
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",  # Only alphabetic tokens ≥ 2 chars
        stop_words=combined_stop,
    )


def fit_vectorizer(
    vectorizer: TfidfVectorizer, texts: "list[str]"
) -> np.ndarray:
    """
    Fit the vectorizer on training texts and transform them.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Unfitted vectorizer.
    texts : list[str]
        Cleaned training documents.

    Returns
    -------
    np.ndarray (sparse)
        TF-IDF feature matrix for the training set.
    """
    return vectorizer.fit_transform(texts)


def transform(vectorizer: TfidfVectorizer, texts: "list[str]") -> np.ndarray:
    """
    Transform texts using an already-fitted vectorizer.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
        Previously fitted vectorizer.
    texts : list[str]
        Documents to transform (test set or new inference samples).

    Returns
    -------
    np.ndarray (sparse)
        TF-IDF feature matrix.
    """
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path = VECTORIZER_PATH) -> None:
    """Persist the fitted vectorizer to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"[features] Vectorizer saved → {path}")


def load_vectorizer(path: Path = VECTORIZER_PATH) -> TfidfVectorizer:
    """
    Load a previously saved TF-IDF vectorizer.

    Parameters
    ----------
    path : Path
        File path to the joblib-serialised vectorizer.

    Returns
    -------
    TfidfVectorizer
        The fitted vectorizer.

    Raises
    ------
    FileNotFoundError
        If the vectorizer file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at '{path}'. "
            "Run 'python train/train.py' first to train and save the model."
        )
    return joblib.load(path)
