"""
train.py – Main training entry point for VeriNews AI baseline model.

Usage (from backend/):
    python train/train.py

Steps performed:
  1. Download dataset (if not already present).
  2. Load & label Fake.csv / True.csv.
  3. Preprocess text.
  4. TF-IDF vectorization.
  5. Train Logistic Regression.
  6. Evaluate on the held-out test set.
  7. Save model and vectorizer.
  8. Save processed dataset for future use.
"""
import sys
import time
from pathlib import Path

# Allow running as a top-level script from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 stdout so Unicode characters print correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from train.config import (
    COMBINED_CSV,
    FAKE_CSV,
    LABEL_FAKE,
    LABEL_REAL,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    TRUE_CSV,
    VECTORIZER_PATH,
)
from train.download_dataset import download
from train.evaluate import run_full_evaluation
from train.features import build_vectorizer, fit_vectorizer, save_vectorizer, transform
from train.preprocess import preprocess_dataframe


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """
    Load Fake.csv and True.csv, attach labels, and concatenate.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns ['text', 'label'].
    """
    fake_df = pd.read_csv(FAKE_CSV, encoding="utf-8")
    true_df = pd.read_csv(TRUE_CSV, encoding="utf-8")

    # Attach labels
    fake_df["label"] = LABEL_FAKE
    true_df["label"] = LABEL_REAL

    # Keep only relevant columns (handle both Kaggle and HuggingFace formats)
    for df in (fake_df, true_df):
        if "title" in df.columns and "text" in df.columns:
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        elif "title" in df.columns:
            df.rename(columns={"title": "text"}, inplace=True)

    combined = pd.concat(
        [fake_df[["text", "label"]], true_df[["text", "label"]]],
        ignore_index=True,
    ).sample(frac=1, random_state=RANDOM_SEED)   # Shuffle rows

    print(f"[train] Loaded  Fake: {len(fake_df):>7,}  Real: {len(true_df):>7,}  "
          f"Total: {len(combined):>7,}")
    return combined


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_training() -> None:
    """Execute the full training pipeline end-to-end."""
    t0 = time.time()

    # 1. Ensure dataset is downloaded
    download()

    # 2. Load data
    print("\n[train] Loading raw data …")
    df = load_raw_data()

    # 3. Preprocess
    print("[train] Preprocessing text …")
    df = preprocess_dataframe(df, text_col="text")
    print(f"[train] After cleaning: {len(df):,} rows remain.")

    # 4. Save processed dataset
    COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMBINED_CSV, index=False)
    print(f"[train] Processed CSV saved → {COMBINED_CSV}")

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    print(f"[train] Split  Train: {len(X_train):,}  Test: {len(X_test):,}")

    # 6. TF-IDF vectorization
    print("[train] Fitting TF-IDF vectorizer …")
    vectorizer = build_vectorizer()
    X_train_tfidf = fit_vectorizer(vectorizer, X_train)
    X_test_tfidf  = transform(vectorizer, X_test)
    print(f"[train] Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # 7. Train Logistic Regression
    print("[train] Training Logistic Regression …")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train_tfidf, y_train)
    print("[train] Training complete.")

    # 8. Evaluate
    print("\n[train] Evaluating on test set …")
    y_pred = model.predict(X_test_tfidf)
    run_full_evaluation(y_test, y_pred)

    # 9. Save artefacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_vectorizer(vectorizer, VECTORIZER_PATH)
    print(f"\n[train] Model saved     → {MODEL_PATH}")
    print(f"[train] Vectorizer saved → {VECTORIZER_PATH}")

    elapsed = time.time() - t0
    print(f"\n[train] ✓ Pipeline finished in {elapsed:.1f}s")


if __name__ == "__main__":
    run_training()
