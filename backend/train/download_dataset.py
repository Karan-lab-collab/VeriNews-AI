"""
download_dataset.py – Dataset acquisition for VeriNews AI.

Strategy (in order):
  1. If Fake.csv and True.csv already exist in datasets/raw/, skip.
  2. Try Kaggle API (requires ~/.kaggle/kaggle.json or KAGGLE_* env vars).
  3. Fall back to HuggingFace `datasets` library (GonzaloA/fake_news).

Run from the backend/ directory:
    python train/download_dataset.py
"""
import sys
from pathlib import Path

# Allow running as a top-level script from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from train.config import FAKE_CSV, RAW_DIR, TRUE_CSV


# ── helpers ───────────────────────────────────────────────────────────────────

def _already_downloaded() -> bool:
    """Return True if both raw CSVs are already present and non-empty."""
    return FAKE_CSV.exists() and TRUE_CSV.exists()


def _try_kaggle() -> bool:
    """
    Attempt to download via the Kaggle API.

    Returns True on success, False if credentials are missing or kaggle
    is not installed.
    """
    try:
        import kaggle  # noqa: F401 — triggers credential validation
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()
        print("[download] Kaggle credentials found. Downloading dataset …")
        api.dataset_download_files(
            "clmentbisaillon/fake-and-real-news-dataset",
            path=str(RAW_DIR),
            unzip=True,
            quiet=False,
        )
        # Kaggle may capitalise filenames differently — normalise
        for src, dst in [(RAW_DIR / "Fake.csv", FAKE_CSV), (RAW_DIR / "True.csv", TRUE_CSV)]:
            if src.exists() and not dst.exists():
                src.rename(dst)
        return FAKE_CSV.exists() and TRUE_CSV.exists()
    except Exception as exc:
        print(f"[download] Kaggle unavailable ({exc}). Trying HuggingFace …")
        return False


def _try_huggingface() -> bool:
    """
    Download via HuggingFace datasets (GonzaloA/fake_news).

    The dataset has columns: title, text, label  (0=FAKE, 1=REAL)
    We split it into Fake.csv and True.csv to match the Kaggle format.

    Returns True on success.
    """
    try:
        from datasets import load_dataset

        print("[download] Loading GonzaloA/fake_news from HuggingFace …")
        ds = load_dataset("GonzaloA/fake_news")

        # Concatenate all splits into a single DataFrame
        frames = []
        for split_name in ds.keys():
            frames.append(ds[split_name].to_pandas())
        df = pd.concat(frames, ignore_index=True)

        # Normalise columns
        # GonzaloA/fake_news: columns = [title, text, label]  label: 0=FAKE, 1=REAL
        df.columns = [c.lower().strip() for c in df.columns]
        if "text" not in df.columns and "content" in df.columns:
            df.rename(columns={"content": "text"}, inplace=True)

        # Combine title + text (mirrors Kaggle dataset style)
        if "title" in df.columns:
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        df["text"] = df["text"].str.strip()

        # Determine label column
        label_col = "label" if "label" in df.columns else df.columns[-1]
        # Identify fake vs real by most common label values
        # HuggingFace GonzaloA/fake_news: 0 = FAKE, 1 = REAL
        fake_df = df[df[label_col] == 0][["text"]].copy()
        true_df = df[df[label_col] == 1][["text"]].copy()

        fake_df["subject"] = "politicsNews"
        true_df["subject"] = "politicsNews"

        print(f"[download] Fake articles : {len(fake_df):,}")
        print(f"[download] Real articles : {len(true_df):,}")

        fake_df.to_csv(FAKE_CSV, index=False, encoding="utf-8")
        true_df.to_csv(TRUE_CSV, index=False, encoding="utf-8")
        print(f"[download] Saved → {FAKE_CSV}")
        print(f"[download] Saved → {TRUE_CSV}")
        return True
    except Exception as exc:
        print(f"[download] HuggingFace download failed: {exc}")
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def download():
    """Main entry point: download dataset using the best available method."""
    print("=" * 60)
    print("VeriNews AI — Dataset Downloader")
    print("=" * 60)

    if _already_downloaded():
        fake_rows = sum(1 for _ in open(FAKE_CSV, encoding="utf-8")) - 1
        true_rows = sum(1 for _ in open(TRUE_CSV, encoding="utf-8")) - 1
        print(f"[download] Dataset already present.")
        print(f"           Fake.csv : {fake_rows:,} rows")
        print(f"           True.csv : {true_rows:,} rows")
        return True

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    success = _try_kaggle() or _try_huggingface()

    if not success:
        print(
            "\n[download] Automatic download failed.\n"
            "Please manually download the dataset from:\n"
            "  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            f"and place Fake.csv and True.csv in:\n  {RAW_DIR}\n"
        )
        sys.exit(1)

    print("\n[download] ✓ Dataset ready.")
    return True


if __name__ == "__main__":
    download()
