"""
explore.py – Exploratory Data Analysis for the VeriNews AI dataset.

Generates:
  - Console summary (counts, nulls, duplicates, avg length)
  - docs/figures/label_distribution.png
  - docs/figures/article_length_distribution.png
  - docs/figures/top_words_fake.png
  - docs/figures/top_words_real.png

Usage (from backend/):
    python notebooks/explore.py
"""
import sys
from pathlib import Path
from collections import Counter

# Allow running from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from train.config import FAKE_CSV, FIGURES_DIR, LABEL_FAKE, LABEL_REAL, LABEL_NAMES, TRUE_CSV
from train.download_dataset import download
from train.preprocess import clean_text


sns.set_theme(style="darkgrid", palette="muted")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    download()
    fake_df = pd.read_csv(FAKE_CSV, encoding="utf-8")
    true_df = pd.read_csv(TRUE_CSV, encoding="utf-8")

    for df in (fake_df, true_df):
        if "title" in df.columns and "text" in df.columns:
            df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        elif "title" in df.columns:
            df.rename(columns={"title": "text"}, inplace=True)

    fake_df["label"] = LABEL_FAKE
    true_df["label"] = LABEL_REAL
    return pd.concat([fake_df[["text", "label"]], true_df[["text", "label"]]], ignore_index=True)


# ── EDA functions ─────────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> None:
    """Print a console summary of dataset characteristics."""
    print("\n" + "=" * 60)
    print("VeriNews AI — Dataset Summary")
    print("=" * 60)

    counts = df["label"].value_counts()
    print(f"\n  Total articles    : {len(df):>8,}")
    print(f"  Fake articles     : {counts.get(LABEL_FAKE, 0):>8,}")
    print(f"  Real articles     : {counts.get(LABEL_REAL, 0):>8,}")

    # Missing values
    null_counts = df.isnull().sum()
    print(f"\n  Missing values    :")
    for col, cnt in null_counts.items():
        print(f"    {col:<12}: {cnt:,}")

    # Duplicates
    dupe_count = df.duplicated(subset=["text"]).sum()
    print(f"\n  Duplicate rows    : {dupe_count:,}")

    # Article lengths
    df["_len"] = df["text"].fillna("").str.split().str.len()
    print(f"\n  Article length (words):")
    print(f"    Mean   : {df['_len'].mean():.1f}")
    print(f"    Median : {df['_len'].median():.1f}")
    print(f"    Min    : {df['_len'].min():,}")
    print(f"    Max    : {df['_len'].max():,}")
    df.drop(columns=["_len"], inplace=True)

    print("=" * 60 + "\n")


def plot_label_distribution(df: pd.DataFrame) -> None:
    """Bar chart of Fake vs. Real article counts."""
    counts = df["label"].value_counts().rename(index=LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(6, 4))
    colours = ["#ef4444", "#22c55e"]
    bars = ax.bar(counts.index, counts.values, color=colours, edgecolor="white", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Label Distribution — Fake vs. Real Articles", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Articles")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, max(counts.values) * 1.15)
    fig.tight_layout()

    out = FIGURES_DIR / "label_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[explore] Saved → {out}")


def plot_length_distribution(df: pd.DataFrame) -> None:
    """Overlapping KDE of article lengths for each class."""
    df = df.copy()
    df["word_count"] = df["text"].fillna("").str.split().str.len()
    # Cap at 99th percentile to avoid extreme outliers compressing the plot
    cap = int(df["word_count"].quantile(0.99))
    df = df[df["word_count"] <= cap]

    fig, ax = plt.subplots(figsize=(8, 4))
    for label_id, colour, name in [(LABEL_FAKE, "#ef4444", "Fake"), (LABEL_REAL, "#22c55e", "Real")]:
        subset = df[df["label"] == label_id]["word_count"]
        sns.kdeplot(subset, ax=ax, fill=True, alpha=0.35, color=colour, label=name, linewidth=1.5)

    ax.set_title("Article Length Distribution (word count)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    out = FIGURES_DIR / "article_length_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[explore] Saved → {out}")


def _top_words(texts: pd.Series, n: int = 20) -> list:
    """Return the top-n words from a Series of text (after basic cleaning)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    all_words: list = []
    for t in texts.dropna():
        cleaned = clean_text(str(t))
        all_words.extend(w for w in cleaned.split() if w not in ENGLISH_STOP_WORDS and len(w) > 2)
    return Counter(all_words).most_common(n)


def plot_top_words(df: pd.DataFrame) -> None:
    """Horizontal bar chart of the top 20 words for Fake and Real articles."""
    for label_id, colour, name in [(LABEL_FAKE, "#ef4444", "fake"), (LABEL_REAL, "#22c55e", "real")]:
        subset = df[df["label"] == label_id]["text"]
        top    = _top_words(subset, n=20)
        if not top:
            continue
        words, counts = zip(*top)

        fig, ax = plt.subplots(figsize=(8, 6))
        y_pos = range(len(words))
        ax.barh(y_pos, counts, color=colour, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top 20 Words — {name.capitalize()} Articles",
                     fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()

        out = FIGURES_DIR / f"top_words_{name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[explore] Saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_eda() -> None:
    print("=" * 60)
    print("VeriNews AI — Exploratory Data Analysis")
    print("=" * 60)

    df = load_data()
    summarise(df)

    print("[explore] Generating plots …")
    plot_label_distribution(df)
    plot_length_distribution(df)
    plot_top_words(df)

    print("\n[explore] ✓ EDA complete. Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    run_eda()
