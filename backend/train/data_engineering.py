# -*- coding: utf-8 -*-
"""
data_engineering.py – Milestone 1B: Multi-dataset pipeline for VeriNews AI.

Downloads, normalises, and merges multiple fake news datasets to reduce
the domain bias identified in Milestone 1.5.

Strategy: MERGE  (Task 2 recommendation)
  Source 1 – ISOT      : existing Fake.csv / True.csv (~40k political articles)
  Source 2 – LIAR      : ~12k PolitiFact statements (diverse statement styles)
  Source 3 – WELFake   : ~72k multi-source articles (if available on HuggingFace)

Key engineering decisions:
  • Publisher datelines removed via preprocess.remove_publisher_datelines().
  • LIAR labels binarised: {true, mostly-true} → REAL; {false, pants-fire} → FAKE.
    Ambiguous labels {half-true, barely-true} are excluded for cleaner signal.
  • All sources normalised to columns: [text, label, source].
  • Global deduplication on cleaned text (exact-match hash).
  • Final shuffle with fixed seed.

Run from backend/:
    python train/data_engineering.py

Output:
    datasets/processed/unified_dataset.csv   (columns: text, label, source)
    datasets/raw/sources/isot.csv
    datasets/raw/sources/liar.csv
    datasets/raw/sources/welfake.csv         (if available)
    docs/dataset_research.md
"""

import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

from train.config import (
    BACKEND_DIR, FAKE_CSV, TRUE_CSV, UNIFIED_CSV,
    DATASET_SOURCES_DIR, DATASET_RESEARCH_DOC,
    LABEL_FAKE, LABEL_REAL, RANDOM_SEED,
)
from train.preprocess import clean_text

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text_hash(text: str) -> str:
    """Stable 64-bit hash for deduplication on cleaned text."""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def _print_source_stats(name: str, df: pd.DataFrame) -> None:
    fake_n = (df["label"] == LABEL_FAKE).sum()
    real_n = (df["label"] == LABEL_REAL).sum()
    print(f"  {name:<12}  total={len(df):>7,}  FAKE={fake_n:>6,}  REAL={real_n:>6,}")


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: ISOT (existing Fake.csv / True.csv)
# ─────────────────────────────────────────────────────────────────────────────

def load_isot() -> pd.DataFrame:
    """
    Load the existing ISOT dataset from raw CSVs.

    The GonzaloA/fake_news HuggingFace version provides these columns:
      text, subject  (Fake.csv)
      text, subject  (True.csv)

    Returns
    -------
    pd.DataFrame with columns: text, label (int), source (str)
    """
    print("[data] Loading ISOT from raw CSVs ...")
    fake_df = pd.read_csv(FAKE_CSV, encoding="utf-8")
    true_df = pd.read_csv(TRUE_CSV, encoding="utf-8")

    fake_df["label"]  = LABEL_FAKE
    true_df["label"]  = LABEL_REAL
    fake_df["source"] = "ISOT"
    true_df["source"] = "ISOT"

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Standardise: use 'text' column
    if "text" not in df.columns and "title" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df.get("text", pd.Series([""] * len(df)))

    df = df[["text", "label", "source"]].dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    print(f"[data]   ISOT loaded: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: LIAR dataset
# ─────────────────────────────────────────────────────────────────────────────

# LIAR integer labels from HuggingFace `liar` dataset
_LIAR_LABEL_TO_INT = {
    0: "pants-fire",   # → FAKE
    1: "false",        # → FAKE
    2: "barely-true",  # → SKIP (ambiguous)
    3: "half-true",    # → SKIP (ambiguous)
    4: "mostly-true",  # → REAL
    5: "true",         # → REAL
}
_LIAR_FAKE_IDS  = {0, 1}          # pants-fire, false
_LIAR_REAL_IDS  = {4, 5}          # mostly-true, true
_LIAR_SKIP_IDS  = {2, 3}          # half-true, barely-true — excluded


def _build_liar_text(row) -> str:
    """
    Compose article-style text from LIAR fields.
    Combines statement + subject + speaker context for richer signal.
    """
    parts = []
    if pd.notna(row.get("statement")) and str(row["statement"]).strip():
        parts.append(str(row["statement"]).strip())
    if pd.notna(row.get("subject")) and str(row["subject"]).strip():
        parts.append("Topic: " + str(row["subject"]).strip())
    if pd.notna(row.get("context")) and str(row["context"]).strip():
        ctx = str(row["context"]).strip()
        if ctx.lower() not in ("nan", "n/a", "none"):
            parts.append("Context: " + ctx)
    return " ".join(parts)


def load_liar() -> pd.DataFrame:
    """
    Download LIAR dataset from HuggingFace and binarise labels.

    Only includes pants-fire/false (FAKE) and mostly-true/true (REAL).
    Ambiguous labels are excluded to keep clean training signal.

    Returns
    -------
    pd.DataFrame with columns: text, label (int), source (str)
    """
    print("[data] Loading LIAR from HuggingFace ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("liar")

        frames = []
        for split_name, split in ds.items():
            split_df = split.to_pandas()
            frames.append(split_df)

        df_raw = pd.concat(frames, ignore_index=True)
        print(f"[data]   LIAR raw rows: {len(df_raw):,}")

        # Binarise
        rows = []
        for _, row in df_raw.iterrows():
            lbl_id = int(row["label"])
            if lbl_id in _LIAR_FAKE_IDS:
                binary_label = LABEL_FAKE
            elif lbl_id in _LIAR_REAL_IDS:
                binary_label = LABEL_REAL
            else:
                continue  # skip ambiguous
            text = _build_liar_text(row)
            if text.strip():
                rows.append({"text": text, "label": binary_label, "source": "LIAR"})

        df = pd.DataFrame(rows)
        print(f"[data]   LIAR after binarisation: {len(df):,} rows "
              f"(dropped {len(df_raw) - len(df):,} ambiguous)")
        return df

    except Exception as exc:
        print(f"[data]   LIAR download failed: {exc}")
        return pd.DataFrame(columns=["text", "label", "source"])


# ─────────────────────────────────────────────────────────────────────────────
# Source 3: mrm8488/fake-news (McIntire-derived, labels are INVERTED)
# ─────────────────────────────────────────────────────────────────────────────
#
# IMPORTANT: Label analysis revealed that mrm8488/fake-news has its labels
# flipped relative to our convention:
#   label=0 → 21,378 articles contain "Reuters" → actually REAL news
#   label=1 → sensationalist political articles → actually FAKE news
#
# We invert the labels before merging.
# This adds ~44k diverse articles beyond the narrow ISOT political corpus.

def load_mrm8488() -> pd.DataFrame:
    """
    Download mrm8488/fake-news from HuggingFace and correct inverted labels.

    This dataset derives from the McIntire Kaggle corpus. After label
    correction, it adds ~21k REAL and ~23k FAKE articles to the training pool,
    diversifying beyond the ISOT political focus.

    Returns
    -------
    pd.DataFrame with columns: text, label (int), source (str)
    """
    print("[data] Loading mrm8488/fake-news from HuggingFace ...")
    try:
        from datasets import load_dataset
        ds     = load_dataset("mrm8488/fake-news")
        frames = [split.to_pandas() for split in ds.values()]
        df_raw = pd.concat(frames, ignore_index=True)
        print(f"[data]   mrm8488 raw rows: {len(df_raw):,}")

        df = df_raw[["text", "label"]].copy()
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str)

        # LABEL INVERSION — labels are flipped in this dataset:
        #   original 0 → Reuters articles → REAL → map to LABEL_REAL (1)
        #   original 1 → sensationalist    → FAKE → map to LABEL_FAKE (0)
        df["label"] = df["label"].map({0: LABEL_REAL, 1: LABEL_FAKE})
        df = df.dropna(subset=["label"])
        df["label"]  = df["label"].astype(int)
        df["source"] = "mrm8488"

        print(f"[data]   mrm8488 after label correction: {len(df):,} rows  "
              f"FAKE={int((df['label']==LABEL_FAKE).sum()):,}  "
              f"REAL={int((df['label']==LABEL_REAL).sum()):,}")
        return df

    except Exception as exc:
        print(f"[data]   mrm8488/fake-news download failed: {exc}")
        return pd.DataFrame(columns=["text", "label", "source"])


# ─────────────────────────────────────────────────────────────────────────────
# Merge + deduplicate
# ─────────────────────────────────────────────────────────────────────────────

def build_unified_dataset(*source_dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multiple source DataFrames into one unified dataset.

    Steps
    -----
    1. Concatenate all sources.
    2. Apply clean_text() (includes dateline removal — Milestone 1B).
    3. Remove rows with empty text after cleaning.
    4. Remove exact-duplicate texts using MD5 hash.
    5. Shuffle with fixed seed.
    6. Reset index.

    Returns
    -------
    pd.DataFrame with columns: text, label, source
    """
    print("\n[data] Building unified dataset ...")
    combined = pd.concat(
        [df for df in source_dfs if len(df) > 0],
        ignore_index=True,
    )
    print(f"[data]   Combined (before cleaning): {len(combined):,} rows")

    # Apply full preprocessing pipeline (includes dateline removal)
    print("[data]   Applying preprocessing pipeline (this may take a minute) ...")
    combined["text"] = combined["text"].apply(clean_text)

    # Drop empty
    before_empty = len(combined)
    combined = combined[combined["text"].str.strip().ne("")].copy()
    print(f"[data]   Removed {before_empty - len(combined):,} empty rows after cleaning")

    # Deduplicate on cleaned text hash
    before_dedup = len(combined)
    combined["_hash"] = combined["text"].apply(_text_hash)
    combined = combined.drop_duplicates(subset=["_hash"]).drop(columns=["_hash"])
    print(f"[data]   Removed {before_dedup - len(combined):,} exact duplicate texts")

    # Shuffle
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"[data]   Final unified dataset: {len(combined):,} rows")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Save per-source and unified CSVs
# ─────────────────────────────────────────────────────────────────────────────

def save_source_csv(df: pd.DataFrame, name: str) -> None:
    """Save a per-source DataFrame to datasets/raw/sources/."""
    if len(df) == 0:
        return
    path = DATASET_SOURCES_DIR / f"{name.lower()}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[data]   Saved {name} -> {path}")


def save_unified(df: pd.DataFrame) -> None:
    UNIFIED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNIFIED_CSV, index=False, encoding="utf-8")
    print(f"[data]   Unified dataset saved -> {UNIFIED_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset research document (Task 1)
# ─────────────────────────────────────────────────────────────────────────────

def write_dataset_research_doc(source_stats: dict) -> None:
    """
    Write docs/dataset_research.md with dataset comparison table and
    our recommendation (Task 1 + Task 2).
    """
    doc = """# Fake News Dataset Research — VeriNews AI Milestone 1B

## Task 1: Dataset Comparison

| Dataset | Size | Domain | Country | Categories | Labels | Access |
|---------|------|--------|---------|------------|--------|--------|
| **ISOT** (current) | ~44k articles | Political news | USA | Politics, World | Binary | HuggingFace: GonzaloA/fake_news |
| **LIAR** | 12,836 statements | Political statements | USA | Politics, Finance, Healthcare, Education | 6-class (binarisable) | HuggingFace: `liar` |
| **WELFake** | 72,134 articles | Multi-source | USA + Global | Politics, Tech, Business, Entertainment | Binary | HuggingFace: rabiaqayyum/WELFake |
| **FakeNewsNet** | ~23k articles | Politics + Entertainment | USA | PolitiFact + GossipCop | Binary | Requires web scraping |
| **McIntire** | ~13k articles | Mixed | USA | Tech, Business, Sports, Entertainment | Binary | Kaggle only (credentials needed) |
| **CoAID** | ~4k documents | Health / COVID | Global | Healthcare | Multi-label | GitHub download |
| **FEVER** | 185k claims | Fact-checking | Global | Wikipedia claims | 3-class | HuggingFace: fever |

### Advantages and Limitations

| Dataset | Advantages | Limitations |
|---------|------------|-------------|
| ISOT | Large, clean, binary labels | Only US political news 2015-2018; causes Reuters/Washington bias |
| LIAR | Multi-class, well-cited, diverse statement styles | Short text (statements not articles); political domain only |
| WELFake | Merges 4 sources, multi-domain articles | May contain overlapping ISOT data; quality varies by source |
| FakeNewsNet | Includes social media metadata | Requires scraping; dead URLs reduce coverage |
| McIntire | Non-political domain (tech, sports, entertainment) | Kaggle credentials needed; smaller size |
| CoAID | Health domain coverage | Very small; COVID-specific |
| FEVER | Very large; diverse claims | Claims, not articles; different task structure |

---

## Task 2: Strategy Recommendation

### Decision: **B — Merge multiple datasets**

**Rationale:**

1. **Publisher leakage is the primary failure mode.** Feature analysis showed `reuters` 
   with coefficient +21.7 — the model learned to classify by publisher, not content.
   Removing datelines and merging with non-Reuters datasets directly addresses this.

2. **Domain breadth matters more than dataset size.** The ISOT dataset is ~40k articles 
   but covers only one domain. Adding LIAR (diverse statement styles) and WELFake 
   (multi-source) increases domain breadth without reducing the proven ISOT data.

3. **Replacement is risky.** Discarding ISOT removes 40k clean binary-labelled articles 
   and risks degrading political news performance where the model already works well.

4. **Keeping existing is insufficient.** The 52-point accuracy gap between train/test 
   (97.6%) and manual validation (45%) is unacceptably large for production deployment.

### Engineering Changes Applied

| Change | Reason | Expected Impact |
|--------|---------|-----------------|
| Remove publisher datelines | Eliminate `reuters` (+21.7) leakage | High |
| Add LIAR dataset | Add diverse statement styles | Medium |
| Add WELFake (if available) | Add multi-domain articles | High |
| Deduplicate across sources | Prevent training data contamination | Low |
| Binarise LIAR labels | Clean signal (skip ambiguous half-true) | Medium |

---

## Source Statistics (This Run)

| Source | Total | FAKE | REAL |
|--------|-------|------|------|
"""
    for src, stats in source_stats.items():
        doc += f"| {src} | {stats['total']:,} | {stats['fake']:,} | {stats['real']:,} |\n"

    doc += "\n---\n\n*Generated by VeriNews AI — Milestone 1B data_engineering.py*\n"

    DATASET_RESEARCH_DOC.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_RESEARCH_DOC, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[data]   Research doc saved -> {DATASET_RESEARCH_DOC}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  VeriNews AI — Milestone 1B: Dataset Engineering Pipeline")
    print("=" * 70)

    # 1. Load all sources
    isot_df   = load_isot()
    liar_df   = load_liar()
    mrm_df    = load_mrm8488()

    # 2. Save per-source CSVs (before cleaning)
    save_source_csv(isot_df,  "isot")
    save_source_csv(liar_df,  "liar")
    save_source_csv(mrm_df,   "mrm8488")

    # 3. Print source stats
    print("\n[data] Source statistics (before merging):")
    source_stats = {}
    for name, df in [("ISOT", isot_df), ("LIAR", liar_df), ("mrm8488", mrm_df)]:
        if len(df) > 0:
            _print_source_stats(name, df)
            source_stats[name] = {
                "total": len(df),
                "fake":  int((df["label"] == LABEL_FAKE).sum()),
                "real":  int((df["label"] == LABEL_REAL).sum()),
            }

    # 4. Build unified dataset
    unified = build_unified_dataset(isot_df, liar_df, mrm_df)

    # 5. Final stats
    print("\n[data] Unified dataset statistics:")
    _print_source_stats("UNIFIED", unified)
    print("\n[data] Source breakdown in unified dataset:")
    for src, cnt in unified["source"].value_counts().items():
        print(f"  {src:<14}: {cnt:>7,} rows ({cnt/len(unified)*100:.1f}%)")

    # 6. Save
    save_unified(unified)

    # 7. Write research doc
    write_dataset_research_doc(source_stats)

    print("\n" + "=" * 70)
    print("  Dataset engineering complete.")
    print(f"  Next: python train/retrain.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
