# -*- coding: utf-8 -*-
"""
dataset.py — Dataset loading, splitting, tokenisation, and analysis.

Usage (from backend/):
    # 1. Generate frozen splits (run once; safe to re-run — skips if manifest exists)
    python train/transformer/dataset.py --prepare

    # 2. Run token-length analysis on the training split
    python train/transformer/dataset.py --analyze-lengths

    # Both at once
    python train/transformer/dataset.py --prepare --analyze-lengths
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from train.transformer.config import (
    LABEL_FAKE, LABEL_REAL, ID2LABEL, LABEL2ID,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    UNIFIED_CSV, SPLIT_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, SPLIT_MANIFEST,
    SMOKE_SAMPLES, TOKEN_STATS_JSON,
    MODEL_CHECKPOINT, SMOKE_MAX_SEQ_LEN,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset fingerprinting
# ─────────────────────────────────────────────────────────────────────────────

def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return hex MD5 digest of a file without loading it all into RAM."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Raw loading
# ─────────────────────────────────────────────────────────────────────────────

def load_unified(path: Path = UNIFIED_CSV) -> pd.DataFrame:
    """
    Load the unified dataset produced by Milestone 1B data_engineering.py.

    Validates:
    - Required columns exist: text, label
    - Label values are exactly {0, 1}  (0=FAKE, 1=REAL)
    """
    df = pd.read_csv(path, encoding="utf-8")

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unified dataset missing columns: {missing}")

    actual_labels = set(df["label"].unique().tolist())
    expected = {LABEL_FAKE, LABEL_REAL}
    if actual_labels != expected:
        raise ValueError(
            f"Unexpected label values in dataset: {actual_labels}. "
            f"Expected: {expected}  (0=FAKE, 1=REAL)"
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop rows whose *text* column is an exact string duplicate.
    Returns (deduplicated_df, n_removed).
    """
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df, before - len(df)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stratified splits
# ─────────────────────────────────────────────────────────────────────────────

def _class_dist(df: pd.DataFrame) -> dict:
    vc = df["label"].value_counts().to_dict()
    return {ID2LABEL[k]: int(v) for k, v in vc.items()}


def make_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float   = VAL_RATIO,
    test_ratio: float  = TEST_RATIO,
    seed: int          = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/10/10 split.

    The val+test pool is first separated from train using (val+test) / total,
    then val and test are split 50/50 from that pool — giving exact 10/10 ratios.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Ratios must sum to 1.0"

    val_test_ratio = val_ratio + test_ratio  # fraction going to val+test pool
    val_of_pool    = val_ratio / val_test_ratio  # val fraction within the pool

    train_df, valtest_df = train_test_split(
        df, test_size=val_test_ratio, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        valtest_df, test_size=1 - val_of_pool, stratify=valtest_df["label"],
        random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Persist frozen splits + manifest
# ─────────────────────────────────────────────────────────────────────────────

def prepare_splits(force: bool = False) -> dict:
    """
    Build and save frozen train/val/test splits with a provenance manifest.

    If the manifest already exists and ``force=False``, loading is skipped
    and the existing manifest is returned — preserving reproducibility.

    Parameters
    ----------
    force : bool
        If True, regenerate splits even if the manifest already exists.

    Returns
    -------
    dict  — the split manifest.
    """
    if SPLIT_MANIFEST.exists() and not force:
        print(f"[dataset] Frozen splits already exist at {SPLIT_DIR}")
        print(f"          To regenerate, run with --force.")
        with open(SPLIT_MANIFEST, encoding="utf-8") as f:
            return json.load(f)

    print(f"[dataset] Loading unified dataset from {UNIFIED_CSV} …")
    df_raw = load_unified()
    rows_before = len(df_raw)
    fingerprint = md5_file(UNIFIED_CSV)

    df, n_removed = remove_duplicates(df_raw)
    rows_after = len(df)

    print(f"[dataset]   Rows before dedup: {rows_before:,}")
    print(f"[dataset]   Duplicates removed: {n_removed:,}")
    print(f"[dataset]   Rows after dedup: {rows_after:,}")

    train_df, val_df, test_df = make_splits(df)

    print(f"[dataset]   Train : {len(train_df):,}  {_class_dist(train_df)}")
    print(f"[dataset]   Val   : {len(val_df):,}  {_class_dist(val_df)}")
    print(f"[dataset]   Test  : {len(test_df):,}  {_class_dist(test_df)}")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8")
    val_df.to_csv(VAL_CSV,   index=False, encoding="utf-8")
    test_df.to_csv(TEST_CSV,  index=False, encoding="utf-8")

    manifest = {
        "source_dataset": str(UNIFIED_CSV),
        "source_fingerprint_md5": fingerprint,
        "rows_before_dedup": rows_before,
        "rows_after_dedup":  rows_after,
        "duplicates_removed": n_removed,
        "random_seed": RANDOM_SEED,
        "split_ratios": {
            "train": TRAIN_RATIO, "validation": VAL_RATIO, "test": TEST_RATIO
        },
        "split_sizes": {
            "train": len(train_df), "validation": len(val_df), "test": len(test_df)
        },
        "class_distribution": {
            "train": _class_dist(train_df),
            "validation": _class_dist(val_df),
            "test":  _class_dist(test_df),
        },
        "label_mapping": {
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
            "label2id": LABEL2ID,
        },
        "split_files": {
            "train": str(TRAIN_CSV),
            "validation": str(VAL_CSV),
            "test": str(TEST_CSV),
        },
    }

    with open(SPLIT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[dataset]   Manifest saved → {SPLIT_MANIFEST}")
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# 6. Token-length analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_token_lengths(
    sample_size: int = 10_000,
    seed: int = RANDOM_SEED,
    save: bool = True,
) -> dict:
    """
    Tokenise a representative sample of the training split and report
    token-length statistics used to choose MAX_SEQ_LEN.

    Parameters
    ----------
    sample_size : int
        Number of articles to tokenise (speeds up analysis; default 10k).
    seed : int
        Random seed for the sample.
    save : bool
        If True, save stats to TOKEN_STATS_JSON.

    Returns
    -------
    dict with keys: median, p90, p95, p99, max, mean, sample_size.
    """
    from transformers import AutoTokenizer  # lazy import — not needed at split time

    print(f"\n[dataset] Token-length analysis using '{MODEL_CHECKPOINT}' tokenizer …")
    print(f"          (sampling {sample_size:,} articles from training split)")

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Training split not found at {TRAIN_CSV}. "
            "Run `python train/transformer/dataset.py --prepare` first."
        )

    train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    sample = train_df.sample(
        n=min(sample_size, len(train_df)), random_state=seed
    )["text"].tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    lengths = []
    for text in sample:
        ids = tokenizer.encode(str(text), add_special_tokens=True)
        lengths.append(len(ids))

    lengths = np.array(lengths)
    stats = {
        "sample_size":      int(len(lengths)),
        "mean":             float(np.mean(lengths)),
        "median":           float(np.median(lengths)),
        "p75":              float(np.percentile(lengths, 75)),
        "p90":              float(np.percentile(lengths, 90)),
        "p95":              float(np.percentile(lengths, 95)),
        "p99":              float(np.percentile(lengths, 99)),
        "max":              int(np.max(lengths)),
        "min":              int(np.min(lengths)),
        "tokenizer":        MODEL_CHECKPOINT,
        "notes": (
            "Lengths include [CLS] and [SEP] special tokens. "
            "Articles longer than MAX_SEQ_LEN will be truncated from the end."
        ),
    }

    print()
    print("  Token-length statistics:")
    print(f"    Sample size : {stats['sample_size']:,}")
    print(f"    Mean        : {stats['mean']:.1f}")
    print(f"    Median      : {stats['median']:.1f}")
    print(f"    75th pct    : {stats['p75']:.1f}")
    print(f"    90th pct    : {stats['p90']:.1f}")
    print(f"    95th pct    : {stats['p95']:.1f}")
    print(f"    99th pct    : {stats['p99']:.1f}")
    print(f"    Max         : {stats['max']}")
    print()
    print("  Truncation trade-off guide:")
    for candidate in [128, 256, 384, 512]:
        pct_covered = float(np.mean(lengths <= candidate)) * 100
        print(f"    max_length={candidate:>3} → covers {pct_covered:.1f}% of articles in full")
    print()

    if save:
        TOKEN_STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_STATS_JSON, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved → {TOKEN_STATS_JSON}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 7. Smoke-test subsample
# ─────────────────────────────────────────────────────────────────────────────

def get_smoke_sample(
    df: pd.DataFrame | None = None,
    n_per_class: int = SMOKE_SAMPLES // 2,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Return a small class-balanced DataFrame for the local smoke test.

    Labels are sampled equally to ensure both classes are represented.
    """
    if df is None:
        df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
    fake = df[df["label"] == LABEL_FAKE].sample(
        n=min(n_per_class, (df["label"] == LABEL_FAKE).sum()), random_state=seed
    )
    real = df[df["label"] == LABEL_REAL].sample(
        n=min(n_per_class, (df["label"] == LABEL_REAL).sum()), random_state=seed
    )
    return pd.concat([fake, real]).sample(frac=1, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class NewsDataset:
    """
    PyTorch-compatible Dataset for DistilBERT fine-tuning.

    Tokenisation is performed once at construction time (for datasets
    small enough to fit in RAM); this avoids repeated tokenisation cost
    during training.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'text' (str) and 'label' (int).
    tokenizer : transformers.PreTrainedTokenizer
        The DistilBERT tokenizer.
    max_length : int
        Maximum token sequence length; longer inputs are truncated from end.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch is required. Install via: pip install torch") from e

        self._torch = torch
        texts  = df["text"].astype(str).tolist()
        labels = df["label"].tolist()

        enc = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset preparation and analysis for DistilBERT.")
    p.add_argument("--prepare",         action="store_true", help="Generate and freeze splits.")
    p.add_argument("--force",           action="store_true", help="Regenerate splits even if manifest exists.")
    p.add_argument("--analyze-lengths", action="store_true", help="Run token-length analysis.")
    p.add_argument("--sample-size",     type=int, default=10_000,
                   help="Articles to sample for length analysis (default: 10000).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.prepare and not args.analyze_lengths:
        print("No action specified. Use --prepare and/or --analyze-lengths.")
        print("  python train/transformer/dataset.py --prepare --analyze-lengths")
        sys.exit(0)

    if args.prepare:
        prepare_splits(force=args.force)

    if args.analyze_lengths:
        analyze_token_lengths(sample_size=args.sample_size)
