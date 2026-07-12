# -*- coding: utf-8 -*-
"""
evaluate_distilbert.py — Evaluation for the fine-tuned DistilBERT model.

Loads the FROZEN test split (never regenerated) and the same 60-example
manual validation set used during baseline research. Reports all metrics
needed for the baseline vs DistilBERT comparison.

Usage (from backend/):
    # After full GPU training; point to the saved checkpoint directory
    python train/transformer/evaluate_distilbert.py --checkpoint saved_models/distilbert_v1/best

    # Evaluate smoke-test checkpoint (for local pipeline verification)
    python train/transformer/evaluate_distilbert.py \\
        --checkpoint saved_models/distilbert_v1/smoke_test \\
        --smoke-test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from train.transformer.config import (
    LABEL_FAKE, LABEL_REAL, LABEL_NAMES, ID2LABEL,
    TEST_CSV, VAL_CSV, SPLIT_MANIFEST,
    VALIDATION_CSV,
    DISTILBERT_DIR, DISTILBERT_RESULTS,
    METRICS_JSON, CLASS_REPORT_TXT, CONFUSION_MATRIX_PNG,
    MANUAL_VAL_CSV, MANUAL_VAL_REPORT, MANUAL_VAL_CM_PNG,
    TRAINING_CONFIG_JSON,
    NASA_ARTICLE,
)

_SMOKE_NOTE = "SMOKE TEST — NOT A RESEARCH RESULT"
_DIV = "=" * 62


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_dir: str):
    """Load tokenizer + model from a HuggingFace save_pretrained directory."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    path = Path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run train_distilbert.py first to generate a checkpoint."
        )
    print(f"[eval] Loading checkpoint from {path} …")
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model     = AutoModelForSequenceClassification.from_pretrained(str(path))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[eval]   Device: {device}")
    return tokenizer, model, device


def predict_batch(
    texts: list[str],
    tokenizer,
    model,
    device,
    max_length: int,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run batch inference. Returns (predictions, confidences).
    predictions : np.ndarray[int]  — class indices (0=FAKE, 1=REAL)
    confidences : np.ndarray[float] — max-softmax probability for the predicted class
    """
    import torch
    import torch.nn.functional as F

    all_preds = []
    all_confs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            [str(t) for t in batch],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        confs = probs.max(axis=-1)
        all_preds.extend(preds.tolist())
        all_confs.extend(confs.tolist())

    return np.array(all_preds), np.array(all_confs)


def get_max_length_from_config() -> int:
    """Read max_seq_len from the saved training_config.json."""
    if TRAINING_CONFIG_JSON.exists():
        with open(TRAINING_CONFIG_JSON, encoding="utf-8") as f:
            return json.load(f).get("max_seq_len", 128)
    return 128


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[eval]   Confusion matrix → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Test-split evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_test_split(
    tokenizer, model, device, max_length: int, smoke_test: bool
) -> dict:
    if smoke_test:
        print(f"\n[eval] {_SMOKE_NOTE}")
        print("[eval] Evaluating on smoke-test validation split (not the frozen test split) …")
        from train.transformer.dataset import get_smoke_sample
        val_df = pd.read_csv(VAL_CSV, encoding="utf-8")
        df = get_smoke_sample(val_df, n_per_class=25)
        split_name = "smoke_val"
    else:
        print(f"\n[eval] Evaluating on FROZEN test split ({TEST_CSV}) …")
        df = pd.read_csv(TEST_CSV, encoding="utf-8")
        split_name = "test"

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    t0 = time.time()
    preds, confs = predict_batch(texts, tokenizer, model, device, max_length)
    elapsed = time.time() - t0

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=LABEL_REAL, zero_division=0)
    rec  = recall_score(labels, preds, pos_label=LABEL_REAL, zero_division=0)
    f1   = f1_score(labels, preds, pos_label=LABEL_REAL, zero_division=0)
    rep  = classification_report(labels, preds, target_names=["FAKE", "REAL"])

    print(f"\n  Test split metrics ({split_name}):")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1        : {f1:.4f}")
    if smoke_test:
        print(f"\n  *** {_SMOKE_NOTE} ***")

    # Latency
    per_sample_ms = (elapsed / len(texts)) * 1000
    print(f"\n  Inference latency ({split_name}, {len(texts)} examples):")
    print(f"    Total    : {elapsed:.2f}s")
    print(f"    Per-sample: {per_sample_ms:.2f}ms")
    print(f"    Device   : {device}")

    # Save
    DISTILBERT_RESULTS.mkdir(parents=True, exist_ok=True)
    metrics = {
        "split":       split_name,
        "smoke_test":  smoke_test,
        "n_examples":  len(texts),
        "accuracy":    round(acc,  4),
        "precision":   round(prec, 4),
        "recall":      round(rec,  4),
        "f1":          round(f1,   4),
        "latency": {
            "total_seconds":   round(elapsed, 3),
            "per_sample_ms":   round(per_sample_ms, 2),
            "device":          str(device),
            "note": "Latency measured on the evaluation device. "
                    "Do not compare CPU vs GPU latency as equivalent.",
        },
        "smoke_test_note": _SMOKE_NOTE if smoke_test else None,
    }
    if not smoke_test:
        with open(METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(CLASS_REPORT_TXT, "w", encoding="utf-8") as f:
            f.write(f"Classification Report — DistilBERT v1 (test split)\n\n{rep}\n")
        save_confusion_matrix(labels, preds, CONFUSION_MATRIX_PNG,
                              "DistilBERT v1 — Test Split")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Manual validation (same 60-example set as baseline)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_manual_validation(
    tokenizer, model, device, max_length: int, smoke_test: bool
) -> dict:
    print(f"\n[eval] Running manual validation (same 60-example set as baseline) …")
    print(f"       {VALIDATION_CSV}")

    val_df = pd.read_csv(VALIDATION_CSV, encoding="utf-8")
    texts      = val_df["text"].tolist()
    labels_str = val_df["label"].tolist()        # "REAL" / "FAKE"
    categories = val_df["category"].tolist()

    labels = [1 if s == "REAL" else 0 for s in labels_str]
    preds, confs = predict_batch(texts, tokenizer, model, device, max_length)

    overall_acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec  = recall_score(labels, preds,  pos_label=1, zero_division=0)
    f1   = f1_score(labels, preds,      pos_label=1, zero_division=0)

    print(f"\n  Overall manual validation accuracy: {overall_acc*100:.1f}%")
    if smoke_test:
        print(f"  *** {_SMOKE_NOTE} ***")

    # Per-category accuracy
    cat_results = defaultdict(lambda: {"correct": 0, "total": 0})
    rows = []
    for i, (text, exp_str, pred_int, conf, cat) in enumerate(
        zip(texts, labels_str, preds, confs, categories)
    ):
        exp_int = 1 if exp_str == "REAL" else 0
        correct = int(pred_int == exp_int)
        cat_results[cat]["correct"] += correct
        cat_results[cat]["total"]   += 1
        rows.append({
            "index":    i + 1,
            "category": cat,
            "expected": exp_str,
            "predicted": ID2LABEL[int(pred_int)],
            "confidence": round(float(conf) * 100, 1),
            "correct":  correct,
        })

    print("\n  Per-category accuracy:")
    print(f"  {'Category':<20} {'Acc':>6}  (correct/total)")
    for cat in sorted(cat_results, key=lambda c: -cat_results[c]["correct"] / max(cat_results[c]["total"], 1)):
        r = cat_results[cat]
        pct = r["correct"] / r["total"] * 100 if r["total"] else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {cat:<20} [{bar}] {pct:5.1f}%  ({r['correct']}/{r['total']})")

    # Save CSV
    out_df = pd.DataFrame(rows)
    out_df.to_csv(MANUAL_VAL_CSV, index=False, encoding="utf-8")
    print(f"\n[eval]   Manual validation CSV → {MANUAL_VAL_CSV}")

    # Save confusion matrix
    save_confusion_matrix(
        labels, preds.tolist(), MANUAL_VAL_CM_PNG,
        "DistilBERT v1 — Manual Validation (60 examples)"
    )

    # Build report markdown
    _write_manual_report(out_df, overall_acc, prec, rec, f1, cat_results, smoke_test)

    return {
        "accuracy":  round(overall_acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "smoke_test_note": _SMOKE_NOTE if smoke_test else None,
    }


def _write_manual_report(
    df: pd.DataFrame,
    acc: float, prec: float, rec: float, f1: float,
    cat_results: dict,
    smoke_test: bool,
) -> None:
    lines = [
        "# DistilBERT v1 — Manual Validation Report\n",
        "> This report uses the **same 60-example validation set** used for the TF-IDF baseline.",
        "> Results are directly comparable to `backend/results/manual_validation_report.md`.\n",
    ]
    if smoke_test:
        lines.insert(1, f"> ⚠ **{_SMOKE_NOTE}**\n")

    lines += [
        "## Overall Metrics\n",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Accuracy  | {acc*100:.1f}% |",
        f"| Precision | {prec*100:.1f}% |",
        f"| Recall    | {rec*100:.1f}% |",
        f"| F1-Score  | {f1*100:.1f}% |\n",
        "## Per-Category Accuracy\n",
        "| Category | Accuracy | Correct / Total |",
        "|----------|----------|-----------------|",
    ]
    for cat in sorted(cat_results, key=lambda c: -cat_results[c]["correct"] / max(cat_results[c]["total"], 1)):
        r = cat_results[cat]
        pct = r["correct"] / r["total"] * 100 if r["total"] else 0
        lines.append(f"| {cat} | {pct:.1f}% | {r['correct']}/{r['total']} |")

    lines += [
        "\n## All Predictions\n",
        "| # | Category | Expected | Predicted | Conf% | Correct |",
        "|---|----------|----------|-----------|-------|---------|",
    ]
    for _, row in df.iterrows():
        ok = "✅" if row["correct"] else "❌"
        lines.append(
            f"| {row['index']} | {row['category']} | {row['expected']} | "
            f"{row['predicted']} | {row['confidence']} | {ok} |"
        )

    with open(MANUAL_VAL_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[eval]   Manual validation report → {MANUAL_VAL_REPORT}")


# ─────────────────────────────────────────────────────────────────────────────
# NASA case study
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_nasa(tokenizer, model, device, max_length: int, smoke_test: bool) -> dict:
    print(f"\n[eval] NASA James Webb case study …")
    preds, confs = predict_batch([NASA_ARTICLE], tokenizer, model, device, max_length)
    pred_label = ID2LABEL[int(preds[0])]
    confidence = float(confs[0]) * 100

    expected = "REAL"
    correct  = pred_label == expected

    print(f"  Text     : {NASA_ARTICLE[:80]}…")
    print(f"  Expected : {expected}")
    print(f"  Predicted: {pred_label}  ({confidence:.2f}% confidence)")
    print(f"  Correct  : {'YES ✅' if correct else 'NO ❌'}")
    if not correct:
        print(f"  Baseline : FAKE (89.18% baseline) → FAKE (82.18% after dataset engineering)")
        print(f"  DistilBERT: {pred_label} ({confidence:.2f}%) — see model_comparison.md")
    if smoke_test:
        print(f"  *** {_SMOKE_NOTE} ***")

    return {
        "expected":   expected,
        "predicted":  pred_label,
        "confidence": round(confidence, 2),
        "correct":    correct,
        "smoke_test_note": _SMOKE_NOTE if smoke_test else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DistilBERT for VeriNews AI.")
    p.add_argument(
        "--checkpoint", type=str,
        default=None,
        help="Path to the saved HuggingFace checkpoint directory."
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Flag as smoke test — labels output accordingly."
    )
    p.add_argument(
        "--max-seq-len", type=int, default=None,
        help="Override max_seq_len (default: read from training_config.json)."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    smoke_test = args.smoke_test

    # Resolve checkpoint
    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpt = str(DISTILBERT_DIR / ("smoke_test" if smoke_test else "best"))

    tokenizer, model, device = load_checkpoint(ckpt)
    max_length = args.max_seq_len or get_max_length_from_config()

    if smoke_test:
        print(f"\n{'='*62}")
        print(f"  {_SMOKE_NOTE}")
        print(f"{'='*62}")

    evaluate_test_split(tokenizer, model, device, max_length, smoke_test)
    evaluate_manual_validation(tokenizer, model, device, max_length, smoke_test)
    evaluate_nasa(tokenizer, model, device, max_length, smoke_test)

    print(f"\n[eval] Evaluation complete. Results → {DISTILBERT_RESULTS}")
    if smoke_test:
        print(f"[eval] Reminder: {_SMOKE_NOTE}")
