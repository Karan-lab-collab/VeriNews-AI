# -*- coding: utf-8 -*-
"""
predict_distilbert.py — Single-text inference with the fine-tuned DistilBERT.

Usage (from backend/):
    python train/transformer/predict_distilbert.py --text "Your article text here"

Or import the predict() function into other modules.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from train.transformer.config import (
    DISTILBERT_DIR, DISTILBERT_RESULTS, ID2LABEL, LABEL_FAKE, LABEL_REAL,
)


def load_model(checkpoint_dir: str | Path | None = None):
    """
    Load tokenizer + model from a checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : str or Path, optional
        Path to a HuggingFace save_pretrained directory.
        Defaults to `saved_models/distilbert_v1/best`.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    path = Path(checkpoint_dir) if checkpoint_dir else DISTILBERT_DIR / "best"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run train_distilbert.py --smoke-test or full GPU training first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model     = AutoModelForSequenceClassification.from_pretrained(str(path))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict(
    text: str,
    tokenizer=None,
    model=None,
    device=None,
    checkpoint_dir: str | Path | None = None,
    max_length: int = 256,
) -> dict:
    """
    Classify a single article text as FAKE or REAL.

    Parameters
    ----------
    text : str
        The article text to classify.
    tokenizer, model, device : optional
        Pre-loaded objects. If None, the checkpoint is loaded automatically.
    checkpoint_dir : str or Path, optional
        Checkpoint to load if tokenizer/model are not supplied.
    max_length : int
        Max token sequence length (should match training config).

    Returns
    -------
    dict with keys: label (str), label_id (int), confidence (float 0-1),
                    p_fake (float), p_real (float).
    """
    import torch
    import torch.nn.functional as F

    if tokenizer is None or model is None or device is None:
        tokenizer, model, device = load_model(checkpoint_dir)

    enc = tokenizer(
        str(text),
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    probs    = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_id  = int(probs.argmax())
    label    = ID2LABEL[pred_id]
    conf     = float(probs[pred_id])

    return {
        "label":      label,
        "label_id":   pred_id,
        "confidence": round(conf, 4),
        "p_fake":     round(float(probs[LABEL_FAKE]), 4),
        "p_real":     round(float(probs[LABEL_REAL]), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify a news article with the fine-tuned DistilBERT model."
    )
    p.add_argument("--text",       type=str, required=True, help="Article text to classify.")
    p.add_argument("--checkpoint", type=str, default=None,  help="Checkpoint directory path.")
    p.add_argument("--max-seq-len",type=int, default=256,   help="Max token sequence length.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = predict(args.text, checkpoint_dir=args.checkpoint, max_length=args.max_seq_len)

    bar_len = int(result["confidence"] * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)

    print()
    print("─" * 52)
    print(f"  Prediction  : {result['label']}")
    print(f"  Confidence  : {result['confidence']*100:.2f}%")
    print(f"  [{bar}]")
    print(f"  P(FAKE)     : {result['p_fake']*100:.2f}%")
    print(f"  P(REAL)     : {result['p_real']*100:.2f}%")
    print("─" * 52)
