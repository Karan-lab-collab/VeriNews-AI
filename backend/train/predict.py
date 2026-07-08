"""
predict.py – CLI prediction tool for VeriNews AI.

Usage (from backend/):
    python train/predict.py
    python train/predict.py --text "Your article headline or body here."

The script loads the saved model and vectorizer, cleans the input text,
and prints the predicted label (FAKE / REAL) with a confidence score.
"""
import sys
import argparse
from pathlib import Path

# Allow running as a top-level script from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from train.config import LABEL_NAMES, MODEL_PATH, VECTORIZER_PATH
from train.features import load_vectorizer
from train.preprocess import clean_text


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """
    Load the trained Logistic Regression model from disk.

    Returns
    -------
    sklearn estimator
        The fitted model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run 'python train/train.py' first."
        )
    return joblib.load(MODEL_PATH)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(text: str) -> dict:
    """
    Predict whether a piece of text is fake or real news.

    Parameters
    ----------
    text : str
        Raw article title, headline, or body text.

    Returns
    -------
    dict
        {
          "label":      "FAKE" or "REAL",
          "confidence": float between 0.0 and 1.0,
          "label_id":   0 (FAKE) or 1 (REAL),
        }
    """
    model      = load_model()
    vectorizer = load_vectorizer()

    cleaned    = clean_text(text)
    if not cleaned:
        return {"label": "UNKNOWN", "confidence": 0.0, "label_id": -1,
                "warning": "Input text is empty after cleaning."}

    X          = vectorizer.transform([cleaned])
    label_id   = int(model.predict(X)[0])
    proba      = model.predict_proba(X)[0]
    confidence = float(proba[label_id])

    return {
        "label":      LABEL_NAMES[label_id],
        "confidence": round(confidence, 4),
        "label_id":   label_id,
    }


# ── Interactive / CLI mode ────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    label      = result["label"]
    confidence = result["confidence"]
    bar_fill   = int(confidence * 30)
    bar        = "█" * bar_fill + "░" * (30 - bar_fill)

    colour = "\033[91m" if label == "FAKE" else "\033[92m"  # red / green
    reset  = "\033[0m"

    print("\n" + "─" * 50)
    print(f"  Prediction  : {colour}{label}{reset}")
    print(f"  Confidence  : {confidence:.2%}")
    print(f"  [{bar}]")
    print("─" * 50 + "\n")


def run_cli() -> None:
    """Parse CLI arguments and run prediction."""
    parser = argparse.ArgumentParser(
        description="VeriNews AI — Fake News Predictor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Text to classify. If omitted, enters interactive mode.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("  VeriNews AI — Fake News Predictor")
    print("=" * 50)

    if args.text:
        result = predict(args.text)
        _print_result(result)
        return

    # Interactive loop
    print("  Type article text and press Enter to classify.")
    print("  Type 'quit' or press Ctrl+C to exit.\n")
    while True:
        try:
            user_input = input("  Enter text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("\nGoodbye!")
            break
        if not user_input:
            print("  (empty input — please enter some text)\n")
            continue

        result = predict(user_input)
        _print_result(result)


if __name__ == "__main__":
    run_cli()
