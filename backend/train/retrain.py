# -*- coding: utf-8 -*-
"""
retrain.py – Milestone 1B: Retrain TF-IDF + Logistic Regression on the
unified multi-source dataset and generate the domain generalisation report.

Usage (from backend/):
    python train/retrain.py

What this does:
  1. Backs up existing baseline model and metrics.
  2. Loads unified_dataset.csv (built by data_engineering.py).
  3. Retrains the SAME LogisticRegression + TF-IDF pipeline on the new data.
  4. Evaluates on the train/test split (v2 metrics).
  5. Re-runs the manual validation (all 60 examples).
  6. Extracts feature importance for the new model.
  7. Re-analyses the NASA article.
  8. Generates: backend/results/domain_generalization_report.md

NOTE: model.pkl and vectorizer.pkl are REPLACED by the retrained versions.
      Originals are backed up as model_baseline.pkl / vectorizer_baseline.pkl.
"""

import sys
import json
import time
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from train.config import (
    BACKEND_DIR,
    LABEL_FAKE, LABEL_REAL, LABEL_NAMES,
    MODEL_PATH, VECTORIZER_PATH,
    BASELINE_MODEL_PATH, BASELINE_VEC_PATH, BASELINE_METRICS,
    UNIFIED_CSV,
    METRICS_PATH, RESULTS_DIR,
    V2_METRICS_PATH, V2_CM_PATH, V2_REPORT_PATH,
    V2_VAL_CSV_PATH, V2_FI_CSV_PATH, V2_VAL_CM_PATH,
    GEN_REPORT_PATH,
    FI_CSV_PATH, VAL_CSV_PATH,
    VALIDATION_CSV,
    RANDOM_SEED, TEST_SIZE, MAX_FEATURES, NGRAM_RANGE, MIN_DF,
)
from train.features import build_vectorizer, fit_vectorizer, load_vectorizer, save_vectorizer
from train.preprocess import clean_text

NASA_ARTICLE = (
    "Scientists at NASA announced today that the James Webb Space Telescope "
    "has discovered new details about a distant exoplanet's atmosphere after "
    "months of observation."
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Backup baseline
# ─────────────────────────────────────────────────────────────────────────────

def backup_baseline() -> None:
    """Copy existing model.pkl + vectorizer.pkl + metrics.json to *_baseline."""
    print("[retrain] Backing up baseline artefacts ...")
    for src, dst in [
        (MODEL_PATH,      BASELINE_MODEL_PATH),
        (VECTORIZER_PATH, BASELINE_VEC_PATH),
        (METRICS_PATH,    BASELINE_METRICS),
    ]:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"[retrain]   {src.name} -> {dst.name}")
        else:
            print(f"[retrain]   WARNING: {src.name} not found — skipping backup")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load unified dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_unified() -> pd.DataFrame:
    if not UNIFIED_CSV.exists():
        raise FileNotFoundError(
            f"Unified dataset not found: {UNIFIED_CSV}\n"
            "Run 'python train/data_engineering.py' first."
        )
    print(f"[retrain] Loading unified dataset <- {UNIFIED_CSV}")
    df = pd.read_csv(UNIFIED_CSV, encoding="utf-8")
    print(f"[retrain]   {len(df):,} rows  |  "
          f"FAKE={int((df['label']==LABEL_FAKE).sum()):,}  "
          f"REAL={int((df['label']==LABEL_REAL).sum()):,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train / evaluate on split
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> tuple:
    """
    Split, vectorise, train, evaluate — identical pipeline to Milestone 1.

    Returns
    -------
    model, vectorizer, metrics_dict, y_test, y_pred, X_test_raw
    """
    print("\n[retrain] Splitting dataset ...")
    X_text = df["text"].tolist()
    y      = df["label"].tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_text, y,
        test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    print(f"[retrain]   Train: {len(X_train_raw):,}  Test: {len(X_test_raw):,}")

    # Vectorise
    print("[retrain] Fitting TF-IDF vectorizer ...")
    vectorizer = build_vectorizer()                     # create unfitted instance
    X_train    = fit_vectorizer(vectorizer, X_train_raw)  # fit + transform train
    X_test     = vectorizer.transform(X_test_raw)
    print(f"[retrain]   Vocabulary size: {len(vectorizer.get_feature_names_out()):,}")

    # Train
    print("[retrain] Training Logistic Regression ...")
    t0 = time.perf_counter()
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
        C=1.0,
    )
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"[retrain]   Training complete in {elapsed:.1f}s")

    # Evaluate
    print("\n[retrain] Evaluating on test set ...")
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    print("\n" + "=" * 60)
    print("  Train/Test Evaluation (v2)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k.capitalize():<14} {v:.4f}  ({v*100:.2f}%)")
    report_str = classification_report(
        y_test, y_pred, target_names=["FAKE", "REAL"]
    )
    print("\n" + report_str)

    return model, vectorizer, metrics, y_test, y_pred, X_test_raw


def save_cm(y_test, y_pred, path: Path, title: str) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"],
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[retrain]   Confusion matrix -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Manual validation (re-run)
# ─────────────────────────────────────────────────────────────────────────────

def _label_to_int(label: str) -> int:
    return LABEL_REAL if label.upper() == "REAL" else LABEL_FAKE


def run_manual_validation(model, vectorizer) -> pd.DataFrame:
    """Re-evaluate every example in validation.csv with the new model."""
    print("\n[retrain] Re-running manual validation ...")
    val_df = pd.read_csv(VALIDATION_CSV, encoding="utf-8")
    val_df.columns = [c.strip() for c in val_df.columns]
    val_df["text"]     = val_df["text"].str.strip()
    val_df["label"]    = val_df["label"].str.strip().str.upper()
    val_df["category"] = val_df["category"].str.strip()

    cleaned   = [clean_text(t) for t in val_df["text"]]
    X         = vectorizer.transform(cleaned)
    ids       = model.predict(X)
    probas    = model.predict_proba(X)

    val_df["predicted"]  = [LABEL_NAMES[int(i)] for i in ids]
    val_df["confidence"] = [round(float(probas[i][int(ids[i])]) * 100, 2)
                            for i in range(len(ids))]
    val_df["correct"]    = val_df["label"] == val_df["predicted"]

    # Print
    print(f"\n  {'#':<4}{'Category':<18}{'Expected':<10}{'Predicted':<10}{'Conf%':<8}{'OK?'}")
    print("  " + "-" * 56)
    for i, row in val_df.iterrows():
        tick = "+" if row["correct"] else "x"
        print(f"  {i+1:<4}{row['category']:<18}{row['label']:<10}"
              f"{row['predicted']:<10}{row['confidence']:<8.1f}[{tick}]")

    acc = val_df["correct"].mean()
    print(f"\n  Overall manual validation accuracy: {acc*100:.1f}%")

    return val_df


def print_category_metrics(val_df: pd.DataFrame) -> dict:
    cat_stats = {}
    print("\n" + "=" * 60)
    print("  Per-Category Accuracy (v2)")
    print("=" * 60)
    for cat, g in sorted(
        val_df.groupby("category"),
        key=lambda x: -x[1]["correct"].mean()
    ):
        correct = g["correct"].sum()
        total   = len(g)
        acc     = correct / total * 100
        cat_stats[cat] = {"correct": int(correct), "total": int(total), "accuracy": round(acc, 1)}
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  {cat:<20} [{bar}]  {acc:>5.1f}%  ({correct}/{total})")
    return cat_stats


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature importance (v2)
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_importance(model, vectorizer, top_n: int = 30) -> pd.DataFrame:
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs         = model.coef_[0]

    real_idx = np.argsort(coefs)[::-1][:top_n]
    fake_idx = np.argsort(coefs)[:top_n]

    real_df = pd.DataFrame({"feature": feature_names[real_idx],
                             "coefficient": coefs[real_idx].round(4),
                             "direction": "REAL"})
    fake_df = pd.DataFrame({"feature": feature_names[fake_idx],
                             "coefficient": coefs[fake_idx].round(4),
                             "direction": "FAKE"})
    fi_df = pd.concat([real_df, fake_df], ignore_index=True)

    print(f"\n  Top 10 REAL features (v2):")
    for _, r in real_df.head(10).iterrows():
        print(f"    {r['feature']:<35} {r['coefficient']:+.4f}")
    print(f"\n  Top 10 FAKE features (v2):")
    for _, r in fake_df.head(10).iterrows():
        print(f"    {r['feature']:<35} {r['coefficient']:+.4f}")

    fi_df.to_csv(V2_FI_CSV_PATH, index=False, encoding="utf-8")
    print(f"\n[retrain]   Feature importance v2 -> {V2_FI_CSV_PATH}")
    return fi_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. NASA article analysis (v2)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_nasa(model, vectorizer) -> dict:
    cleaned  = clean_text(NASA_ARTICLE)
    x        = vectorizer.transform([cleaned])
    label_id = int(model.predict(x)[0])
    proba    = model.predict_proba(x)[0]
    label    = LABEL_NAMES[label_id]
    conf     = float(proba[label_id]) * 100

    print("\n" + "=" * 60)
    print("  NASA/James Webb Article — v2 Model")
    print("=" * 60)
    print(f"  Predicted : {label}  ({conf:.2f}%)")
    print(f"  P(FAKE)={proba[0]*100:.2f}%   P(REAL)={proba[1]*100:.2f}%")

    return {
        "predicted": label,
        "confidence": round(conf, 2),
        "p_fake": round(float(proba[0]) * 100, 2),
        "p_real": round(float(proba[1]) * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Generate domain generalisation report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    baseline_metrics: dict,
    v2_metrics: dict,
    baseline_val_df: pd.DataFrame,
    v2_val_df: pd.DataFrame,
    baseline_fi: pd.DataFrame,
    v2_fi: pd.DataFrame,
    v2_cat_stats: dict,
    nasa_v1: dict,
    nasa_v2: dict,
    unified_stats: dict,
) -> None:
    """Write the comprehensive domain generalisation report."""

    def _delta(v2, v1, pct=True):
        d = v2 - v1
        s = f"+{d:.2f}" if d >= 0 else f"{d:.2f}"
        return f"{s}%" if pct else s

    # Baseline category stats
    base_cat = {}
    for cat, g in baseline_val_df.groupby("category"):
        base_cat[cat] = round(g["correct"].mean() * 100, 1)

    # v2 category stats
    v2_cat_acc = {cat: s["accuracy"] for cat, s in v2_cat_stats.items()}

    # All categories
    all_cats = sorted(set(list(base_cat.keys()) + list(v2_cat_acc.keys())))

    cat_rows = "\n".join(
        f"| {cat} | {base_cat.get(cat, 0):.1f}% | {v2_cat_acc.get(cat, 0):.1f}% | "
        f"{_delta(v2_cat_acc.get(cat, 0), base_cat.get(cat, 0))} |"
        for cat in all_cats
    )

    # Feature comparison
    b_top5_real = " | ".join(
        baseline_fi[baseline_fi.direction == "REAL"]["feature"].head(5).tolist()
    )
    v2_top5_real = " | ".join(
        v2_fi[v2_fi.direction == "REAL"]["feature"].head(5).tolist()
    )
    b_top5_fake = " | ".join(
        baseline_fi[baseline_fi.direction == "FAKE"]["feature"].head(5).tolist()
    )
    v2_top5_fake = " | ".join(
        v2_fi[v2_fi.direction == "FAKE"]["feature"].head(5).tolist()
    )

    # Reuters coefficient comparison
    b_reuters = baseline_fi[baseline_fi.feature == "reuters"]["coefficient"]
    v2_reuters = v2_fi[v2_fi.feature == "reuters"]["coefficient"]
    reuters_b = float(b_reuters.iloc[0]) if len(b_reuters) > 0 else 0
    reuters_v2 = float(v2_reuters.iloc[0]) if len(v2_reuters) > 0 else 0

    # Biggest improvements and still-failing
    improvements = sorted(
        [(c, v2_cat_acc.get(c, 0) - base_cat.get(c, 0)) for c in all_cats],
        key=lambda x: -x[1]
    )
    regressions = [(c, d) for c, d in improvements if d < 0]
    improvements_pos = [(c, d) for c, d in improvements if d > 0]

    base_val_acc  = baseline_val_df["correct"].mean() * 100
    v2_val_acc    = v2_val_df["correct"].mean() * 100

    # Conclusion logic
    improved     = v2_val_acc > base_val_acc
    reuters_down = reuters_v2 < reuters_b
    needs_bert   = v2_val_acc < 75.0

    report = f"""# VeriNews AI — Domain Generalisation Report (Milestone 1B)

> **Experiment**: Dataset Engineering — Replace narrow ISOT with unified multi-source dataset
> **Model**: Same TF-IDF + Logistic Regression (unchanged)
> **Hypothesis**: Domain bias is caused by training data, not algorithm

---

## 1. What Was Changed

| Change | Description | Rationale |
|--------|-------------|-----------|
| Publisher datelines removed | Stripped "CITY (Reuters) -" prefixes | Eliminate `reuters` coefficient +21.7 leakage |
| Added LIAR dataset | 12k PolitiFact statements (binarised) | Adds diverse text styles |
| Added WELFake (if available) | 72k multi-source articles | Adds domain breadth |
| Deduplication across sources | MD5 hash on cleaned text | Prevent training contamination |
| LIAR label binarisation | true/mostly-true=REAL; false/pants-fire=FAKE; skip ambiguous | Clean training signal |

### Dataset Composition

| Source | Articles |
|--------|---------|
{chr(10).join(f"| {src} | {cnt:,} |" for src, cnt in unified_stats.items())}
| **Total** | **{sum(unified_stats.values()):,}** |

---

## 2. Train/Test Split Metrics: Before vs After

| Metric | Baseline (v1) | After Engineering (v2) | Delta |
|--------|--------------|----------------------|-------|
| Accuracy  | {baseline_metrics['accuracy']*100:.2f}% | {v2_metrics['accuracy']*100:.2f}% | {_delta(v2_metrics['accuracy']*100, baseline_metrics['accuracy']*100)} |
| Precision | {baseline_metrics['precision']*100:.2f}% | {v2_metrics['precision']*100:.2f}% | {_delta(v2_metrics['precision']*100, baseline_metrics['precision']*100)} |
| Recall    | {baseline_metrics['recall']*100:.2f}% | {v2_metrics['recall']*100:.2f}% | {_delta(v2_metrics['recall']*100, baseline_metrics['recall']*100)} |
| F1-Score  | {baseline_metrics['f1']*100:.2f}% | {v2_metrics['f1']*100:.2f}% | {_delta(v2_metrics['f1']*100, baseline_metrics['f1']*100)} |

---

## 3. Manual Validation Accuracy: Before vs After

| Validation Set | Baseline (v1) | After Engineering (v2) | Delta |
|---------------|--------------|----------------------|-------|
| Overall (60 examples) | {base_val_acc:.1f}% | {v2_val_acc:.1f}% | {_delta(v2_val_acc, base_val_acc)} |

### Category-wise Performance

| Category | Baseline | v2 | Delta |
|----------|----------|----|-------|
{cat_rows}

---

## 4. NASA / James Webb Article Analysis

| Metric | Baseline (v1) | After Engineering (v2) |
|--------|--------------|----------------------|
| Predicted | {nasa_v1['predicted']} | {nasa_v2['predicted']} |
| Confidence | {nasa_v1['confidence']}% | {nasa_v2['confidence']}% |
| P(FAKE) | {nasa_v1['p_fake']}% | {nasa_v2['p_fake']}% |
| P(REAL) | {nasa_v1['p_real']}% | {nasa_v2['p_real']}% |

{"**Result**: Model now correctly classifies the NASA article as REAL." if nasa_v2['predicted'] == 'REAL' else "**Result**: Model still misclassifies the NASA article as FAKE. The domain bias persists despite dataset engineering."}

---

## 5. Feature Importance: Reuters Bias Analysis

| | Baseline | v2 |
|-|----------|----|
| Top 5 REAL features | `{b_top5_real}` | `{v2_top5_real}` |
| Top 5 FAKE features | `{b_top5_fake}` | `{v2_top5_fake}` |
| `reuters` coefficient | {reuters_b:+.4f} | {reuters_v2:+.4f} |

{"**Reuters bias ELIMINATED** — the `reuters` token no longer dominates REAL predictions." if reuters_v2 < 5.0 else f"**Reuters bias REDUCED** from +{reuters_b:.1f} to +{reuters_v2:.1f} — but still present." if reuters_v2 < reuters_b else "**Reuters bias UNCHANGED** — dateline removal had insufficient effect."}

---

## 6. Key Findings

### Did dataset engineering improve manual validation?

{"**YES** — Manual validation accuracy improved from {:.1f}% to {:.1f}% (+{:.1f} percentage points). Dataset engineering was effective.".format(base_val_acc, v2_val_acc, v2_val_acc - base_val_acc) if improved else "**PARTIALLY** — Manual validation accuracy changed from {:.1f}% to {:.1f}%. The improvement was limited.".format(base_val_acc, v2_val_acc)}

### Did the Reuters bias decrease?

{"**YES** — The `reuters` coefficient dropped from {:.1f} to {:.1f}. Publisher dateline removal was effective.".format(reuters_b, reuters_v2) if reuters_v2 < reuters_b else "**NO** — The `reuters` coefficient remained at ~{:.1f}. The dateline removal regex may not have matched all instances.".format(reuters_v2)}

### What types of articles improved?

{chr(10).join(f"- **{cat}**: +{d:.1f}%" for cat, d in improvements_pos[:5]) if improvements_pos else "- No significant improvements detected."}

### What still fails?

{chr(10).join(f"- **{cat}**: {d:.1f}% change" for cat, d in regressions[:5]) if regressions else "- No significant regressions."}

Categories with <50% accuracy still present: {", ".join(c for c in all_cats if v2_cat_acc.get(c, 0) < 50)}

---

## 7. Should We Continue with Logistic Regression? Or DistilBERT?

### Analysis

| Question | Finding |
|----------|---------|
| Is the domain bias primarily from training data? | {"Yes — dataset engineering improved results" if improved else "Partially — training data was a factor but not the only cause"} |
| Can TF-IDF + LR reach acceptable accuracy (>75%) on diverse validation? | {"Yes, with more data" if v2_val_acc > 75 else "Unlikely — current ceiling appears below 75%"} |
| Is the improvement sufficient for production? | {"Yes" if v2_val_acc > 80 else "No — more work needed"} |

### Conclusion

{"**Continue with Logistic Regression** — Dataset engineering has pushed manual validation accuracy above 75%. The baseline is now acceptable for a v1 API with appropriate confidence thresholds." if v2_val_acc > 75 else "**Recommend DistilBERT for Milestone 2** — Despite dataset engineering, the TF-IDF + Logistic Regression model cannot overcome its fundamental bag-of-words limitation. Manual validation accuracy of {:.1f}% is insufficient for production use. The model lacks semantic understanding needed to distinguish legitimate science news from sensationalist fake news.".format(v2_val_acc)}

---

## 8. Recommendations for Next Milestone

### If Continuing with LR (Short-term)

1. **Class weighting** (`class_weight='balanced'`): Will improve recall for underrepresented classes.
2. **Larger vocabulary** (increase `MAX_FEATURES` to 100k): Captures more domain terms.
3. **Trigram support** (`ngram_range=(1,3)`): Captures "NASA announced", "peer-reviewed study".
4. **More diverse training data**: CoAID (health), McIntire (sports/entertainment), FeverFact.
5. **Probability calibration**: Platt scaling to make confidence scores meaningful.

### Milestone 2: DistilBERT Fine-tuning (Recommended)

1. Replace TF-IDF with contextual embeddings (DistilBERT encoder).
2. Fine-tune on the unified dataset with domain-balanced sampling.
3. Expected manual validation accuracy: 80-90% (based on published benchmarks).
4. Integration: FastAPI endpoint accepts text → returns label + confidence.

---

*Generated by VeriNews AI — Milestone 1B retrain.py*
"""

    GEN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GEN_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[retrain] Domain generalisation report -> {GEN_REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  VeriNews AI — Milestone 1B: Retrain on Unified Dataset")
    print("=" * 70)

    # Step 1: Backup baseline
    backup_baseline()

    # Step 2: Load baseline metrics and validation results for comparison
    print("\n[retrain] Loading baseline metrics for comparison ...")
    with open(BASELINE_METRICS if BASELINE_METRICS.exists() else METRICS_PATH,
              encoding="utf-8") as f:
        baseline_metrics = json.load(f)
    print(f"[retrain]   Baseline accuracy: {baseline_metrics['accuracy']*100:.2f}%")

    baseline_val_df = pd.read_csv(VAL_CSV_PATH, encoding="utf-8") \
        if VAL_CSV_PATH.exists() else pd.DataFrame()

    baseline_fi = pd.read_csv(FI_CSV_PATH, encoding="utf-8") \
        if FI_CSV_PATH.exists() else pd.DataFrame()

    baseline_nasa = {
        "predicted": "FAKE",
        "confidence": 89.18,
        "p_fake": 89.18,
        "p_real": 10.82,
    }

    # Step 3: Load unified dataset
    df = load_unified()

    unified_stats = df["source"].value_counts().to_dict() if "source" in df.columns else {}

    # Step 4: Retrain
    model, vectorizer, v2_metrics, y_test, y_pred, _ = train_and_evaluate(df)

    # Step 5: Save v2 artefacts
    print("\n[retrain] Saving v2 artefacts ...")
    joblib.dump(model, MODEL_PATH)
    save_vectorizer(vectorizer)
    with open(V2_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(v2_metrics, f, indent=2)
    # Also update the canonical metrics.json
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(v2_metrics, f, indent=2)
    print(f"[retrain]   model.pkl -> {MODEL_PATH}")
    print(f"[retrain]   vectorizer.pkl -> {VECTORIZER_PATH}")
    print(f"[retrain]   metrics_v2.json -> {V2_METRICS_PATH}")

    # Confusion matrix
    save_cm(y_test, y_pred, V2_CM_PATH,
            "VeriNews AI v2 — Train/Test Confusion Matrix")

    # Step 6: Manual validation
    v2_val_df = run_manual_validation(model, vectorizer)
    v2_cat_stats = print_category_metrics(v2_val_df)
    v2_val_df.to_csv(V2_VAL_CSV_PATH, index=False, encoding="utf-8")

    # Manual validation confusion matrix
    y_val_true = [LABEL_REAL if l == "REAL" else LABEL_FAKE for l in v2_val_df["label"]]
    y_val_pred = [LABEL_REAL if l == "REAL" else LABEL_FAKE for l in v2_val_df["predicted"]]
    save_cm(y_val_true, y_val_pred, V2_VAL_CM_PATH,
            "VeriNews AI v2 — Manual Validation Confusion Matrix")

    # Step 7: Feature importance
    v2_fi = extract_feature_importance(model, vectorizer, top_n=30)

    # Step 8: NASA article
    nasa_v2 = analyse_nasa(model, vectorizer)

    # Step 9: Generate report
    generate_report(
        baseline_metrics=baseline_metrics,
        v2_metrics=v2_metrics,
        baseline_val_df=baseline_val_df,
        v2_val_df=v2_val_df,
        baseline_fi=baseline_fi,
        v2_fi=v2_fi,
        v2_cat_stats=v2_cat_stats,
        nasa_v1=baseline_nasa,
        nasa_v2=nasa_v2,
        unified_stats=unified_stats,
    )

    # Final summary
    base_val_acc = baseline_val_df["correct"].mean() * 100 if len(baseline_val_df) > 0 else 0
    v2_val_acc   = v2_val_df["correct"].mean() * 100

    print("\n" + "=" * 70)
    print("  Milestone 1B Complete — Summary")
    print("=" * 70)
    print(f"  Train/Test accuracy:      {baseline_metrics['accuracy']*100:.2f}% -> {v2_metrics['accuracy']*100:.2f}%")
    print(f"  Manual validation:        {base_val_acc:.1f}% -> {v2_val_acc:.1f}%")
    print(f"  NASA article prediction:  baseline=FAKE -> v2={nasa_v2['predicted']}")
    print(f"\n  Reports saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
