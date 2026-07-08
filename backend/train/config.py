"""
config.py – Central configuration for paths and hyperparameters.

All other modules import from here — no hardcoded paths elsewhere.
"""
from pathlib import Path

# ── Root directories ──────────────────────────────────────────────────────────
BACKEND_DIR   = Path(__file__).resolve().parent.parent
DATASETS_DIR  = BACKEND_DIR / "datasets"
RAW_DIR       = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"

TRAIN_DIR      = BACKEND_DIR / "train"
MODELS_DIR     = BACKEND_DIR / "saved_models"
RESULTS_DIR    = BACKEND_DIR / "results"
NOTEBOOKS_DIR  = BACKEND_DIR / "notebooks"
FIGURES_DIR    = BACKEND_DIR.parent / "docs" / "figures"

# ── Dataset files ─────────────────────────────────────────────────────────────
FAKE_CSV      = RAW_DIR / "Fake.csv"
TRUE_CSV      = RAW_DIR / "True.csv"
COMBINED_CSV  = PROCESSED_DIR / "combined_cleaned.csv"

# ── Saved artefacts ───────────────────────────────────────────────────────────
MODEL_PATH      = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"

# ── Results ───────────────────────────────────────────────────────────────────
METRICS_PATH     = RESULTS_DIR / "metrics.json"
REPORT_PATH      = RESULTS_DIR / "report.txt"
CONF_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"

# ── Milestone 1.5 — Validation paths ─────────────────────────────────────────
VALIDATION_DIR      = BACKEND_DIR / "validation"
VALIDATION_CSV      = VALIDATION_DIR / "validation.csv"
VAL_REPORT_PATH     = RESULTS_DIR / "manual_validation_report.md"
VAL_CSV_PATH        = RESULTS_DIR / "manual_validation.csv"
VAL_CM_PATH         = RESULTS_DIR / "manual_confusion_matrix.png"
FI_CSV_PATH         = RESULTS_DIR / "feature_importance.csv"

# ── Milestone 1B — Dataset engineering paths ──────────────────────────────────
UNIFIED_CSV          = PROCESSED_DIR / "unified_dataset.csv"
DATASET_SOURCES_DIR  = RAW_DIR / "sources"          # per-source CSVs before merging
BASELINE_MODEL_PATH  = MODELS_DIR / "model_baseline.pkl"
BASELINE_VEC_PATH    = MODELS_DIR / "vectorizer_baseline.pkl"
BASELINE_METRICS     = RESULTS_DIR / "metrics_baseline.json"
V2_METRICS_PATH      = RESULTS_DIR / "metrics_v2.json"
V2_REPORT_PATH       = RESULTS_DIR / "report_v2.txt"
V2_CM_PATH           = RESULTS_DIR / "confusion_matrix_v2.png"
V2_VAL_CSV_PATH      = RESULTS_DIR / "manual_validation_v2.csv"
V2_FI_CSV_PATH       = RESULTS_DIR / "feature_importance_v2.csv"
V2_VAL_CM_PATH       = RESULTS_DIR / "manual_confusion_matrix_v2.png"
GEN_REPORT_PATH      = RESULTS_DIR / "domain_generalization_report.md"
DATASET_RESEARCH_DOC = BACKEND_DIR.parent / "docs" / "dataset_research.md"

# ── Hyperparameters ───────────────────────────────────────────────────────────
RANDOM_SEED    = 42
TEST_SIZE      = 0.20          # 80/20 train-test split
MAX_FEATURES   = 50_000        # TF-IDF vocabulary ceiling
NGRAM_RANGE    = (1, 2)        # Unigrams + bigrams
MIN_DF         = 2             # Ignore terms appearing < 2 times

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_FAKE = 0
LABEL_REAL = 1
LABEL_NAMES = {LABEL_FAKE: "FAKE", LABEL_REAL: "REAL"}

# ── Ensure directories exist ──────────────────────────────────────────────────
for _dir in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
             FIGURES_DIR, NOTEBOOKS_DIR, DATASET_SOURCES_DIR,
             VALIDATION_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
