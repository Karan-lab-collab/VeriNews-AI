# -*- coding: utf-8 -*-
"""
config.py — Central configuration for the DistilBERT Milestone 2 pipeline.

All transformer modules import from here. Do not hardcode paths or
hyperparameters in individual scripts.

NOTE ON MAX_SEQ_LEN
-------------------
A preliminary value (128) is used for the local smoke test only.
The value for full GPU training is determined by token-length analysis
run via `dataset.py --analyze-lengths`. That analysis reports:
  - median token length
  - 90th / 95th percentile
  - maximum
The final value chosen for full training is recorded in
`results/distilbert_v1/experiment_metadata.json`.

Do NOT change FULL_MAX_SEQ_LEN without re-running the analysis and
updating experiment_metadata.json.
"""
from pathlib import Path

# ── Root directories (derived from project layout) ───────────────────────────
_THIS_FILE   = Path(__file__).resolve()
BACKEND_DIR  = _THIS_FILE.parent.parent.parent          # .../backend/
TRAIN_DIR    = BACKEND_DIR / "train"
DATASETS_DIR = BACKEND_DIR / "datasets"
PROCESSED_DIR = DATASETS_DIR / "processed"

# ── Baseline dataset (produced by Milestone 1B data_engineering.py) ──────────
UNIFIED_CSV  = PROCESSED_DIR / "unified_dataset.csv"

# ── Frozen split directory ────────────────────────────────────────────────────
# Split CSVs and the manifest are written once and never regenerated
# unless the source dataset or seed changes.
SPLIT_DIR     = PROCESSED_DIR / "distilbert_v1"
TRAIN_CSV     = SPLIT_DIR / "train.csv"
VAL_CSV       = SPLIT_DIR / "validation.csv"
TEST_CSV      = SPLIT_DIR / "test.csv"
SPLIT_MANIFEST = SPLIT_DIR / "split_manifest.json"

# ── Model artefacts ───────────────────────────────────────────────────────────
MODELS_DIR          = BACKEND_DIR / "saved_models"
DISTILBERT_DIR      = MODELS_DIR / "distilbert_v1"       # HF save_pretrained dir

# ── Results ───────────────────────────────────────────────────────────────────
RESULTS_DIR         = BACKEND_DIR / "results"
DISTILBERT_RESULTS  = RESULTS_DIR / "distilbert_v1"

METRICS_JSON            = DISTILBERT_RESULTS / "metrics.json"
CLASS_REPORT_TXT        = DISTILBERT_RESULTS / "classification_report.txt"
CONFUSION_MATRIX_PNG    = DISTILBERT_RESULTS / "confusion_matrix.png"
TRAINING_CONFIG_JSON    = DISTILBERT_RESULTS / "training_config.json"
EXPERIMENT_METADATA_JSON = DISTILBERT_RESULTS / "experiment_metadata.json"
MANUAL_VAL_CSV          = DISTILBERT_RESULTS / "manual_validation_distilbert.csv"
MANUAL_VAL_REPORT       = DISTILBERT_RESULTS / "manual_validation_distilbert_report.md"
MANUAL_VAL_CM_PNG       = DISTILBERT_RESULTS / "manual_confusion_matrix_distilbert.png"
TOKEN_STATS_JSON        = DISTILBERT_RESULTS / "token_length_stats.json"

# ── Existing manual validation set (UNCHANGED from Milestone 1.5) ─────────────
VALIDATION_CSV  = BACKEND_DIR / "validation" / "validation.csv"

# ── Existing baseline results (read-only, for comparison report) ───────────────
BASELINE_RESULTS = RESULTS_DIR   # flat; individual files read by name

# ── Model checkpoint ─────────────────────────────────────────────────────────
MODEL_CHECKPOINT = "distilbert-base-uncased"

# ── Label convention (confirmed from unified_dataset.csv inspection) ──────────
# label column dtype: int64
# 0 = FAKE, 1 = REAL  (matches train/config.py LABEL_FAKE / LABEL_REAL)
LABEL_FAKE = 0
LABEL_REAL = 1
LABEL_NAMES   = {LABEL_FAKE: "FAKE", LABEL_REAL: "REAL"}
ID2LABEL      = {LABEL_FAKE: "FAKE", LABEL_REAL: "REAL"}
LABEL2ID      = {"FAKE": LABEL_FAKE, "REAL": LABEL_REAL}

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42          # matches baseline for comparability

# ── Split ratios ─────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10        # must sum to 1.0

# ── Sequence length ───────────────────────────────────────────────────────────
# SMOKE TEST: deliberately short to keep CPU run fast.
# FULL TRAIN: set after token-length analysis (see dataset.py --analyze-lengths).
SMOKE_MAX_SEQ_LEN = 128   # used only during --smoke-test

# Evidence-based MAX_SEQ_LEN for full GPU training.
# Source: results/distilbert_v1/token_length_stats.json (10,000 training articles)
#
# Token-length distribution (DistilBERT tokenizer, incl. [CLS]+[SEP]):
#   Mean    : 456.3  tokens
#   Median  : 414.0  tokens
#   p75     : 578.0  tokens
#   p90     : 823.1  tokens
#   p95     : 997.0  tokens
#   p99     : 1585.0 tokens
#   Max     : 5919   tokens
#
# Truncation trade-off (fraction of articles fully covered without truncation):
#   max_length=128 → 14.5%   (85.5% of articles truncated)
#   max_length=256 → 26.3%   (73.7% truncated)
#   max_length=384 → 44.2%   (55.8% truncated)
#   max_length=512 → 67.7%   (32.3% truncated)
#
# Compute/memory trade-off:
#   Self-attention cost grows approximately O(n²) with sequence length.
#   Increasing max_length from 256 to 512 roughly quadruples attention memory
#   and significantly increases per-step training time on GPU.
#   This means a shorter max_length enables larger effective batch sizes,
#   faster steps, and lower OOM risk — at the cost of more truncation.
#
# Decision: FULL_MAX_SEQ_LEN = 512 (for the first full experiment).
#   The unified dataset contains long-form articles with a median token length
#   of 414. A 256-token limit would truncate approximately 73.7% of sampled
#   articles, whereas 512 tokens fully cover approximately 67.7%. Because
#   standard DistilBERT supports at most 512 positional tokens, 512 was
#   selected for the first full experiment to preserve the maximum available
#   article context. This increases compute and memory cost and still truncates
#   approximately 32.3% of sampled articles.
#
#   Truncation strategy: from the END of each article (HuggingFace default).
#   This is motivated by the journalistic inverted-pyramid convention, where
#   headline, lead, and key claims appear early in the text. This is a design
#   ASSUMPTION for the first experiment, not a finding from VeriNews data.
#   It has NOT been independently validated on the VeriNews dataset.
#   Future experiments should explore head+tail truncation strategies.
#
#   If CUDA OOM occurs at batch_size=8 and max_length=512, reduce max_length
#   to 384 as a fallback and document the trade-off in model_comparison.md.
#
FULL_MAX_SEQ_LEN = 512    # justified by token-length analysis; see notes above

# ── Training hyperparameters ─────────────────────────────────────────────────
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 0.01
NUM_EPOCHS     = 3

# GPU training batch configuration (conservative for T4 15 GB at max_length=512)
# Physical batch size is intentionally small to avoid OOM.
# Effective batch size = TRAIN_BATCH × GRAD_ACCUM = 8 × 2 = 16.
TRAIN_BATCH        = 8    # per_device_train_batch_size
EVAL_BATCH         = 8    # per_device_eval_batch_size
GRAD_ACCUM_STEPS   = 2    # gradient_accumulation_steps
EFFECTIVE_BATCH    = TRAIN_BATCH * GRAD_ACCUM_STEPS   # = 16

# fp16 mixed precision: enabled when CUDA is available.
# Do NOT use fp16 on CPU (no benefit; may cause instability).
USE_FP16_IF_CUDA = True

WARMUP_RATIO     = 0.06   # ~6 % of total optimizer steps
METRIC_FOR_BEST  = "eval_f1"  # checkpoint saved on best validation F1

# ── Smoke test ────────────────────────────────────────────────────────────────
SMOKE_SAMPLES  = 200      # class-balanced (100 FAKE + 100 REAL)
SMOKE_EPOCHS   = 1
SMOKE_BATCH    = 8

# ── Ensure output directories exist on import ─────────────────────────────────
for _d in (SPLIT_DIR, DISTILBERT_DIR, DISTILBERT_RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# ── NASA article (identical to baseline for case-study comparability) ─────────
NASA_ARTICLE = (
    "Scientists at NASA announced today that the James Webb Space Telescope "
    "has discovered new details about a distant exoplanet's atmosphere after "
    "months of observation."
)
