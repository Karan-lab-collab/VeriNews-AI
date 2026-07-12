# -*- coding: utf-8 -*-
"""
train_distilbert.py — Hardware-aware DistilBERT fine-tuning entry point.

Usage (from backend/):
    # Local smoke test (CPU — verifies pipeline only; ~200 samples, 1 epoch)
    python train/transformer/train_distilbert.py --smoke-test

    # Full GPU training (must be run in a CUDA environment, e.g., Colab/Kaggle)
    python train/transformer/train_distilbert.py --max-seq-len 512

    # Full training with explicit config (recommended)
    python train/transformer/train_distilbert.py \\
        --max-seq-len 512 \\
        --epochs 3 \\
        --batch-size 8 \\
        --grad-accum 2 \\
        --lr 2e-5

HARDWARE GUARD
--------------
Full training on CPU is blocked automatically. The script detects the device
and refuses to proceed unless --smoke-test is specified or CUDA/MPS is available.
This prevents accidentally launching a multi-hour CPU run on the development machine.

PRECISION
---------
fp16 mixed precision is automatically enabled when CUDA is available.
It is NOT enabled on CPU (no benefit; can cause instability on CPU).
fp16 roughly halves GPU memory consumption and significantly speeds up
matrix operations on Tensor Core GPUs (T4, A100, V100).

GRADIENT ACCUMULATION
---------------------
The default configuration uses per_device_train_batch_size=8 and
gradient_accumulation_steps=2, giving an effective batch size of 16.
This conservative setting reduces OOM risk at max_length=512 on a T4 GPU.

CUDA OOM FALLBACK (strict protocol)
------------------------------------
If CUDA OOM occurs at batch_size=8:
  Step 1 → reduce batch_size to 4, increase grad_accum to 4.
            Effective batch stays 16. Eval batch may also be reduced to 4.
  Step 2 → If OOM persists at batch_size=4, STOP and report failure.
            Do NOT silently reduce max_seq_len from 512 to 384.
            Changing max_seq_len is a meaningful experiment configuration
            change that requires an explicit decision before continuing.

EXPERIMENT PROTOCOL — DistilBERT v1
-------------------------------------
  - Fixed 3 epochs. No early stopping.
  - Evaluate every epoch, save every epoch.
  - Best checkpoint selected by validation F1.
  - GPU training duration is unknown and will be measured at runtime.
    Training speed (samples/s, steps/s) and wall time are recorded in
    experiment_metadata.json and training_config.json.

LABEL SEMANTICS
---------------
label2id = {"FAKE": 0, "REAL": 1}
id2label = {0: "FAKE", 1: "REAL"}
These mappings are passed directly into the HuggingFace model config
(AutoModelForSequenceClassification), so the saved config.json preserves
the human-readable label names for future inference and FastAPI integration.

SMOKE TEST WARNING
------------------
Smoke-test metrics are labelled "SMOKE TEST — NOT A RESEARCH RESULT" and must
NOT be included in model_comparison.md or any research documentation.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from train.transformer.config import (
    MODEL_CHECKPOINT,
    RANDOM_SEED,
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    TRAIN_BATCH, EVAL_BATCH, GRAD_ACCUM_STEPS, USE_FP16_IF_CUDA,
    WARMUP_RATIO, METRIC_FOR_BEST,
    SMOKE_EPOCHS, SMOKE_BATCH, SMOKE_MAX_SEQ_LEN, SMOKE_SAMPLES,
    TRAIN_CSV, VAL_CSV, SPLIT_MANIFEST,
    DISTILBERT_DIR, DISTILBERT_RESULTS,
    TRAINING_CONFIG_JSON, EXPERIMENT_METADATA_JSON,
    TOKEN_STATS_JSON, ID2LABEL, LABEL2ID,
)
from train.transformer.dataset import (
    NewsDataset, prepare_splits, get_smoke_sample,
)

_SMOKE_BANNER = "=" * 70 + "\n  ⚠  SMOKE TEST — NOT A RESEARCH RESULT  ⚠\n" + "=" * 70


# ─────────────────────────────────────────────────────────────────────────────
# Device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    """Detect CUDA → MPS → CPU in priority order. Print selection."""
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] CUDA available — GPU : {name}  ({vram:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[device] Apple MPS available — using MPS.")
    else:
        dev = torch.device("cpu")
        import os
        cpu = platform.processor() or "unknown"
        print(f"[device] No GPU detected — using CPU ({cpu}).")
    return dev


def cpu_full_train_guard(device, smoke_test: bool) -> None:
    """Refuse full training on CPU; allow smoke test."""
    if str(device) == "cpu" and not smoke_test:
        print()
        print("=" * 70)
        print("  ERROR: Full DistilBERT training on CPU is not supported.")
        print()
        print("  This machine has no CUDA or MPS GPU. Running full training on")
        print("  CPU would take many hours and produce unreliable results.")
        print()
        print("  Options:")
        print("    1. Run the local smoke test:   --smoke-test")
        print("    2. Run full training on Colab: see notebooks/train_distilbert_colab.ipynb")
        print("=" * 70)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training configuration summary
# ─────────────────────────────────────────────────────────────────────────────

def print_training_config(
    device,
    max_seq_len: int,
    phys_batch: int,
    grad_accum: int,
    fp16: bool,
    smoke_test: bool,
    args: argparse.Namespace,
) -> None:
    """Print the full training configuration before any computation begins."""
    import torch
    effective_batch = phys_batch * grad_accum
    precision = "fp16 (mixed)" if fp16 else "fp32 (full)"

    sep = "─" * 60
    print(f"\n{sep}")
    if smoke_test:
        print("  SMOKE TEST configuration (NOT a research run)")
    else:
        print("  Full training configuration")
    print(f"{sep}")
    print(f"  Model checkpoint    : {MODEL_CHECKPOINT}")
    print(f"  Max sequence length : {max_seq_len} tokens")
    print(f"  Device              : {device}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU                 : {name} ({vram:.1f} GB VRAM)")
    print(f"  Precision           : {precision}")
    print(f"  Physical batch size : {phys_batch} (per device)")
    print(f"  Gradient accumulation: {grad_accum} steps")
    print(f"  Effective batch size : {effective_batch}")
    print(f"  Learning rate       : {args.lr}")
    print(f"  Weight decay        : {args.weight_decay}")
    print(f"  Epochs              : {args.epochs}")
    print(f"  Warmup ratio        : {WARMUP_RATIO}")
    print(f"  Best-model metric   : {METRIC_FOR_BEST}")
    print(f"  Label mapping       : {ID2LABEL}")
    print(f"{sep}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (computed outside Trainer for full control)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 for binary classification."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {"accuracy": acc, "precision": float(prec), "recall": float(rec), "f1": float(f1)}


# ─────────────────────────────────────────────────────────────────────────────
# Build experiment metadata record
# ─────────────────────────────────────────────────────────────────────────────

def build_metadata(
    device,
    max_seq_len: int,
    phys_batch: int,
    grad_accum: int,
    fp16: bool,
    smoke_test: bool,
    args: argparse.Namespace,
) -> dict:
    import torch

    token_stats = {}
    if TOKEN_STATS_JSON.exists():
        with open(TOKEN_STATS_JSON, encoding="utf-8") as f:
            token_stats = json.load(f)

    manifest = {}
    if SPLIT_MANIFEST.exists():
        with open(SPLIT_MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f)

    return {
        "run_type": "smoke_test" if smoke_test else "full_training",
        "model_checkpoint": MODEL_CHECKPOINT,
        "hardware": {
            "platform":      platform.platform(),
            "python":        sys.version,
            "torch_version": torch.__version__,
            "device":        str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gpu_vram_gb": (
                round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                if torch.cuda.is_available() else None
            ),
        },
        "dataset": {
            "source":            manifest.get("source_dataset"),
            "fingerprint_md5":   manifest.get("source_fingerprint_md5"),
            "split_sizes":       manifest.get("split_sizes"),
            "class_distribution": manifest.get("class_distribution"),
        },
        "label_mapping": {
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
            "label2id": LABEL2ID,
            "note": "0=FAKE, 1=REAL. Confirmed from unified_dataset.csv inspection. "
                    "Passed directly into HuggingFace model config.",
        },
        "token_length_stats": token_stats,
        "training": {
            "max_seq_len":               max_seq_len,
            "truncation_strategy":       "tail (end of article)",
            "truncation_assumption":     (
                "Inverted-pyramid assumption: key signals are front-loaded. "
                "NOT validated on VeriNews dataset — design assumption only."
            ),
            "random_seed":               RANDOM_SEED,
            "learning_rate":             args.lr,
            "weight_decay":              args.weight_decay,
            "num_epochs":                args.epochs,
            "per_device_train_batch":    phys_batch,
            "per_device_eval_batch":     args.eval_batch,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size":      phys_batch * grad_accum,
            "fp16":                      fp16,
            "warmup_ratio":              WARMUP_RATIO,
            "metric_for_best_model":     METRIC_FOR_BEST,
        },
        "smoke_test_note": (
            "SMOKE TEST — NOT A RESEARCH RESULT. "
            "Metrics from this run must not appear in model_comparison.md "
            "or research documentation."
        ) if smoke_test else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OOM handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_oom(phys_batch: int, grad_accum: int, max_seq_len: int) -> None:
    """
    Print a structured OOM recovery protocol and exit.

    OOM fallback preserves the intended effective batch size (16) by
    halving physical batch and doubling gradient accumulation steps.
    Reducing max_seq_len is NOT part of the automatic fallback — it is
    a meaningful experiment configuration change requiring an explicit
    decision before continuing.
    """
    new_batch   = max(1, phys_batch // 2)
    new_accum   = grad_accum * 2
    eff_current = phys_batch * grad_accum
    eff_new     = new_batch * new_accum

    print()
    print("=" * 70)
    print("  CUDA OUT OF MEMORY")
    print()
    print(f"  Failed config  : batch_size={phys_batch}, grad_accum={grad_accum}, "
          f"effective_batch={eff_current}, max_seq_len={max_seq_len}")
    print()
    if phys_batch > 4:
        print("  Recovery — Step 1 (preserves effective batch size):")
        print(f"    --batch-size {new_batch} --grad-accum {new_accum}")
        print(f"    (effective batch = {eff_new}; eval batch may also be halved)")
        print()
        print("  If OOM persists at batch_size=4, proceed to Step 2.")
    else:
        print("  Recovery — Step 2 (batch_size already at minimum):")
        print("  STOP this run and report the failure.")
        print("  Do NOT reduce max_seq_len automatically.")
        print("  Reducing max_seq_len from 512 to 384 is an experiment configuration")
        print("  change that requires an explicit decision before continuing.")
    print()
    print("  Do NOT silently change max_seq_len and continue.")
    print("=" * 70)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    smoke_test  = args.smoke_test
    max_seq_len = SMOKE_MAX_SEQ_LEN if smoke_test else args.max_seq_len
    phys_batch  = SMOKE_BATCH if smoke_test else args.batch_size
    grad_accum  = 1 if smoke_test else args.grad_accum   # no accumulation for smoke test
    fp16        = (USE_FP16_IF_CUDA and torch.cuda.is_available()) and not smoke_test
    # Compute these BEFORE the config print so the banner displays correctly
    epochs     = SMOKE_EPOCHS if smoke_test else args.epochs
    eval_batch = phys_batch if smoke_test else args.eval_batch

    if smoke_test:
        print(_SMOKE_BANNER)
    else:
        print("=" * 70)
        print("  VeriNews AI — Milestone 2: DistilBERT Fine-Tuning")
        print("=" * 70)

    # ── Device check ──────────────────────────────────────────────────────────
    device = get_device()
    cpu_full_train_guard(device, smoke_test)

    # ── Print configuration ───────────────────────────────────────────────────
    # Pass actual epochs (not args.epochs) so smoke-test shows 1, not 3
    args_display = argparse.Namespace(**vars(args))
    args_display.epochs = epochs
    print_training_config(device, max_seq_len, phys_batch, grad_accum, fp16, smoke_test, args_display)

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_seed(RANDOM_SEED)

    # ── Ensure splits exist ───────────────────────────────────────────────────
    if not SPLIT_MANIFEST.exists():
        print("[train] Splits not found — generating now …")
        prepare_splits()

    # ── Load data ─────────────────────────────────────────────────────────────
    if smoke_test:
        print(f"[train] SMOKE TEST: loading {SMOKE_SAMPLES} class-balanced samples …")
        train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
        val_df   = pd.read_csv(VAL_CSV,   encoding="utf-8")
        train_df = get_smoke_sample(train_df, n_per_class=SMOKE_SAMPLES // 2)
        val_df   = get_smoke_sample(val_df,   n_per_class=25)
    else:
        print(f"[train] Loading frozen splits …")
        train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8")
        val_df   = pd.read_csv(VAL_CSV,   encoding="utf-8")

    print(f"[train]   train  : {len(train_df):,} examples")
    print(f"[train]   val    : {len(val_df):,} examples")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\n[train] Loading tokenizer: {MODEL_CHECKPOINT} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("[train] Tokenising datasets …")
    train_dataset = NewsDataset(train_df, tokenizer, max_seq_len)
    val_dataset   = NewsDataset(val_df,   tokenizer, max_seq_len)

    # ── Model — label2id / id2label wired into HF model config ───────────────
    # These are passed as constructor arguments, not just Python constants.
    # The saved config.json will contain:
    #   "id2label": {"0": "FAKE", "1": "REAL"}
    #   "label2id": {"FAKE": 0, "REAL": 1}
    # This preserves label semantics for future inference and FastAPI integration.
    print(f"[train] Loading model: {MODEL_CHECKPOINT} …")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels    = 2,
        id2label      = {int(k): v for k, v in ID2LABEL.items()},
        label2id      = LABEL2ID,
    )

    # ── Output dirs ───────────────────────────────────────────────────────────
    checkpoint_dir = DISTILBERT_DIR / ("smoke_test" if smoke_test else "best")
    log_dir        = DISTILBERT_RESULTS / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    # epochs / eval_batch already computed above (before config printout)
    #
    # Deprecation notes (transformers 5.x):
    #   - warmup_ratio removed → compute warmup_steps from total optimizer steps
    #   - logging_dir removed → set TENSORBOARD_LOGGING_DIR env var instead
    total_steps  = (len(train_dataset) // (phys_batch * grad_accum)) * epochs
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(log_dir)

    training_args = TrainingArguments(
        output_dir                    = str(checkpoint_dir),
        num_train_epochs              = epochs,
        per_device_train_batch_size   = phys_batch,
        per_device_eval_batch_size    = eval_batch,
        gradient_accumulation_steps   = grad_accum,
        learning_rate                 = args.lr,
        weight_decay                  = args.weight_decay,
        warmup_steps                  = warmup_steps,
        fp16                          = fp16,
        eval_strategy                 = "epoch",
        save_strategy                 = "epoch",
        load_best_model_at_end        = True,
        metric_for_best_model         = "f1",
        greater_is_better             = True,
        seed                          = RANDOM_SEED,
        logging_steps                 = 25 if smoke_test else 100,
        report_to                     = "none",
        use_cpu                       = (str(device) == "cpu"),
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[train] Starting {'SMOKE TEST' if smoke_test else 'FULL'} training …")
    t0 = time.time()
    try:
        trainer.train()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
            handle_oom(phys_batch, grad_accum, max_seq_len)
        raise

    elapsed = time.time() - t0
    print(f"\n[train] Training completed in {elapsed:.1f}s")

    # ── Save model ────────────────────────────────────────────────────────────
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    print(f"[train] Checkpoint saved → {checkpoint_dir}")

    # Verify label mapping is in saved config
    saved_cfg_path = checkpoint_dir / "config.json"
    if saved_cfg_path.exists():
        with open(saved_cfg_path, encoding="utf-8") as f:
            saved_cfg = json.load(f)
        assert saved_cfg.get("id2label"), "id2label missing from saved config.json!"
        assert saved_cfg.get("label2id"), "label2id missing from saved config.json!"
        print(f"[train] Label mapping verified in config.json:")
        print(f"          id2label = {saved_cfg['id2label']}")
        print(f"          label2id = {saved_cfg['label2id']}")

    # ── Capture measured training speed from Trainer log history ─────────────
    # These are ACTUAL measured values from this run, not estimates.
    train_log = trainer.state.log_history
    # Extract last training log entry for speed metrics
    speed_entry = next(
        (e for e in reversed(train_log) if "train_samples_per_second" in e), {}
    )
    epoch_metrics = [
        {k: v for k, v in e.items()}
        for e in train_log if "eval_f1" in e
    ]

    measured_speed = {
        "total_wall_time_seconds": round(elapsed, 2),
        "train_samples_per_second": speed_entry.get("train_samples_per_second"),
        "train_steps_per_second":   speed_entry.get("train_steps_per_second"),
        "train_loss":               speed_entry.get("train_loss"),
        "note": "All timing values are measured from this actual run.",
    }

    # ── Save configs ──────────────────────────────────────────────────────────
    DISTILBERT_RESULTS.mkdir(parents=True, exist_ok=True)

    # training_config.json records the ACTUAL runtime configuration.
    # If an OOM fallback changed batch_size or grad_accum, those changed values
    # are recorded here — not the originally-requested values.
    training_config = {
        "smoke_test":                  smoke_test,
        "model_checkpoint":            MODEL_CHECKPOINT,
        "actual_max_seq_len":          max_seq_len,
        "actual_epochs":               epochs,
        "actual_per_device_train_batch": phys_batch,
        "actual_per_device_eval_batch":  eval_batch,
        "actual_gradient_accumulation":  grad_accum,
        "actual_effective_batch_size":   phys_batch * grad_accum,
        "actual_fp16":                   fp16,
        "learning_rate":                 args.lr,
        "weight_decay":                  args.weight_decay,
        "warmup_ratio":                  WARMUP_RATIO,
        "checkpoint_dir":                str(checkpoint_dir),
        "timing":                        measured_speed,
        "epoch_validation_metrics":      epoch_metrics,
    }
    with open(TRAINING_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    metadata = build_metadata(device, max_seq_len, phys_batch, grad_accum, fp16, smoke_test, args)
    # Overwrite training section with ACTUAL runtime values
    metadata["training"].update({
        "actual_max_seq_len":            max_seq_len,
        "actual_per_device_train_batch": phys_batch,
        "actual_per_device_eval_batch":  eval_batch,
        "actual_gradient_accumulation":  grad_accum,
        "actual_effective_batch_size":   phys_batch * grad_accum,
        "actual_fp16":                   fp16,
    })
    metadata["timing"] = measured_speed
    metadata["epoch_validation_metrics"] = epoch_metrics
    with open(EXPERIMENT_METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[train] training_config.json    → {TRAINING_CONFIG_JSON}")
    print(f"[train] experiment_metadata.json → {EXPERIMENT_METADATA_JSON}")

    # ── Print training speed summary ──────────────────────────────────────────
    print()
    print("  Training timing (measured):")
    print(f"    Wall time          : {elapsed:.1f}s")
    if measured_speed.get("train_samples_per_second"):
        print(f"    Samples/second     : {measured_speed['train_samples_per_second']:.2f}")
    if measured_speed.get("train_steps_per_second"):
        print(f"    Steps/second       : {measured_speed['train_steps_per_second']:.3f}")
    import torch as _torch
    if _torch.cuda.is_available():
        print(f"    GPU                : {_torch.cuda.get_device_name(0)}")

    # ── Print per-epoch validation F1 (non-smoke only) ───────────────────────
    if not smoke_test and epoch_metrics:
        print()
        print("  Validation F1 by epoch:")
        for em in epoch_metrics:
            ep  = em.get("epoch", "?")
            f1  = em.get("eval_f1", float("nan"))
            acc = em.get("eval_accuracy", float("nan"))
            print(f"    Epoch {ep}: F1={f1:.4f}  Acc={acc:.4f}")

    # ── Print final validation metrics ────────────────────────────────────────
    print()
    if smoke_test:
        print(_SMOKE_BANNER)
        print("  Final validation metrics (SMOKE TEST — NOT A RESEARCH RESULT):")
    else:
        print("  Final validation metrics (best checkpoint):")

    final_metrics = trainer.evaluate()
    for k, v in sorted(final_metrics.items()):
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    if smoke_test:
        print()
        print("  Smoke test complete. To run full training use a GPU environment.")
        print("  See: notebooks/train_distilbert_colab.ipynb")
        print(_SMOKE_BANNER)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DistilBERT for VeriNews AI (Milestone 2)."
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="End-to-end pipeline verification on 200 examples, 1 epoch (CPU-safe)."
    )
    p.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Maximum token sequence length for full training. "
             "Default=512 (evidence-based; see config.py for rationale)."
    )
    p.add_argument("--epochs",      type=int,   default=NUM_EPOCHS,
                   help="Training epochs (default: 3).")
    p.add_argument("--batch-size",  type=int,   default=TRAIN_BATCH,
                   help=f"Per-device train batch size (default: {TRAIN_BATCH}). "
                        "Conservative; increase only if GPU VRAM allows.")
    p.add_argument("--eval-batch",  type=int,   default=EVAL_BATCH,
                   help=f"Per-device eval batch size (default: {EVAL_BATCH}).")
    p.add_argument("--grad-accum",  type=int,   default=GRAD_ACCUM_STEPS,
                   help=f"Gradient accumulation steps (default: {GRAD_ACCUM_STEPS}). "
                        f"Effective batch = batch-size × grad-accum.")
    p.add_argument("--lr",          type=float, default=LEARNING_RATE,
                   help="Learning rate (default: 2e-5).")
    p.add_argument("--weight-decay",type=float, default=WEIGHT_DECAY,
                   help="AdamW weight decay (default: 0.01).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args)
