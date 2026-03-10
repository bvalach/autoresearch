"""
One-time data and model preparation for autoresearch-mlx-lora.

Downloads a small base model (Qwen3.5-0.8B) and prepares TinyStories
as train/val JSONL splits for mlx-lm LoRA fine-tuning.

Usage:
    uv run prepare.py                  # full prep (model + data)
    uv run prepare.py --model-only     # download model only
    uv run prepare.py --data-only      # download and prepare data only

Artifacts are stored in ~/.cache/autoresearch-mlx/.
"""

import os
import sys
import json
import math
import argparse

import mlx.core as mx

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_SAMPLES = 500       # number of validation samples for BPB evaluation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-mlx")
DATA_DIR = os.path.join(CACHE_DIR, "data")
MODEL_DIR = os.path.join(CACHE_DIR, "models")

# Base model: Qwen3.5-0.8B — small enough for LoRA on 24 GB unified memory
BASE_MODEL = "Qwen/Qwen3.5-0.8B"

# Dataset: TinyStories — low entropy, clean, good results with small models
DATASET_NAME = "karpathy/tinystories-gpt4-clean"

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_model():
    """Download and convert base model to MLX format."""
    model_path = os.path.join(MODEL_DIR, BASE_MODEL.replace("/", "--"))
    if os.path.exists(model_path) and os.listdir(model_path):
        print(f"Model: already downloaded at {model_path}")
        return model_path

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Model: downloading {BASE_MODEL} ...")
    from mlx_lm import convert
    # Convert from HuggingFace to MLX safetensors format
    convert(BASE_MODEL, mlx_path=model_path)
    print(f"Model: saved to {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data():
    """Download TinyStories and prepare train/val JSONL for mlx-lm."""
    train_path = os.path.join(DATA_DIR, "train.jsonl")
    val_path = os.path.join(DATA_DIR, "valid.jsonl")

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_lines = sum(1 for _ in open(train_path))
        val_lines = sum(1 for _ in open(val_path))
        print(f"Data: already prepared — {train_lines:,} train, {val_lines:,} val samples")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Data: downloading {DATASET_NAME} ...")

    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, split="train")

    # Shuffle with fixed seed and split 95/5
    ds = ds.shuffle(seed=42)
    n_val = max(EVAL_SAMPLES * 2, len(ds) // 20)  # at least 2x eval samples
    n_train = len(ds) - n_val

    print(f"Data: {n_train:,} train, {n_val:,} val samples")

    # Write JSONL — mlx-lm expects {"text": "..."} format
    with open(train_path, "w") as f:
        for i in range(n_train):
            json.dump({"text": ds[i]["text"]}, f, ensure_ascii=False)
            f.write("\n")

    with open(val_path, "w") as f:
        for i in range(n_train, len(ds)):
            json.dump({"text": ds[i]["text"]}, f, ensure_ascii=False)
            f.write("\n")

    print(f"Data: saved to {DATA_DIR}")


# ---------------------------------------------------------------------------
# Evaluation: bits per byte (BPB) — the fixed metric
# ---------------------------------------------------------------------------

def evaluate_bpb(model, tokenizer, data_path=None, max_samples=None):
    """
    Bits per byte (BPB): vocab-size-independent evaluation metric.

    For each sample, computes cross-entropy loss per token, sums the nats
    weighted by each token's byte length, then converts nats/byte to bits/byte.
    Special tokens (byte length 0) are excluded.

    This is the ground truth metric. Do not modify.
    """
    import mlx.nn as nn

    if data_path is None:
        data_path = os.path.join(DATA_DIR, "valid.jsonl")
    if max_samples is None:
        max_samples = EVAL_SAMPLES

    # Read validation data
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line)["text"])
            if len(samples) >= max_samples:
                break

    total_nats = 0.0
    total_bytes = 0

    for text in samples:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue

        # Tokenize and run forward pass
        input_ids = mx.array(tokens[:-1])[None, :]  # (1, T-1)
        target_ids = mx.array(tokens[1:])            # (T-1,)

        logits = model(input_ids)                    # (1, T-1, vocab)
        logits = logits.squeeze(0)                   # (T-1, vocab)

        # Per-token cross-entropy in nats (positive values)
        losses = nn.losses.cross_entropy(logits, target_ids, reduction="none")

        # Byte length per target token
        for i, tid in enumerate(tokens[1:]):
            decoded = tokenizer.decode([tid])
            nbytes = len(decoded.encode("utf-8"))
            if nbytes > 0:
                total_nats += losses[i].item()
                total_bytes += nbytes

    if total_bytes == 0:
        return float("inf")

    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_model_path():
    """Return path to the downloaded MLX model."""
    return os.path.join(MODEL_DIR, BASE_MODEL.replace("/", "--"))

def get_data_dir():
    """Return path to the prepared data directory."""
    return DATA_DIR


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare model and data for autoresearch-mlx-lora")
    parser.add_argument("--model-only", action="store_true", help="Only download the base model")
    parser.add_argument("--data-only", action="store_true", help="Only download and prepare data")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    if not args.data_only:
        download_model()
        print()

    if not args.model_only:
        prepare_data()
        print()

    print("Done! Ready to train.")
